
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v16-ctc")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v16", create_if_missing=True)
v14_vol  = modal.Volume.from_name("bt-checkpoints-v14", create_if_missing=False)
v13_vol  = modal.Volume.from_name("bt-checkpoints-v13", create_if_missing=False)
v20_vol  = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=False)
v50_vol  = modal.Volume.from_name("bt-checkpoints-v50", create_if_missing=True)
v12_vol  = modal.Volume.from_name("bt-checkpoints-v12", create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

def _download_models():
    from transformers import VideoMAEModel, AutoModelForCausalLM, AutoTokenizer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0", "peft>=0.12",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient", "qwen-vl-utils",
        "editdistance", "sentencepiece", "openneuro-py",
    ])
    .run_function(_download_models)
)

# ═══════════════════════════════════════════════════════════════════════════════
# V16: frozen lexical support training on top of the best V14 decoder.
#
# What failed in V15:
#   [1] lexical reranking itself was reasonable
#   [2] unfreezing Stage B damaged the proven V14 decoder regime
#
# What worked:
#   [1] V14 best gave the strongest diversity / word-fragment base
#   [2] lexical evidence can shape candidates without blank collapse
#
# This branch keeps the V14 encoder and CTC decoder fixed and trains only
# support heads:
#   [1] ngram inventory
#   [2] word inventory
#   [3] short-phrase inventory
#   [4] word-count head
# The intent is to improve candidate reranking without ever moving the base CTC
# path away from the best V14 checkpoint.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ───────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
STOPS      = set("bcdgkpt")
FRICATIVES = set("fhjsvxz")
SONORANTS  = set("lmnrwy")
CONS_LIST = sorted(CONSONANTS)
CONS_TO_ID = {c: i for i, c in enumerate(CONS_LIST)}
NGRAM_LIST = []
NGRAM_TO_ID = {}
WORD_LIST = []
WORD_TO_ID = {}
PHRASE_LIST = []
PHRASE_TO_ID = {}
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "were", "have", "into",
    "will", "than", "both", "been", "they", "their", "them", "then", "there",
    "when", "what", "your", "about", "would", "could", "should", "into", "over",
    "under", "after", "before", "while", "where", "which", "whose", "because",
}

CLASS_BLANK = 0
CLASS_VOWEL = 1
CLASS_STOP = 2
CLASS_FRICATIVE = 3
CLASS_SONORANT = 4
CLASS_DIGIT = 5
CLASS_OTHER = 6
CLASS_SPACE = 7
CLASS_VOCAB_SIZE = 8

NUM_EEG_BANDS    = 6
PHASE1_MIX_EPOCHS = 6

def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]


def char_to_class_id(ch):
    if ch == " ":
        return CLASS_SPACE
    if ch in VOWELS:
        return CLASS_VOWEL
    if ch in STOPS:
        return CLASS_STOP
    if ch in FRICATIVES:
        return CLASS_FRICATIVE
    if ch in SONORANTS:
        return CLASS_SONORANT
    if ch in CONSONANTS:
        return CLASS_OTHER
    if ch.isdigit():
        return CLASS_DIGIT
    return CLASS_OTHER


def text_to_class_ids(text):
    class_ids = []
    prev = None
    for c in text.lower():
        if c not in CHAR_TO_ID:
            continue
        cls = char_to_class_id(c)
        if cls != prev:
            class_ids.append(cls)
            prev = cls
    return class_ids


def text_to_cons_target(text):
    target = np.zeros(len(CONS_LIST), dtype=np.float32)
    for c in set(text.lower()):
        if c in CONS_TO_ID:
            target[CONS_TO_ID[c]] = 1.0
    return target


def normalize_text(text):
    text = text.lower()
    return "".join(c if c in CHAR_TO_ID and c != "_" else " " for c in text)


def build_ngram_vocab(texts, max_features=256):
    from collections import Counter

    counter = Counter()
    for text in texts:
        for word in normalize_text(text).split():
            if len(word) < 2:
                continue
            for n in (2, 3):
                if len(word) < n:
                    continue
                for i in range(len(word) - n + 1):
                    ng = word[i:i + n]
                    counter[ng] += 1

    top = [ng for ng, _ in counter.most_common(max_features)]
    return top, {ng: i for i, ng in enumerate(top)}


def text_to_ngram_target(text):
    target = np.zeros(len(NGRAM_LIST), dtype=np.float32)
    if not NGRAM_LIST:
        return target
    for word in normalize_text(text).split():
        if len(word) < 2:
            continue
        seen = set()
        for n in (2, 3):
            if len(word) < n:
                continue
            for i in range(len(word) - n + 1):
                ng = word[i:i + n]
                idx = NGRAM_TO_ID.get(ng)
                if idx is not None and idx not in seen:
                    target[idx] = 1.0
                    seen.add(idx)
    return target


def build_word_vocab(texts, max_features=512):
    from collections import Counter

    counter = Counter()
    for text in texts:
        for word in normalize_text(text).split():
            if len(word) >= 3:
                counter[word] += 1
    top = [w for w, _ in counter.most_common(max_features)]
    return top, {w: i for i, w in enumerate(top)}


def build_phrase_vocab(texts, max_features=384):
    from collections import Counter

    counter = Counter()
    for text in texts:
        words = [w for w in normalize_text(text).split() if len(w) >= 2]
        for n in (2, 3):
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                counter[phrase] += 1
    top = [p for p, _ in counter.most_common(max_features)]
    return top, {p: i for i, p in enumerate(top)}


def text_to_word_target(text):
    target = np.zeros(len(WORD_LIST), dtype=np.float32)
    if not WORD_LIST:
        return target
    for word in set(normalize_text(text).split()):
        idx = WORD_TO_ID.get(word)
        if idx is not None:
            target[idx] = 1.0
    return target


def text_to_word_count(text):
    return np.float32(len(normalize_text(text).split()))


def text_to_phrase_target(text):
    target = np.zeros(len(PHRASE_LIST), dtype=np.float32)
    if not PHRASE_LIST:
        return target
    words = [w for w in normalize_text(text).split() if len(w) >= 2]
    seen = set()
    for n in (2, 3):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            idx = PHRASE_TO_ID.get(phrase)
            if idx is not None and idx not in seen:
                target[idx] = 1.0
                seen.add(idx)
    return target


class CharNGramLM:
    def __init__(self, texts, order=3):
        from collections import Counter, defaultdict

        self.order = order
        self.vocab = [c for c in CHAR_VOCAB if c != "_"]
        self.vocab_size = len(self.vocab)
        self.unigrams = Counter()
        self.context_counts = defaultdict(Counter)

        for text in texts:
            norm = f"  {normalize_text(text)} "
            for ch in norm:
                if ch in CHAR_TO_ID and ch != "_":
                    self.unigrams[ch] += 1
            for i in range(len(norm)):
                ch = norm[i]
                if ch not in CHAR_TO_ID or ch == "_":
                    continue
                for ctx_len in range(1, order):
                    if i - ctx_len < 0:
                        continue
                    ctx = norm[i - ctx_len:i]
                    if any(c not in CHAR_TO_ID or c == "_" for c in ctx):
                        continue
                    self.context_counts[ctx][ch] += 1

        self.total_unigrams = sum(self.unigrams.values())

    def log_prob(self, history, ch):
        if ch not in CHAR_TO_ID or ch == "_":
            return -8.0
        history = "".join(c for c in history[-(self.order - 1):] if c in CHAR_TO_ID and c != "_")
        for ctx_len in range(min(len(history), self.order - 1), 0, -1):
            ctx = history[-ctx_len:]
            counts = self.context_counts.get(ctx)
            if counts:
                total = sum(counts.values())
                count = counts.get(ch, 0)
                return float(np.log((count + 0.2) / (total + 0.2 * self.vocab_size)))
        uni = self.unigrams.get(ch, 0)
        return float(np.log((uni + 0.2) / (self.total_unigrams + 0.2 * self.vocab_size)))


def _extract_candidate_features(text):
    norm = normalize_text(text)
    words = [w for w in norm.split() if w]
    ngrams = []
    phrases = []
    consonants = set()
    for word in words:
        for ch in word:
            if ch in CONS_TO_ID:
                consonants.add(ch)
        for n in (2, 3):
            if len(word) < n:
                continue
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i + n])
    for n in (2, 3):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i:i + n]))
    return words, ngrams, phrases, consonants


def rerank_candidates_with_evidence(candidates, word_probs, ngram_probs, phrase_probs, cons_probs, count_pred):
    from collections import Counter

    reranked = []
    count_pred = float(count_pred)
    for text, base_score in candidates:
        words, ngrams, phrases, consonants = _extract_candidate_features(text)

        word_support = 0.0
        unsupported = 0.0
        for word in words:
            idx = WORD_TO_ID.get(word)
            if idx is not None:
                prob = float(word_probs[idx])
                word_support += prob * (0.75 if word in STOPWORDS else 1.0)
                if prob < 0.35:
                    unsupported += (0.35 - prob) * (0.4 if word in STOPWORDS else 1.0)
            elif len(word) >= 3:
                unsupported += 0.08 if word in STOPWORDS else 0.14

        ngram_support = 0.0
        seen_ng = set()
        for ng in ngrams:
            idx = NGRAM_TO_ID.get(ng)
            if idx is not None and idx not in seen_ng:
                ngram_support += float(ngram_probs[idx])
                seen_ng.add(idx)

        phrase_support = 0.0
        phrase_penalty = 0.0
        seen_phrase = set()
        for phrase in phrases:
            idx = PHRASE_TO_ID.get(phrase)
            if idx is not None and idx not in seen_phrase:
                prob = float(phrase_probs[idx])
                phrase_support += prob
                if prob < 0.35:
                    phrase_penalty += (0.35 - prob)
                seen_phrase.add(idx)
            elif phrase:
                phrase_penalty += 0.05

        cons_support = 0.0
        for ch in consonants:
            cons_support += float(cons_probs[CONS_TO_ID[ch]])

        word_count = len(words)
        count_penalty = abs(word_count - count_pred)
        repeats = sum(max(0, n - 1) for w, n in Counter(words).items() if w in STOPWORDS or len(w) <= 3)

        score = (
            float(base_score)
            + 0.70 * word_support
            + 0.45 * ngram_support
            + 0.80 * phrase_support
            + 0.22 * cons_support
            - 0.65 * unsupported
            - 0.35 * phrase_penalty
            - 0.20 * count_penalty
            - 0.22 * repeats
        )
        reranked.append((text, score))

    reranked.sort(key=lambda x: -x[1])
    return reranked


def _common_prefix_ratio(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


def _select_diverse_beams(candidates, keep, lm_weight, diversity_weight):
    if len(candidates) <= keep:
        return candidates

    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < keep:
        best_idx = 0
        best_score = None
        for idx, cand in enumerate(remaining):
            _, text, _, ctc_score, lm_score = cand
            combined = ctc_score + lm_weight * lm_score
            if selected:
                overlap = max(_common_prefix_ratio(text, s[1]) for s in selected)
            else:
                overlap = 0.0
            score = combined - diversity_weight * overlap
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected

def greedy_ctc_decode(logits):
    """logits: (T, B, Vocab) → list[str]"""
    ids = logits.argmax(dim=-1)
    results = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out = ""; prev = -1
        for i in seq:
            if i != prev and i != 0:
                out += ID_TO_CHAR.get(i, "")
            prev = i
        results.append(out)
    return results

def beam_ctc_decode(logits, beam_width=5):
    """logits: (T, B, Vocab) → list[str] using beam search."""
    return [beams[0][0] if beams else "" for beams in beam_ctc_decode_nbest(
        logits, beam_width=beam_width, nbest=1)]


def beam_ctc_decode_nbest(logits, beam_width=12, nbest=5, blank_penalty=0.20,
                         char_lm=None, lm_weight=0.35, diversity_weight=0.16):
    """logits: (T, B, Vocab) → list[list[(text, norm_score)]]."""
    log_probs = F.log_softmax(logits, dim=-1)
    T, B, V = log_probs.shape
    results = []
    for b in range(B):
        beams = [([], "", -1, 0.0, 0.0)]
        for t in range(T):
            new_beams = {}
            frame_lp = log_probs[t, b].clone()
            frame_lp[0] -= blank_penalty
            topk_vals, topk_ids = frame_lp.topk(min(beam_width * 3, V))
            for seq, text, last_tok, ctc_score, lm_score in beams:
                for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_ctc_score = ctc_score + val
                    if tok_id == 0:
                        key = tuple(seq)
                        existing = new_beams.get(key)
                        cand = (seq, text, 0, new_ctc_score, lm_score)
                        cand_score = new_ctc_score + lm_weight * lm_score
                        if existing is None or (existing[3] + lm_weight * existing[4]) < cand_score:
                            new_beams[key] = cand
                    elif tok_id == last_tok:
                        key = tuple(seq)
                        existing = new_beams.get(key)
                        cand = (seq, text, tok_id, new_ctc_score, lm_score)
                        cand_score = new_ctc_score + lm_weight * lm_score
                        if existing is None or (existing[3] + lm_weight * existing[4]) < cand_score:
                            new_beams[key] = cand
                    else:
                        new_seq = seq + [tok_id]
                        ch = ID_TO_CHAR.get(tok_id, "")
                        new_text = text + ch
                        new_lm_score = lm_score + (char_lm.log_prob(text, ch) if char_lm is not None else 0.0)
                        key = tuple(new_seq)
                        existing = new_beams.get(key)
                        cand = (new_seq, new_text, tok_id, new_ctc_score, new_lm_score)
                        cand_score = new_ctc_score + lm_weight * new_lm_score
                        if existing is None or (existing[3] + lm_weight * existing[4]) < cand_score:
                            new_beams[key] = cand
            candidates = sorted(
                new_beams.values(),
                key=lambda x: -(x[3] + lm_weight * x[4])
            )[:max(beam_width * 2, nbest)]
            beams = _select_diverse_beams(candidates, max(beam_width, nbest), lm_weight, diversity_weight)
        beams = _select_diverse_beams(beams, nbest, lm_weight, diversity_weight)
        if not beams:
            results.append([("", 1.0)])
            continue
        scores = torch.tensor([beam[3] + lm_weight * beam[4] for beam in beams], dtype=torch.float32)
        norm_scores = torch.softmax(scores - scores.max(), dim=0).tolist()
        decoded = []
        for (_, text, _, _, _), score in zip(beams, norm_scores):
            decoded.append((text, float(score)))
        results.append(decoded)
    return results


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)

def compute_wer(pred, ref):
    import editdistance
    pred_w = pred.strip().split()
    ref_w  = ref.strip().split()
    if not ref_w: return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


# ─── EEG Augmentation ─────────────────────────────────────────────────────────

def augment_eeg(eeg_np, strength='moderate'):
    eeg = eeg_np.copy()
    T, C = eeg.shape
    if strength == 'moderate':
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (64, 0.85, 1.15, 0.05, 0.10, 2)
    else:
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (16, 0.95, 1.05, 0.02, 0.05, 1)

    shift = np.random.randint(-jitter, jitter + 1)
    if shift != 0:
        eeg = np.roll(eeg, shift, axis=0)
        if shift > 0: eeg[:shift] = 0
        else: eeg[shift:] = 0

    scales = np.random.uniform(amp_lo, amp_hi, size=(1, C)).astype(np.float32)
    eeg = eeg * scales
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std

    n_drop = int(C * ch_drop_p)
    if n_drop > 0:
        drop_idx = np.random.choice(C, n_drop, replace=False)
        eeg[:, drop_idx] = 0

    for _ in range(mask_spans):
        span_len = np.random.randint(16, 33)
        start = np.random.randint(0, max(1, T - span_len))
        eeg[start:start+span_len] = 0

    return eeg


# ─── Memory Bank ──────────────────────────────────────────────────────────────

class MemoryBank:
    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr  = 0
        self.n    = 0

    def enqueue(self, emb):
        emb = emb.detach().cpu().float()
        B = emb.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B) % self.size
        self.bank[idx] = emb
        self.ptr = (self.ptr + B) % self.size
        self.n = min(self.n + B, self.size)

    def get(self, device):
        if self.n < 2: return None
        return self.bank[:self.n].to(device)


# ─── Modules ──────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, Ch = x.shape
        x_split = x.reshape(B * T, self.n_bands, self.n_electrodes)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        out = self.conv(imgs.view(B * T, C, H, W))
        return out.view(B, T, 3, H, W)


class CrossAttentionBridge(nn.Module):
    def __init__(self, in_ch=768, out_ch=1024, n_latents=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, in_ch))
        self.mha     = nn.MultiheadAttention(in_ch, num_heads=16, batch_first=True)
        self.ln_k    = nn.LayerNorm(in_ch)
        self.ln_q    = nn.LayerNorm(in_ch)
        self.proj    = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)


# ─── Model ────────────────────────────────────────────────────────────────────

class EEG_CTC_V16(nn.Module):
    def __init__(self, ngram_dim, word_dim, phrase_dim):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt = ChannelAdapter(NUM_EEG_BANDS)
        self.video_enc = VideoMAEModel(v_cfg)
        self.ctc_head = nn.Linear(768, VOCAB_SIZE)
        self.class_ctc_head = nn.Linear(768, CLASS_VOCAB_SIZE)
        self.cons_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, len(CONS_LIST)),
        )
        self.ngram_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, ngram_dim),
        )
        self.word_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, word_dim),
        )
        self.phrase_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, phrase_dim),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1),
        )
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.class_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.cons_loss_fn = nn.BCEWithLogitsLoss()
        self.ngram_loss_fn = nn.BCEWithLogitsLoss()
        self.word_loss_fn = nn.BCEWithLogitsLoss()
        self.phrase_loss_fn = nn.BCEWithLogitsLoss()
        self.count_loss_fn = nn.SmoothL1Loss()
        self.char_lm = None

    def encode(self, traj):
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    def forward(self, traj, char_ids, char_lens, class_ids, class_lens,
                cons_targets, ngram_targets, word_targets, phrase_targets, word_counts):
        B = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        class_logits = self.class_ctc_head(v_seq).transpose(0, 1)
        pooled = v_seq.mean(dim=1)
        cons_logits = self.cons_head(pooled)
        ngram_logits = self.ngram_head(pooled)
        word_logits = self.word_head(pooled)
        phrase_logits = self.phrase_head(pooled)
        count_pred = F.softplus(self.count_head(pooled).squeeze(-1))
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)

        loss_ctc = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)
        loss_class = self.class_loss_fn(
            F.log_softmax(class_logits, dim=-1), class_ids, input_lens, class_lens)
        loss_cons = self.cons_loss_fn(cons_logits, cons_targets)
        loss_ngram = self.ngram_loss_fn(ngram_logits, ngram_targets)
        loss_word = self.word_loss_fn(word_logits, word_targets)
        loss_phrase = self.phrase_loss_fn(phrase_logits, phrase_targets)
        loss_count = self.count_loss_fn(count_pred, word_counts)

        probs = F.softmax(ctc_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div = -ent.mean()

        return {
            "loss_ctc": loss_ctc,
            "loss_class": loss_class,
            "loss_cons": loss_cons,
            "loss_ngram": loss_ngram,
            "loss_word": loss_word,
            "loss_phrase": loss_phrase,
            "loss_count": loss_count,
            "loss_div": loss_div,
        }

    @torch.no_grad()
    def decode_ctc(self, traj, beam_width=16, nbest=5):
        traj = traj.to(next(self.parameters()).dtype)
        v_seq = self.encode(traj)
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        pooled = v_seq.mean(dim=1)
        cons_probs = torch.sigmoid(self.cons_head(pooled)).float().cpu().numpy()
        ngram_probs = torch.sigmoid(self.ngram_head(pooled)).float().cpu().numpy()
        word_probs = torch.sigmoid(self.word_head(pooled)).float().cpu().numpy()
        phrase_probs = torch.sigmoid(self.phrase_head(pooled)).float().cpu().numpy()
        count_pred = F.softplus(self.count_head(pooled).squeeze(-1)).float().cpu().tolist()
        beam_groups = beam_ctc_decode_nbest(
            ctc_logits, beam_width=beam_width, nbest=max(nbest, 6), char_lm=self.char_lm)
        reranked_groups = []
        for beams, wp, ngp, php, cp, cnt in zip(beam_groups, word_probs, ngram_probs, phrase_probs, cons_probs, count_pred):
            reranked = rerank_candidates_with_evidence(beams, wp, ngp, php, cp, cnt)[:nbest]
            if not reranked:
                reranked_groups.append([("", 1.0)])
                continue
            score_tensor = torch.tensor([s for _, s in reranked], dtype=torch.float32)
            probs = torch.softmax(score_tensor - score_tensor.max(), dim=0).tolist()
            reranked_groups.append([(text, float(prob)) for (text, _), prob in zip(reranked, probs)])
        return reranked_groups


# ─── Pretrained weight transfer ───────────────────────────────────────────────

def load_pretrained_videomae_encoder(video_enc):
    from transformers import VideoMAEModel as _VMAE
    print("Loading pretrained VideoMAE-base encoder weights...")
    src    = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    src_sd = src.state_dict()
    dst_sd = video_enc.state_dict()
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone(); n_copied += 1
        else:
            n_skipped += 1
    video_enc.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped} "
          f"(patch emb + pos emb — re-init from scratch)")
    del src; torch.cuda.empty_cache()


def load_v14_phase1_warmstart(model, ckpt_path="/v14/v14_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V14 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V14 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = (
        "rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.",
        "class_ctc_head.", "cons_head.", "ngram_head."
    )
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V14 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_v13_phase1_warmstart(model, ckpt_path="/v13/v13_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V13 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V13 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "class_ctc_head.", "cons_head.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V13 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_v50_phase1_warmstart(model, ckpt_path="/v50/v50_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V50 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V50 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V50 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_v20_phase1_warmstart(model, ckpt_path="/v20/v20_phase1_best.pt"):
    """Load the proven V20 stack, expanding the old 4→3 adapter into 6→3."""
    if not Path(ckpt_path).exists():
        print(f"V20 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V20 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k == "ch_adapt.conv.weight" and k in dst_sd:
            dst = dst_sd[k]
            if v.shape[0] == dst.shape[0] and v.shape[1] == 4 and dst.shape[1] == 6:
                expanded = torch.zeros_like(dst)
                expanded[:, :4] = v
                dst_sd[k] = expanded
                n_copied += 1
                continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V20 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


# ─── Data loading ─────────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False

def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    inner_sp = base_path / "InnerSpeech"
    for d in [zuco_v1, zuco_v2, inner_sp]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        res   = __import__('subprocess').run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if not verify_mat_file(local):
                remote = next((p for p in paths if p.strip().endswith(name)), None)
                if remote:
                    print(f"  Fetching {name}...")
                    __import__('subprocess').run(
                        ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                        timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat","resultsZDM_SR.mat",
                               "resultsZAB_NR.mat","resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat","resultsYAG_NR.mat","resultsYAK_NR.mat"])

    if not any(inner_sp.rglob("*-epo.fif")):
        import openneuro
        print("  Fetching InnerSpeech ds003626 derivatives/sub-01 ...")
        openneuro.download(dataset="ds003626", target_dir=str(inner_sp),
                           include=["derivatives/sub-01/"])
    vol.commit()

def load_mat_any(path):
    import scipy.io as sio, h5py
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data['sentenceData']
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for s in sentences:
            if hasattr(s,'rawData') and hasattr(s,'content') and isinstance(s.rawData, np.ndarray):
                yield s.rawData, str(s.content)
    except:
        try:
            with h5py.File(path, 'r') as f:
                if 'sentenceData' in f:
                    for key in f['sentenceData'].keys():
                        s = f['sentenceData'][key]
                        if 'rawData' in s and 'content' in s:
                            yield np.array(s['rawData']).T, "hdf5_content_placeholder"
        except: pass

def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch > t: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch:   eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:   eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    mu  = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt

    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo/64.0, hi/64.0], btype='bandpass')
            env  = np.abs(filtfilt(b, a, data, axis=1))
            return np.convolve(env.flatten(), np.ones(5)/5, mode='same').reshape(env.shape)
        except:
            return np.zeros_like(data)

    alpha = band_env(8,  13,  eeg)
    beta  = band_env(13, 30,  eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gamma_delta = np.diff(gamma, axis=1, prepend=gamma[:, :1])

    def rezscore(x):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - mu) / std

    delta = rezscore(delta)
    gamma_delta = rezscore(gamma_delta)

    combined = np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, delta.T, gamma_delta.T], axis=1
    )  # (1024, 384)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, augment='none',
                 include_zuco=True, include_innerspeech=False):
        self.samples = []
        self.augment = augment
        p = Path(base_path)
        skipped = 0
        zuco_count = 0
        inner_count = 0

        if include_zuco:
            for zp in [p/"ZuCo_v1", p/"ZuCo_v2"]:
                for mat in zp.rglob("*.mat"):
                    for raw, text in load_mat_any(mat):
                        if not text or "placeholder" in text.lower():
                            skipped += 1
                            continue
                        normed = normalize_eeg(raw)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': text})
                            zuco_count += 1

        if include_innerspeech:
            import mne
            for fif in (p/"InnerSpeech").rglob("*-epo.fif"):
                try:
                    epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
                    epochs.resample(128)
                    data = epochs.get_data()
                    labels = (
                        epochs.metadata['condition'].tolist()
                        if epochs.metadata is not None and 'condition' in epochs.metadata.columns
                        else [f"word_{i}" for i in range(len(data))]
                    )
                    for d, label in zip(data, labels):
                        normed = normalize_eeg(d)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': str(label)})
                            inner_count += 1
                except:
                    continue

        print(
            f"[Dataset] total={len(self.samples)} zuco={zuco_count} inner={inner_count} "
            f"skipped={skipped} augment={augment}"
        )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        eeg = s['eeg']
        if self.augment != 'none':
            eeg = augment_eeg(eeg, strength=self.augment)
        char_ids = text_to_char_ids(s['text'])
        class_ids = text_to_class_ids(s['text'])
        cons_target = text_to_cons_target(s['text'])
        ngram_target = text_to_ngram_target(s['text'])
        word_target = text_to_word_target(s['text'])
        phrase_target = text_to_phrase_target(s['text'])
        word_count = text_to_word_count(s['text'])
        return eeg, s['text'], char_ids, class_ids, cons_target, ngram_target, word_target, phrase_target, word_count


def collate_fn(batch):
    eegs, texts, char_ids_list, class_ids_list, cons_targets, ngram_targets, word_targets, phrase_targets, word_counts = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(c) for c in char_ids_list], dtype=torch.long)
    max_chars = max(char_lens) if max(char_lens) > 0 else 1
    char_pad  = torch.zeros(len(char_ids_list), max_chars, dtype=torch.long)
    for i, c in enumerate(char_ids_list):
        if len(c) > 0:
            char_pad[i, :len(c)] = torch.tensor(c)
    class_lens = torch.tensor([len(c) for c in class_ids_list], dtype=torch.long)
    max_classes = max(class_lens) if max(class_lens) > 0 else 1
    class_pad = torch.zeros(len(class_ids_list), max_classes, dtype=torch.long)
    for i, c in enumerate(class_ids_list):
        if len(c) > 0:
            class_pad[i, :len(c)] = torch.tensor(c)
    cons_tensor = torch.from_numpy(np.stack(cons_targets))
    ngram_tensor = torch.from_numpy(np.stack(ngram_targets))
    word_tensor = torch.from_numpy(np.stack(word_targets))
    phrase_tensor = torch.from_numpy(np.stack(phrase_targets))
    count_tensor = torch.tensor(word_counts, dtype=torch.float32)
    return eegs, list(texts), char_pad, char_lens, class_pad, class_lens, cons_tensor, ngram_tensor, word_tensor, phrase_tensor, count_tensor


# ─── Hint builders ────────────────────────────────────────────────────────────

def build_ctc_hint_batch(model, traj, tokenizer, device):
    with torch.no_grad():
        v_seq      = model.encode(traj)
        ctc_logits = model.ctc_head(v_seq).transpose(0, 1)
        ctc_texts  = beam_ctc_decode(ctc_logits, beam_width=5)
    prompts = []
    for hint in ctc_texts:
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_hint_batch(texts, tokenizer, device, corruption_rate=0.3):
    prompts = []
    for text in texts:
        words     = text.split()
        corrupted = []
        for w in words:
            r = np.random.random()
            if r < corruption_rate:                pass
            elif r < corruption_rate * 1.3:        corrupted.append('...')
            else:                                  corrupted.append(w)
        hint = ' '.join(corrupted) if corrupted else '...'
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# ─── Training loop ────────────────────────────────────────────────────────────

ACCUM_STEPS = 8


def set_phase1_trainable(model, freeze_encoder):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.cons_head.parameters():
        p.requires_grad = True
    for p in model.ngram_head.parameters():
        p.requires_grad = True
    for p in model.word_head.parameters():
        p.requires_grad = True
    for p in model.phrase_head.parameters():
        p.requires_grad = True
    for p in model.count_head.parameters():
        p.requires_grad = True


def make_phase1_optimizer(model, steps_per_epoch, epochs, freeze_encoder):
    set_phase1_trainable(model, freeze_encoder)
    groups = []

    def add_group(params, lr, name):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.01})

    add_group(model.cons_head.parameters(), 8e-5, "cons")
    add_group(model.ngram_head.parameters(), 1e-4, "ngram")
    add_group(model.word_head.parameters(), 1e-4, "word")
    add_group(model.phrase_head.parameters(), 1e-4, "phrase")
    add_group(model.count_head.parameters(), 7e-5, "count")

    printable = ", ".join(
        f"{g['name']}:{sum(p.numel() for p in g['params'])/1e6:.1f}M@{g['lr']:.0e}"
        for g in groups
    )
    optimizer = torch.optim.AdamW([
        {"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]}
        for g in groups
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
    )
    return optimizer, scheduler, printable


def _beam_diversity(texts):
    if len(texts) < 2:
        return 0.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(compute_cer(texts[i], texts[j]))
    return float(np.mean(scores)) if scores else 0.0


def _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_val_cer):
    device = next(model.parameters()).device
    model.eval()
    cer_scores, wer_scores = [], []
    oracle_cer_scores, oracle_wer_scores, diversity_scores = [], [], []
    vowel_correct = vowel_total = cons_correct = cons_total = 0
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for batch in val_loader:
            eeg_v, texts_v, _, _, _, _, _, _, _, _, _ = batch
            beam_groups = model.decode_ctc(eeg_v.to(device).to(torch.bfloat16), beam_width=16, nbest=5)
            for beams, ref in zip(beam_groups, texts_v):
                ref_lower = ref.lower()
                beam_texts = [(b[0] or "").strip() for b in beams]
                top = beam_texts[0] if beam_texts else ""
                oracle = min(beam_texts, key=lambda x: (compute_wer(x, ref_lower), compute_cer(x, ref_lower))) if beam_texts else ""
                cer_scores.append(compute_cer(top, ref_lower))
                wer_scores.append(compute_wer(top, ref_lower))
                oracle_cer_scores.append(compute_cer(oracle, ref_lower))
                oracle_wer_scores.append(compute_wer(oracle, ref_lower))
                diversity_scores.append(_beam_diversity(beam_texts))
                pred_chars = set(top)
                ref_chars = set(ref_lower)
                for c in ref_chars:
                    if c in VOWELS:
                        vowel_total += 1
                        if c in pred_chars:
                            vowel_correct += 1
                    elif c in CONSONANTS:
                        cons_total += 1
                        if c in pred_chars:
                            cons_correct += 1
                if len(ref.split()) > 3 or shown < 2:
                    print(f"  REF: '{ref[:100]}'")
                    print(f"  TOP1: '{top[:100]}'  CER={cer_scores[-1]:.2f} WER={wer_scores[-1]:.2f}")
                    if len(beam_texts) > 1:
                        print(f"  TOP2: '{beam_texts[1][:100]}'")
                    if len(beam_texts) > 2:
                        print(f"  TOP3: '{beam_texts[2][:100]}'")
                    shown += 1
                if shown >= 4:
                    break
            if shown >= 4:
                break

    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    if cer_scores:
        v_acc = vowel_correct / max(vowel_total, 1)
        c_acc = cons_correct / max(cons_total, 1)
        print(f"  Mean CER: {mean_cer:.3f}  Mean WER: {np.mean(wer_scores):.3f}")
        print(f"  Oracle CER: {np.mean(oracle_cer_scores):.3f}  Oracle WER: {np.mean(oracle_wer_scores):.3f}")
        print(f"  Beam diversity: {np.mean(diversity_scores):.3f}")
        print(f"  Vowel recall: {v_acc:.2f} ({vowel_correct}/{vowel_total})  "
              f"Consonant recall: {c_acc:.2f} ({cons_correct}/{cons_total})")

    torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_ep{epoch}.pt")
    ckpt_vol.commit()
    if mean_cer < best_val_cer:
        best_val_cer = mean_cer
        torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
        print(f"  ✓ New best val CER {best_val_cer:.3f}")
    return best_val_cer


def _train_phase1_stage(model, epochs, train_loader, val_loader, ckpt_prefix,
                        epoch_offset=0, best_train_ctc=float("inf"),
                        best_val_cer=float("inf"), freeze_encoder=False,
                        stage_name=""):
    device = next(model.parameters()).device
    optimizer, scheduler, groups = make_phase1_optimizer(
        model, len(train_loader), epochs, freeze_encoder=freeze_encoder)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"── Phase 1: {stage_name} ──")
    print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    print(f"  Phase 1 groups: {groups}")

    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        model.train()
        tot_ctc = tot_class = tot_cons = tot_ngram = tot_word = tot_phrase = tot_count = tot_div = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eeg, _, c_ids, c_lens, class_ids, class_lens, cons_targets, ngram_targets, word_targets, phrase_targets, word_counts = batch
            eeg = eeg.to(device).to(torch.bfloat16)
            c_ids = c_ids.to(device)
            c_lens = c_lens.to(device)
            class_ids = class_ids.to(device)
            class_lens = class_lens.to(device)
            cons_targets = cons_targets.to(device)
            ngram_targets = ngram_targets.to(device)
            word_targets = word_targets.to(device)
            phrase_targets = phrase_targets.to(device)
            word_counts = word_counts.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    eeg, c_ids, c_lens, class_ids, class_lens,
                    cons_targets, ngram_targets, word_targets, phrase_targets, word_counts
                )
                loss = (
                    0.35 * out["loss_cons"] +
                    0.35 * out["loss_ngram"] +
                    0.50 * out["loss_word"] +
                    0.60 * out["loss_phrase"] +
                    0.08 * out["loss_count"] +
                    0.10 * out["loss_div"]
                ) / ACCUM_STEPS

            loss.backward()
            tot_ctc += out["loss_ctc"].item()
            tot_class += out["loss_class"].item()
            tot_cons += out["loss_cons"].item()
            tot_ngram += out["loss_ngram"].item()
            tot_word += out["loss_word"].item()
            tot_phrase += out["loss_phrase"].item()
            tot_count += out["loss_count"].item()
            tot_div += out["loss_div"].item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        n = len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | CTC:{tot_ctc/n:.3f} ACTC:{tot_class/n:.3f} "
            f"CONS:{tot_cons/n:.3f} NGRAM:{tot_ngram/n:.3f} WORD:{tot_word/n:.3f} "
            f"PHRASE:{tot_phrase/n:.3f} "
            f"COUNT:{tot_count/n:.3f} Div:{tot_div/n:.4f} | LR0:{lr:.2e}"
        )

        if (tot_ctc / n) < best_train_ctc:
            best_train_ctc = tot_ctc / n
            print(f"  ✓ New best train CTC {best_train_ctc:.3f}")

        if epoch % 3 == 0:
            best_val_cer = _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_val_cer)

    return best_train_ctc, best_val_cer


def run_phase1_curriculum(model, epochs_p1, train_loader_mix, train_loader_pure,
                          val_loader, ckpt_prefix, start_epoch=0,
                          best_train_ctc=float("inf"), best_val_cer=float("inf")):
    mix_epochs = min(PHASE1_MIX_EPOCHS, epochs_p1)
    current_epoch = start_epoch

    if current_epoch < mix_epochs and train_loader_mix is not None:
        stage_epochs = mix_epochs - current_epoch
        print(f"[Phase1] Stage A: frozen lexical support fit for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_mix,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_val_cer=best_val_cer, freeze_encoder=True,
            stage_name="frozen V14 lexical support fit")
        current_epoch = mix_epochs

    if current_epoch < epochs_p1:
        stage_epochs = epochs_p1 - current_epoch
        print(f"[Phase1] Stage B: frozen ZuCo lexical consolidation for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_pure,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_val_cer=best_val_cer, freeze_encoder=True,
            stage_name="frozen ZuCo lexical consolidation")

    return best_train_ctc, best_val_cer


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol, "/v14": v14_vol, "/v13": v13_vol, "/v20": v20_vol, "/v50": v50_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p1: int = 12):
    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_zuco = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=True, include_innerspeech=False)
    n_train  = int(0.9 * len(ds_zuco))
    indices  = list(range(len(ds_zuco)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_zuco = torch.utils.data.Subset(ds_zuco, train_idx)
    val_ds     = torch.utils.data.Subset(ds_zuco, val_idx)

    ds_inner = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=False, include_innerspeech=True)
    train_loader_mix = None
    if len(ds_inner) > 0:
        inner_cap = min(len(ds_inner), len(train_idx))
        inner_idx = np.random.permutation(len(ds_inner))[:inner_cap].tolist()
        train_inner = torch.utils.data.Subset(ds_inner, inner_idx)
        train_mix = torch.utils.data.ConcatDataset([train_zuco, train_inner])
        print(f"[Phase1] Using balanced InnerSpeech subset: {inner_cap} train words")
    else:
        train_mix = train_zuco
        print("[Phase1] InnerSpeech not found; training on ZuCo only")

    mkloader = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader_pure = mkloader(train_zuco, True)
    val_loader        = mkloader(val_ds, False)
    if len(ds_inner) > 0:
        train_loader_mix = mkloader(train_mix, True)

    global NGRAM_LIST, NGRAM_TO_ID, WORD_LIST, WORD_TO_ID, PHRASE_LIST, PHRASE_TO_ID
    train_texts_for_vocab = [ds_zuco.samples[i]["text"] for i in train_idx]
    if len(ds_inner) > 0:
        train_texts_for_vocab.extend(ds_inner.samples[i]["text"] for i in inner_idx)
    NGRAM_LIST, NGRAM_TO_ID = build_ngram_vocab(train_texts_for_vocab, max_features=256)
    WORD_LIST, WORD_TO_ID = build_word_vocab(train_texts_for_vocab, max_features=512)
    PHRASE_LIST, PHRASE_TO_ID = build_phrase_vocab(train_texts_for_vocab, max_features=384)
    print(f"[V16] ngram vocab size: {len(NGRAM_LIST)}")
    print(f"[V16] word vocab size: {len(WORD_LIST)}")
    print(f"[V16] phrase vocab size: {len(PHRASE_LIST)}")

    model = EEG_CTC_V16(
        ngram_dim=len(NGRAM_LIST),
        word_dim=len(WORD_LIST),
        phrase_dim=len(PHRASE_LIST),
    ).to(torch.bfloat16).cuda()
    model.char_lm = CharNGramLM(train_texts_for_vocab, order=4)

    # ── Resume logic ──────────────────────────────────────────────────────────
    import glob as _glob
    p1_ckpts = sorted(_glob.glob("/persist/v16_phase1_ep*.pt"))
    best_p1 = "/persist/v16_phase1_best.pt"
    final = "/persist/v16_final.pt"
    p1_start_epoch = 0
    best_val_cer = float("inf")
    best_train_ctc = float("inf")

    if Path(final).exists():
        print("✓ V16 already complete.")
        return

    if Path(best_p1).exists() and p1_ckpts:
        last_p1       = p1_ckpts[-1]
        last_p1_epoch = int(last_p1.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p1, map_location="cuda"))
        print(f"✓ Resuming Phase 1 from epoch {last_p1_epoch}")
        p1_start_epoch = last_p1_epoch
    else:
        if not load_v14_phase1_warmstart(model):
            if not load_v13_phase1_warmstart(model):
                if not load_v50_phase1_warmstart(model):
                    if not load_v20_phase1_warmstart(model):
                        load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'='*60}")
    print(" V16: frozen lexical reranking on top of V14 best")
    print(" V14 fixed decoder + word / phrase support + no encoder unfreeze")
    print(f" {'(RESUMED)' if p1_ckpts else 'FRESH START'}")
    print(f"{'='*60}\n")

    best_train_ctc, best_val_cer = run_phase1_curriculum(
        model, epochs_p1=epochs_p1,
        train_loader_mix=train_loader_mix,
        train_loader_pure=train_loader_pure,
        val_loader=val_loader,
        ckpt_prefix="/persist/v16",
        start_epoch=p1_start_epoch,
        best_train_ctc=best_train_ctc,
        best_val_cer=best_val_cer,
    )

    if Path(best_p1).exists():
        model.load_state_dict(torch.load(best_p1, map_location="cuda"))

    torch.save(model.state_dict(), "/persist/v16_final.pt")
    ckpt_vol.commit()
    print("\n✓ V16 complete. Saved /persist/v16_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 12):
    if mode == "pipeline":
        print(f"Launching V16: Phase1={epochs_p1}ep")
        run_pipeline.remote(epochs_p1=epochs_p1)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
