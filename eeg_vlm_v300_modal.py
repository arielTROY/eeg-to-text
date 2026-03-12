
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import re
import hashlib
import subprocess
from pathlib import Path

app = modal.App("eeg-vlm-v300-whisper-bridge")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v300", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

def _download_models():
    from transformers import VideoMAEModel, WhisperForConditionalGeneration, WhisperProcessor
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl", "espeak-ng", "ffmpeg"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "sentencepiece", "openneuro-py",
        "pronouncing", "torchaudio",
    ])
    .run_function(_download_models)
)

# ═══════════════════════════════════════════════════════════════════════════════
# V300: EEG-video to Whisper latent bridge.
#
# What changes:
#   [1] keep EEG-as-video, but stop decoding text directly from EEG
#   [2] synthesize teacher audio from text with espeak-ng
#   [3] distill EEG-video into frozen Whisper encoder states
#   [4] decode with the frozen Whisper decoder plus light local auxiliaries
#
# The goal is to move the target from text to speech-like acoustic/phonetic
# latents. If EEG contains phonetic structure, this is the right supervision
# level; direct text decoding has repeatedly been too brittle.
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
SP_MODEL = None
WP_VOCAB_SIZE = 0
WP_BLANK_ID = 0
_PHONE_CACHE = {}
_PRONOUNCING = None
DEC_PAD_ID = 0
DEC_BOS_ID = 0
DEC_EOS_ID = 0
DEC_VOCAB_SIZE = 0
MAX_DEC_LEN = 255

CLASS_BLANK = 0
CLASS_VOWEL = 1
CLASS_STOP = 2
CLASS_FRICATIVE = 3
CLASS_SONORANT = 4
CLASS_DIGIT = 5
CLASS_OTHER = 6
CLASS_SPACE = 7
CLASS_VOCAB_SIZE = 8

ARPABET_PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
PHONE_BOUNDARY = "|"
PHONE_FALLBACKS = [f"#{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz"]
PHONE_FEATURES = ARPABET_PHONES + [PHONE_BOUNDARY] + PHONE_FALLBACKS
PHONE_VOCAB = ["_"] + PHONE_FEATURES
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_VOCAB)}
ID_TO_PHONE = {i: p for i, p in enumerate(PHONE_VOCAB)}
PHONE_VOCAB_SIZE = len(PHONE_VOCAB)
PHONE_FEATURE_TO_ID = {p: i for i, p in enumerate(PHONE_FEATURES)}
PHONO_BASE_DIM = len(PHONE_FEATURES)
NUM_EEG_BANDS = 3
NUM_TOPO_VIEWS = 2
PHASE1_STAGE_A_EPOCHS = 1
WHISPER_PROCESSOR = None
WHISPER_PROMPT_IDS = None
WHISPER_CACHE_DIR = Path("/persist/whisper_teacher_cache")

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


def canonicalize_text(text):
    return " ".join(normalize_text(text).split())


def get_pronouncing():
    global _PRONOUNCING
    if _PRONOUNCING is None:
        try:
            import pronouncing as _p
            _PRONOUNCING = _p
        except Exception:
            _PRONOUNCING = False
    return _PRONOUNCING or None


def _normalize_phone_token(token):
    return re.sub(r"\d", "", token)


def word_to_phone_tokens(word):
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return []
    cached = _PHONE_CACHE.get(clean)
    if cached is not None:
        return list(cached)

    phones = []
    pronouncing = get_pronouncing()
    if pronouncing is not None:
        try:
            candidates = pronouncing.phones_for_word(clean)
        except Exception:
            candidates = []
        if candidates:
            phones = [
                _normalize_phone_token(tok)
                for tok in candidates[0].split()
                if _normalize_phone_token(tok) in PHONE_FEATURE_TO_ID
            ]

    if not phones:
        phones = [f"#{c.upper()}" for c in clean if f"#{c.upper()}" in PHONE_FEATURE_TO_ID]

    _PHONE_CACHE[clean] = tuple(phones)
    return phones


def text_to_phone_tokens(text):
    words = re.findall(r"[a-zA-Z']+", text)
    out = []
    for word in words:
        phones = word_to_phone_tokens(word)
        if not phones:
            continue
        if out:
            out.append(PHONE_BOUNDARY)
        out.extend(phones)
    return out


def text_to_phone_ids(text):
    return [PHONE_TO_ID[p] for p in text_to_phone_tokens(text) if p in PHONE_TO_ID]


def decode_phone_ids(ids):
    return " ".join(ID_TO_PHONE.get(i, "") for i in ids if i in ID_TO_PHONE and i != 0)


def greedy_phone_ctc_decode(logits):
    ids = logits.argmax(dim=-1)
    outputs = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out = []
        prev = -1
        for tok_id in seq:
            if tok_id != prev and tok_id != 0:
                out.append(tok_id)
            prev = tok_id
        outputs.append(decode_phone_ids(out))
    return outputs


def build_phone_bigram_vocab(texts, max_features=384):
    from collections import Counter

    counter = Counter()
    for text in texts:
        toks = text_to_phone_tokens(text)
        for i in range(len(toks) - 1):
            counter[(toks[i], toks[i + 1])] += 1
    top = [bg for bg, _ in counter.most_common(max_features)]
    return top, {bg: i for i, bg in enumerate(top)}


def phono_vector_dim():
    return PHONO_BASE_DIM + len(PHONE_BIGRAMS)


def text_to_phono_vector(text):
    vec = np.zeros(phono_vector_dim(), dtype=np.float32)
    toks = text_to_phone_tokens(text)
    seen_uni = set()
    seen_bi = set()
    for tok in toks:
        idx = PHONE_FEATURE_TO_ID.get(tok)
        if idx is not None and idx not in seen_uni:
            vec[idx] = 1.0
            seen_uni.add(idx)
    for i in range(len(toks) - 1):
        bg = (toks[i], toks[i + 1])
        idx = PHONE_BIGRAM_TO_ID.get(bg)
        if idx is not None and idx not in seen_bi:
            vec[PHONO_BASE_DIM + idx] = 1.0
            seen_bi.add(idx)
    if vec.sum() > 0:
        vec /= np.linalg.norm(vec) + 1e-8
    return vec


def build_phono_lookup(texts):
    lookup = {}
    for text in texts:
        key = canonicalize_text(text)
        if key and key not in lookup:
            lookup[key] = text_to_phono_vector(key)
    return lookup


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


def build_sentencepiece_model(texts, model_prefix="/persist/v100_wordpiece", vocab_size=192):
    import sentencepiece as spm

    text_path = f"{model_prefix}.txt"
    model_path = f"{model_prefix}.model"
    if not Path(model_path).exists():
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            for text in texts:
                norm = normalize_text(text).strip()
                if norm:
                    f.write(norm + "\n")
        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            input_sentence_size=100000,
            shuffle_input_sentence=True,
            split_by_whitespace=False,
        )
    return spm.SentencePieceProcessor(model_file=model_path)


def text_to_wp_ids(text):
    if SP_MODEL is None:
        return []
    return [tok + 1 for tok in SP_MODEL.encode(normalize_text(text), out_type=int)]


def decode_wp_ids(ids):
    if SP_MODEL is None:
        return ""
    base_ids = [int(i) - 1 for i in ids if int(i) > 0]
    if not base_ids:
        return ""
    return normalize_text(SP_MODEL.decode(base_ids)).strip()


def build_decoder_io_from_wp_ids(wp_ids):
    wp_ids = list(wp_ids)[: MAX_DEC_LEN - 1]
    dec_in = [DEC_BOS_ID] + wp_ids
    dec_out = wp_ids + [DEC_EOS_ID]
    return dec_in, dec_out


def decode_decoder_ids(ids):
    core = []
    for tok in ids:
        tok = int(tok)
        if tok in (DEC_PAD_ID, DEC_BOS_ID):
            continue
        if tok == DEC_EOS_ID:
            break
        core.append(tok)
    return decode_wp_ids(core)


def get_whisper_processor():
    global WHISPER_PROCESSOR, WHISPER_PROMPT_IDS
    if WHISPER_PROCESSOR is None:
        from transformers import WhisperProcessor

        WHISPER_PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        WHISPER_PROMPT_IDS = WHISPER_PROCESSOR.get_decoder_prompt_ids(language="english", task="transcribe")
    return WHISPER_PROCESSOR


def _text_cache_key(text):
    return hashlib.sha1(canonicalize_text(text).encode("utf-8")).hexdigest()[:16]


def _teacher_paths(text):
    key = _text_cache_key(text)
    base = WHISPER_CACHE_DIR / key
    return base.with_suffix(".wav"), base.with_suffix(".pt")


def _synthesize_tts_wav(text, wav_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    clean = canonicalize_text(text).strip()
    if not clean:
        clean = "."
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(wav_path), clean],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_whisper_teacher_cache(texts, device="cuda"):
    from scipy.io import wavfile
    from transformers import WhisperForConditionalGeneration

    processor = get_whisper_processor()
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    unique_texts = []
    seen = set()
    for text in texts:
        key = canonicalize_text(text)
        if key and key not in seen:
            seen.add(key)
            unique_texts.append(text)

    built = 0
    for idx, text in enumerate(unique_texts, 1):
        wav_path, pt_path = _teacher_paths(text)
        if pt_path.exists():
            continue
        if not wav_path.exists():
            _synthesize_tts_wav(text, wav_path)
        sr, wav = wavfile.read(str(wav_path))
        wav = np.asarray(wav)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if np.issubdtype(wav.dtype, np.integer):
            scale = max(abs(np.iinfo(wav.dtype).min), np.iinfo(wav.dtype).max)
            wav = wav.astype(np.float32) / float(scale)
        else:
            wav = wav.astype(np.float32)
        if sr != 16000:
            wav = scipy.signal.resample_poly(wav, 16000, sr).astype(np.float32)
            sr = 16000
        feats = processor(audio=wav, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            enc = model.model.encoder(input_features=feats).last_hidden_state.squeeze(0).cpu().half()
        labels = processor.tokenizer(
            canonicalize_text(text),
            add_special_tokens=True,
            return_attention_mask=False,
        ).input_ids
        torch.save({"states": enc, "labels": labels}, pt_path)
        built += 1
        if built % 100 == 0:
            print(f"[WhisperTeacher] built {built}/{len(unique_texts)}")
            ckpt_vol.commit()
    del model
    ckpt_vol.commit()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_whisper_teacher(text):
    _, pt_path = _teacher_paths(text)
    return torch.load(pt_path, map_location="cpu")


def phono_vector_for_text(text):
    key = canonicalize_text(text)
    cached = PHONO_VECTOR_LOOKUP.get(key)
    if cached is not None:
        return cached
    vec = text_to_phono_vector(key)
    PHONO_VECTOR_LOOKUP[key] = vec
    return vec


def text_to_boundary_target(text, seq_len=256):
    target = np.zeros(seq_len, dtype=np.float32)
    wp_ids = text_to_wp_ids(text)
    if len(wp_ids) <= 1:
        return target
    boundary_positions = np.linspace(0, seq_len - 1, len(wp_ids) + 1, dtype=int)[1:-1]
    for pos in boundary_positions:
        lo = max(0, pos - 1)
        hi = min(seq_len, pos + 2)
        target[lo:hi] = np.maximum(target[lo:hi], np.array([0.35, 1.0, 0.35])[:hi-lo], dtype=np.float32)
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


def beam_ctc_decode_wp_nbest(logits, beam_width=12, nbest=5, blank_penalty=0.15, diversity_weight=0.12):
    log_probs = F.log_softmax(logits, dim=-1)
    T, B, V = log_probs.shape
    results = []
    for b in range(B):
        beams = [([], -1, 0.0)]
        for t in range(T):
            new_beams = {}
            frame_lp = log_probs[t, b].clone()
            frame_lp[WP_BLANK_ID] -= blank_penalty
            topk_vals, topk_ids = frame_lp.topk(min(beam_width * 3, V))
            for seq, last_tok, ctc_score in beams:
                for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_ctc_score = ctc_score + val
                    if tok_id == WP_BLANK_ID or tok_id == last_tok:
                        key = tuple(seq)
                        existing = new_beams.get(key)
                        cand = (seq, tok_id, new_ctc_score)
                        if existing is None or existing[2] < new_ctc_score:
                            new_beams[key] = cand
                    else:
                        new_seq = seq + [tok_id]
                        key = tuple(new_seq)
                        existing = new_beams.get(key)
                        cand = (new_seq, tok_id, new_ctc_score)
                        if existing is None or existing[2] < new_ctc_score:
                            new_beams[key] = cand
            candidates = sorted(new_beams.values(), key=lambda x: -x[2])[:max(beam_width * 2, nbest)]
            selected = []
            remaining = list(candidates)
            while remaining and len(selected) < max(beam_width, nbest):
                best_idx = 0
                best_score = None
                for idx, cand in enumerate(remaining):
                    seq, _, score = cand
                    text = decode_wp_ids(seq)
                    if selected:
                        overlap = max(_common_prefix_ratio(text, decode_wp_ids(s[0])) for s in selected)
                    else:
                        overlap = 0.0
                    div_score = score - diversity_weight * overlap
                    if best_score is None or div_score > best_score:
                        best_score = div_score
                        best_idx = idx
                selected.append(remaining.pop(best_idx))
            beams = selected
        if not beams:
            results.append([("", 1.0, 0)])
            continue
        dedup = {}
        for seq, _, score in beams:
            text = decode_wp_ids(seq)
            if not text:
                continue
            if text not in dedup or dedup[text][1] < score:
                dedup[text] = (len(seq), score)
        ranked = sorted(dedup.items(), key=lambda x: -x[1][1])[:nbest]
        if not ranked:
            results.append([("", 1.0, 0)])
            continue
        scores = torch.tensor([score for _, (_, score) in ranked], dtype=torch.float32)
        norm_scores = torch.softmax(scores - scores.max(), dim=0).tolist()
        results.append([
            (text, float(prob), tok_len)
            for (text, (tok_len, _)), prob in zip(ranked, norm_scores)
        ])
    return results


def _vowel_skeleton(text):
    return "".join(ch for ch in normalize_text(text) if ch in VOWELS or ch == " ")


def _ngram_support(text, probs):
    score = 0.0
    seen = set()
    for word in normalize_text(text).split():
        if len(word) < 2:
            continue
        for n in (2, 3):
            if len(word) < n:
                continue
            for i in range(len(word) - n + 1):
                ng = word[i:i + n]
                idx = NGRAM_TO_ID.get(ng)
                if idx is not None and idx not in seen:
                    score += float(probs[idx])
                    seen.add(idx)
    return score


def _consonant_support(text, probs):
    score = 0.0
    seen = set()
    for ch in normalize_text(text):
        idx = CONS_TO_ID.get(ch)
        if idx is not None and idx not in seen:
            score += float(probs[idx])
            seen.add(idx)
    return score


def _estimate_boundary_count(boundary_probs):
    probs = np.asarray(boundary_probs, dtype=np.float32)
    if probs.size < 3:
        return 1
    peaks = 0
    for i in range(1, len(probs) - 1):
        if probs[i] > 0.45 and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
            peaks += 1
    return max(1, peaks + 1)


def _phone_support(text, phone_probs):
    toks = text_to_phone_ids(text)
    if not toks:
        return 0.0
    seen = set()
    score = 0.0
    for tok in toks:
        if tok == 0 or tok in seen or tok >= len(phone_probs):
            continue
        score += float(phone_probs[tok])
        seen.add(tok)
    return score


def merge_char_wp_candidates(char_beams, wp_beams, ngram_probs, cons_probs, boundary_probs, phone_probs, nbest=5):
    char_hint = char_beams[0][0] if char_beams else ""
    char_vowels = _vowel_skeleton(char_hint)
    expected_count = _estimate_boundary_count(boundary_probs)
    merged = {}

    for text, score in char_beams:
        norm = normalize_text(text).strip()
        if not norm:
            continue
        vowel_match = max(0.0, 1.0 - min(compute_cer(_vowel_skeleton(norm), char_vowels), 1.5))
        merged[norm] = max(merged.get(norm, -1e9), 0.75 * float(score) + 0.30 * vowel_match)

    for text, score, tok_len in wp_beams:
        norm = normalize_text(text).strip()
        if not norm:
            continue
        vowel_match = max(0.0, 1.0 - min(compute_cer(_vowel_skeleton(norm), char_vowels), 1.5))
        ngram_support = _ngram_support(norm, ngram_probs)
        cons_support = _consonant_support(norm, cons_probs)
        phone_support = _phone_support(norm, phone_probs)
        count_penalty = abs(tok_len - expected_count)
        merged_score = (
            1.10 * float(score)
            + 0.40 * vowel_match
            + 0.18 * ngram_support
            + 0.12 * cons_support
            + 0.18 * phone_support
            - 0.08 * count_penalty
        )
        merged[norm] = max(merged.get(norm, -1e9), merged_score)

    ranked = sorted(merged.items(), key=lambda x: -x[1])[:nbest]
    if not ranked:
        return [("", 1.0)]
    score_tensor = torch.tensor([score for _, score in ranked], dtype=torch.float32)
    probs = torch.softmax(score_tensor - score_tensor.max(), dim=0).tolist()
    return [(text, float(prob)) for (text, _), prob in zip(ranked, probs)]


def rerank_with_phonology(candidates, eeg_phono_vec, alpha=0.45):
    rescored = []
    for text, base_prob in candidates:
        cand_vec = phono_vector_for_text(text)
        if cand_vec.sum() > 0:
            phono_sim = float(np.dot(cand_vec, eeg_phono_vec))
        else:
            phono_sim = 0.0
        score = np.log(max(base_prob, 1e-8)) + alpha * phono_sim
        rescored.append((text, score))
    if not rescored:
        return [("", 1.0)]
    rescored = sorted(rescored, key=lambda x: -x[1])[: len(candidates)]
    score_tensor = torch.tensor([score for _, score in rescored], dtype=torch.float32)
    probs = torch.softmax(score_tensor - score_tensor.max(), dim=0).tolist()
    return [(text, float(prob)) for (text, _), prob in zip(rescored, probs)]


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


def compute_per(pred, ref):
    import editdistance
    pred_p = pred.strip().split()
    ref_p = ref.strip().split()
    if not ref_p:
        return 0.0 if not pred_p else 1.0
    return min(editdistance.eval(pred_p, ref_p) / len(ref_p), 2.0)


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

class MultiViewTopographyRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:n_electrodes]
        xyz = np.stack([montage.get_positions()["ch_pos"][name] for name in ch_names], axis=0)
        pos_2d = xyz[:, :2].astype(np.float32)
        pos_2d -= pos_2d.mean(axis=0, keepdims=True)
        pos_2d /= np.abs(pos_2d).max() + 1e-6
        pos = torch.from_numpy(pos_2d)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(1, -1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        px = grid_x.flatten().unsqueeze(1)
        py = grid_y.flatten().unsqueeze(1)
        ex = pos[:, 0].unsqueeze(0)
        ey = pos[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        base_w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px ** 2 + py ** 2)
        base_w[(r > 1.1).squeeze(1), :] = 0.0

        left_mask = (pos[:, 0] <= 0.10).float().unsqueeze(0)
        w_full = base_w / base_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_left = base_w * left_mask
        w_left = w_left / w_left.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W_full", w_full)
        self.register_buffer("W_left", w_left)

    def forward(self, x):
        B, T, Ch = x.shape
        x_split = x.reshape(B * T, self.n_bands, self.n_electrodes)
        full = x_split @ self.W_full.T.to(device=x.device, dtype=x.dtype)
        left = x_split @ self.W_left.T.to(device=x.device, dtype=x.dtype)
        full = full.reshape(B, T, self.n_bands, 64, 64)
        left = left.reshape(B, T, self.n_bands, 64, 64)
        return torch.cat([full, left], dim=2)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS * NUM_TOPO_VIEWS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        out = self.conv(imgs.view(B * T, C, H, W))
        return out.view(B, T, 3, H, W)


class EEGWhisperBridge(nn.Module):
    def __init__(self, ngram_dim):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, WhisperForConditionalGeneration

        self.clip_frames = 128
        self.n_clips = 8
        self.temporal_tokens_per_clip = 32

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=self.clip_frames, tubelet_size=4, hidden_size=768
        )
        self.rasterizer = MultiViewTopographyRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt = ChannelAdapter(NUM_EEG_BANDS * NUM_TOPO_VIEWS)
        self.video_enc = VideoMAEModel(v_cfg)
        self.clip_embed = nn.Embedding(self.n_clips, 768)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.bridge = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 384),
        )
        self.length_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1))
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper.eval()
        self.ctc_head = nn.Linear(768, VOCAB_SIZE)
        self.phone_ctc_head = nn.Linear(768, PHONE_VOCAB_SIZE)
        self.cons_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, len(CONS_LIST)))
        self.ngram_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, ngram_dim))
        self.boundary_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1))
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.phone_ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.cons_loss_fn = nn.BCEWithLogitsLoss()
        self.ngram_loss_fn = nn.BCEWithLogitsLoss()
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()

    def encode(self, traj):
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        B, T, C, H, W = imgs.shape
        clips = imgs.reshape(B, self.n_clips, self.clip_frames, C, H, W)
        clips = clips.reshape(B * self.n_clips, self.clip_frames, C, H, W)
        v_out = self.video_enc(pixel_values=clips).last_hidden_state
        v_out = v_out.reshape(B * self.n_clips, self.temporal_tokens_per_clip, 16, 768).mean(dim=2)
        v_out = v_out.reshape(B, self.n_clips, self.temporal_tokens_per_clip, 768)
        clip_ids = torch.arange(self.n_clips, device=traj.device).view(1, self.n_clips, 1)
        v_out = v_out + self.clip_embed(clip_ids).to(dtype=v_out.dtype)
        memory = v_out.reshape(B, self.n_clips * self.temporal_tokens_per_clip, 768)
        return self.temporal_encoder(memory)

    def _predict_teacher_lens(self, memory):
        raw = F.softplus(self.length_head(memory.mean(dim=1)).squeeze(-1))
        return raw + 32.0

    def _resample_bridge(self, bridge_states, target_lens):
        B, _, D = bridge_states.shape
        max_len = int(target_lens.max().item())
        out = bridge_states.new_zeros(B, max_len, D)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=bridge_states.device)
        for i in range(B):
            tgt_len = int(target_lens[i].item())
            src = bridge_states[i:i+1].transpose(1, 2)
            resized = F.interpolate(src, size=tgt_len, mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
            out[i, :tgt_len] = resized
            mask[i, :tgt_len] = 1
        return out, mask

    def forward(self, traj, char_ids, char_lens, phone_ids, phone_lens, cons_targets, ngram_targets, boundary_targets,
                teacher_states, teacher_lens, whisper_labels):
        from transformers.modeling_outputs import BaseModelOutput

        B = traj.shape[0]
        device = traj.device
        memory = self.encode(traj)
        bridge_states = self.bridge(memory)
        pred_teacher_lens = self._predict_teacher_lens(memory)
        resampled_bridge, teacher_mask = self._resample_bridge(bridge_states, teacher_lens)

        teacher_states = teacher_states.to(dtype=resampled_bridge.dtype)
        mask_f = teacher_mask.unsqueeze(-1).to(dtype=resampled_bridge.dtype)
        align_loss = F.smooth_l1_loss(resampled_bridge * mask_f, teacher_states * mask_f, reduction="sum") / mask_f.sum().clamp(min=1.0)
        len_loss = F.l1_loss(torch.log1p(pred_teacher_lens), torch.log1p(teacher_lens.float()))

        enc_outputs = BaseModelOutput(last_hidden_state=resampled_bridge)
        whisper_out = self.whisper(
            encoder_outputs=enc_outputs,
            attention_mask=teacher_mask,
            labels=whisper_labels,
        )
        loss_whisper = whisper_out.loss

        ctc_logits = self.ctc_head(memory).transpose(0, 1)
        phone_logits = self.phone_ctc_head(memory).transpose(0, 1)
        cons_logits = self.cons_head(memory.mean(dim=1))
        ngram_logits = self.ngram_head(memory.mean(dim=1))
        boundary_logits = self.boundary_head(memory).squeeze(-1)
        input_lens = torch.full((B,), memory.shape[1], dtype=torch.long, device=device)
        loss_ctc = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens
        )
        loss_phone_ctc = self.phone_ctc_loss_fn(
            F.log_softmax(phone_logits, dim=-1), phone_ids, input_lens, phone_lens
        )
        loss_cons = self.cons_loss_fn(cons_logits, cons_targets)
        loss_ngram = self.ngram_loss_fn(ngram_logits, ngram_targets)
        loss_boundary = self.boundary_loss_fn(boundary_logits, boundary_targets)
        probs = F.softmax(ctc_logits.transpose(0, 1), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        loss_div = -ent.mean()
        return {
            "loss_whisper": loss_whisper,
            "loss_align": align_loss,
            "loss_len": len_loss,
            "loss_ctc": loss_ctc,
            "loss_phone_ctc": loss_phone_ctc,
            "loss_cons": loss_cons,
            "loss_ngram": loss_ngram,
            "loss_boundary": loss_boundary,
            "loss_div": loss_div,
        }

    @torch.no_grad()
    def generate(self, traj, max_new_tokens=48):
        from transformers.modeling_outputs import BaseModelOutput

        processor = get_whisper_processor()
        prompt_ids = WHISPER_PROMPT_IDS
        traj = traj.to(next(self.parameters()).dtype)
        memory = self.encode(traj)
        bridge_states = self.bridge(memory)
        pred_lens = self._predict_teacher_lens(memory).round().clamp(min=48, max=300).long()
        resampled_bridge, teacher_mask = self._resample_bridge(bridge_states, pred_lens)
        enc_outputs = BaseModelOutput(last_hidden_state=resampled_bridge)
        gen_ids = self.whisper.generate(
            encoder_outputs=enc_outputs,
            attention_mask=teacher_mask,
            forced_decoder_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        return [canonicalize_text(t) for t in processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]


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


def load_v17_phase1_warmstart(model, ckpt_path="/v17/v17_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V17 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V17 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = (
        "rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "wp_ctc_head.",
        "class_ctc_head.", "cons_head.", "ngram_head.", "boundary_head."
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
    print(f"  ✓ Loaded {n_copied} V17 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_v14_phase1_warmstart(model, ckpt_path="/v14/v14_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V14 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V14 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "class_ctc_head.", "cons_head.", "ngram_head.")
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
    for d in [zuco_v1, zuco_v2]:
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

    theta = band_env(4, 8, eeg)
    alpha = band_env(8, 13, eeg)
    gamma = band_env(30, 80, eeg)

    combined = np.concatenate([theta.T, alpha.T, gamma.T], axis=1)  # (1024, 192)
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
        phone_ids = text_to_phone_ids(s['text'])
        cons_target = text_to_cons_target(s['text'])
        ngram_target = text_to_ngram_target(s['text'])
        boundary_target = text_to_boundary_target(s['text'])
        teacher = load_whisper_teacher(s['text'])
        return eeg, s['text'], char_ids, phone_ids, cons_target, ngram_target, boundary_target, teacher


def collate_fn(batch):
    eegs, texts, char_ids_list, phone_ids_list, cons_targets, ngram_targets, boundary_targets, teacher_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(c) for c in char_ids_list], dtype=torch.long)
    max_chars = max(char_lens) if max(char_lens) > 0 else 1
    char_pad  = torch.zeros(len(char_ids_list), max_chars, dtype=torch.long)
    for i, c in enumerate(char_ids_list):
        if len(c) > 0:
            char_pad[i, :len(c)] = torch.tensor(c)
    phone_lens = torch.tensor([len(p) for p in phone_ids_list], dtype=torch.long)
    max_phone = max(phone_lens) if max(phone_lens) > 0 else 1
    phone_pad = torch.zeros(len(phone_ids_list), max_phone, dtype=torch.long)
    for i, p in enumerate(phone_ids_list):
        if len(p) > 0:
            phone_pad[i, :len(p)] = torch.tensor(p)
    cons_tensor = torch.from_numpy(np.stack(cons_targets))
    ngram_tensor = torch.from_numpy(np.stack(ngram_targets))
    boundary_tensor = torch.from_numpy(np.stack(boundary_targets))
    teacher_lens = torch.tensor([t["states"].shape[0] for t in teacher_list], dtype=torch.long)
    max_teacher = max(teacher_lens).item() if len(teacher_lens) > 0 else 1
    teacher_dim = teacher_list[0]["states"].shape[1]
    teacher_pad = torch.zeros(len(teacher_list), max_teacher, teacher_dim, dtype=torch.float32)
    for i, t in enumerate(teacher_list):
        states = t["states"].float()
        teacher_pad[i, : states.shape[0]] = states
    label_lens = [len(t["labels"]) for t in teacher_list]
    max_label = max(label_lens) if label_lens else 1
    whisper_labels = torch.full((len(teacher_list), max_label), -100, dtype=torch.long)
    for i, t in enumerate(teacher_list):
        labels = torch.tensor(t["labels"], dtype=torch.long)
        whisper_labels[i, : labels.shape[0]] = labels
    return (
        eegs, list(texts), char_pad, char_lens, phone_pad, phone_lens,
        cons_tensor, ngram_tensor, boundary_tensor,
        teacher_pad, teacher_lens, whisper_labels,
    )


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


def set_videomae_last_layers_trainable(video_enc, n_last=4):
    layers = getattr(getattr(video_enc, "encoder", None), "layer", None)
    if layers is None:
        return
    for layer in list(layers)[-n_last:]:
        for p in layer.parameters():
            p.requires_grad = True
    if getattr(video_enc, "layernorm", None) is not None:
        for p in video_enc.layernorm.parameters():
            p.requires_grad = True


def set_phase1_trainable(model, freeze_encoder):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.ch_adapt.parameters():
        p.requires_grad = True
    for p in model.clip_embed.parameters():
        p.requires_grad = True
    for p in model.temporal_encoder.parameters():
        p.requires_grad = True
    for p in model.bridge.parameters():
        p.requires_grad = True
    for p in model.length_head.parameters():
        p.requires_grad = True
    for p in model.ctc_head.parameters():
        p.requires_grad = True
    for p in model.phone_ctc_head.parameters():
        p.requires_grad = True
    for p in model.cons_head.parameters():
        p.requires_grad = True
    for p in model.ngram_head.parameters():
        p.requires_grad = True
    for p in model.boundary_head.parameters():
        p.requires_grad = True
    if not freeze_encoder:
        set_videomae_last_layers_trainable(model.video_enc, n_last=4)


def make_phase1_optimizer(model, steps_per_epoch, epochs, freeze_encoder):
    set_phase1_trainable(model, freeze_encoder)
    groups = []

    def add_group(params, lr, name):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.01})

    add_group(model.ch_adapt.parameters(), 2e-4 if freeze_encoder else 3e-4, "adapter")
    add_group(model.clip_embed.parameters(), 1.2e-4, "clip_emb")
    add_group(model.temporal_encoder.parameters(), 1.5e-4 if freeze_encoder else 1.2e-4, "temporal_enc")
    add_group(model.bridge.parameters(), 1.2e-4 if freeze_encoder else 1.0e-4, "bridge")
    add_group(model.length_head.parameters(), 1.0e-4 if freeze_encoder else 8e-5, "len")
    add_group(model.ctc_head.parameters(), 1.1e-4 if freeze_encoder else 1.0e-4, "ctc")
    add_group(model.phone_ctc_head.parameters(), 9e-5 if freeze_encoder else 1.1e-4, "phone")
    add_group(model.cons_head.parameters(), 8e-5 if freeze_encoder else 1e-4, "cons")
    add_group(model.ngram_head.parameters(), 8e-5 if freeze_encoder else 1e-4, "ngram")
    add_group(model.boundary_head.parameters(), 7e-5 if freeze_encoder else 9e-5, "bnd")
    if not freeze_encoder:
        add_group(
            [p for p in model.video_enc.parameters() if p.requires_grad],
            1.2e-5,
            "enc_last4",
        )

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


def _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_phono_top1, best_oracle_wer):
    device = next(model.parameters()).device
    model.eval()
    cer_scores, wer_scores, per_scores = [], [], []
    vowel_correct = vowel_total = cons_correct = cons_total = 0
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for batch in val_loader:
            eeg_v = batch[0]
            texts_v = batch[1]
            char_ids_v = batch[2]
            char_lens_v = batch[3]
            phone_ids_v = batch[4]
            phone_lens_v = batch[5]
            preds = model.generate(eeg_v.to(device).to(torch.bfloat16), max_new_tokens=48)
            memory = model.encode(eeg_v.to(device).to(torch.bfloat16))
            ctc_logits = model.ctc_head(memory).transpose(0, 1)
            phone_logits = model.phone_ctc_head(memory).transpose(0, 1)
            ctc_preds = greedy_ctc_decode(ctc_logits)
            phone_preds = greedy_phone_ctc_decode(phone_logits)
            for b_idx, (top, ref) in enumerate(zip(preds, texts_v)):
                ref_lower = ref.lower()
                _ = "".join(ID_TO_CHAR.get(i, "") for i in char_ids_v[b_idx][: char_lens_v[b_idx]].tolist())
                phone_ref = decode_phone_ids(phone_ids_v[b_idx][: phone_lens_v[b_idx]].tolist())
                per_scores.append(compute_per(phone_preds[b_idx], phone_ref))
                cer_scores.append(compute_cer(top, ref_lower))
                wer_scores.append(compute_wer(top, ref_lower))
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
                    print(f"  CTC: '{ctc_preds[b_idx][:100]}'")
                    print(f"  PHC: '{phone_preds[b_idx][:100]}'  PER={per_scores[-1]:.2f}")
                    shown += 1
                if shown >= 4:
                    break
            if shown >= 4:
                break

    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    mean_wer = float(np.mean(wer_scores)) if wer_scores else float("inf")
    if cer_scores:
        v_acc = vowel_correct / max(vowel_total, 1)
        c_acc = cons_correct / max(cons_total, 1)
        print(f"  Mean CER: {mean_cer:.3f}  Mean WER: {mean_wer:.3f}")
        if per_scores:
            print(f"  Mean PER: {np.mean(per_scores):.3f}")
        print(f"  Vowel recall: {v_acc:.2f} ({vowel_correct}/{vowel_total})  "
              f"Consonant recall: {c_acc:.2f} ({cons_correct}/{cons_total})")

    torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_ep{epoch}.pt")
    ckpt_vol.commit()
    if mean_wer < best_oracle_wer:
        best_oracle_wer = mean_wer
        torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
        print(f"  ✓ New best seq2seq checkpoint: WER={best_oracle_wer:.3f}")
    return best_phono_top1, best_oracle_wer


def _train_phase1_stage(model, epochs, train_loader, val_loader, ckpt_prefix,
                        epoch_offset=0, best_train_ctc=float("inf"),
                        best_phono_top1=-1.0, best_oracle_wer=float("inf"), freeze_encoder=False,
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
        model.whisper.eval()
        tot_wh = tot_align = tot_len = tot_ctc = tot_phone = tot_cons = tot_ngram = tot_bnd = tot_div = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eeg, _, char_ids, char_lens, phone_ids, phone_lens, cons_targets, ngram_targets, boundary_targets, teacher_states, teacher_lens, whisper_labels = batch
            eeg = eeg.to(device).to(torch.bfloat16)
            char_ids = char_ids.to(device)
            char_lens = char_lens.to(device)
            phone_ids = phone_ids.to(device)
            phone_lens = phone_lens.to(device)
            cons_targets = cons_targets.to(device)
            ngram_targets = ngram_targets.to(device)
            boundary_targets = boundary_targets.to(device)
            teacher_states = teacher_states.to(device).to(torch.bfloat16)
            teacher_lens = teacher_lens.to(device)
            whisper_labels = whisper_labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    eeg, char_ids, char_lens, phone_ids, phone_lens, cons_targets, ngram_targets, boundary_targets,
                    teacher_states, teacher_lens, whisper_labels
                )
                loss = (
                    3.0 * out["loss_whisper"] +
                    2.0 * out["loss_align"] +
                    0.50 * out["loss_len"] +
                    1.25 * out["loss_ctc"] +
                    1.25 * out["loss_phone_ctc"] +
                    0.25 * out["loss_cons"] +
                    0.20 * out["loss_ngram"] +
                    0.50 * out["loss_boundary"] +
                    0.05 * out["loss_div"]
                ) / ACCUM_STEPS

            loss.backward()
            tot_wh += out["loss_whisper"].item()
            tot_align += out["loss_align"].item()
            tot_len += out["loss_len"].item()
            tot_ctc += out["loss_ctc"].item()
            tot_phone += out["loss_phone_ctc"].item()
            tot_cons += out["loss_cons"].item()
            tot_ngram += out["loss_ngram"].item()
            tot_bnd += out["loss_boundary"].item()
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
            f"Epoch {epoch:3d} | WH:{tot_wh/n:.3f} ALN:{tot_align/n:.3f} LEN:{tot_len/n:.3f} CTC:{tot_ctc/n:.3f} "
            f"PCTC:{tot_phone/n:.3f} CONS:{tot_cons/n:.3f} NGRAM:{tot_ngram/n:.3f} "
            f"BND:{tot_bnd/n:.3f} Div:{tot_div/n:.4f} | LR0:{lr:.2e}"
        )

        if (tot_wh / n) < best_train_ctc:
            best_train_ctc = tot_wh / n
            print(f"  ✓ New best train WH {best_train_ctc:.3f}")

        best_phono_top1, best_oracle_wer = _validate_phase1(
            model, val_loader, epoch, ckpt_prefix, best_phono_top1, best_oracle_wer
        )

    return best_train_ctc, best_phono_top1, best_oracle_wer


def run_phase1_curriculum(model, epochs_p1, train_loader_pure,
                          val_loader, ckpt_prefix, start_epoch=0,
                          best_train_ctc=float("inf"), best_phono_top1=-1.0,
                          best_oracle_wer=float("inf")):
    current_epoch = start_epoch
    frozen_epochs = min(PHASE1_STAGE_A_EPOCHS, epochs_p1)

    if current_epoch < frozen_epochs:
        stage_epochs = frozen_epochs - current_epoch
        print(f"[Phase1] Stage A: frozen local-phonology alignment for {stage_epochs} epochs")
        best_train_ctc, best_phono_top1, best_oracle_wer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_pure,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_phono_top1=best_phono_top1, best_oracle_wer=best_oracle_wer,
            freeze_encoder=True,
            stage_name="frozen warm-start + local phone / chunk supervision")
        current_epoch = frozen_epochs

    if current_epoch < epochs_p1:
        stage_epochs = epochs_p1 - current_epoch
        print(f"[Phase1] Stage B: low-LR encoder adaptation for {stage_epochs} epochs")
        best_train_ctc, best_phono_top1, best_oracle_wer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_pure,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_phono_top1=best_phono_top1, best_oracle_wer=best_oracle_wer,
            freeze_encoder=False,
            stage_name="last-4-layer low-LR topography adaptation")

    return best_train_ctc, best_phono_top1, best_oracle_wer


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p1: int = 2, use_innerspeech: bool = False):
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

    print("[Phase1] Using ZuCo-only startup path")

    mkloader = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader_pure = mkloader(train_zuco, True)
    val_loader        = mkloader(val_ds, False)

    global NGRAM_LIST, NGRAM_TO_ID, SP_MODEL, WP_VOCAB_SIZE
    global DEC_PAD_ID, DEC_BOS_ID, DEC_EOS_ID, DEC_VOCAB_SIZE
    train_texts_zuco = [ds_zuco.samples[i]["text"] for i in train_idx]
    train_texts_for_vocab = list(train_texts_zuco)
    NGRAM_LIST, NGRAM_TO_ID = build_ngram_vocab(train_texts_for_vocab, max_features=256)
    SP_MODEL = build_sentencepiece_model(train_texts_for_vocab, model_prefix="/persist/v300_wordpiece", vocab_size=192)
    WP_VOCAB_SIZE = SP_MODEL.get_piece_size() + 1
    DEC_PAD_ID = 0
    DEC_BOS_ID = WP_VOCAB_SIZE
    DEC_EOS_ID = WP_VOCAB_SIZE + 1
    DEC_VOCAB_SIZE = WP_VOCAB_SIZE + 2
    print(f"[V300] ngram vocab size: {len(NGRAM_LIST)}")
    print(f"[V300] wordpiece vocab size: {WP_VOCAB_SIZE - 1}")

    all_texts = [s["text"] for s in ds_zuco.samples]
    print(f"[WhisperTeacher] ensuring cache for {len(all_texts)} sentences")
    build_whisper_teacher_cache(all_texts, device="cuda")

    model = EEGWhisperBridge(
        ngram_dim=len(NGRAM_LIST),
    ).to(torch.bfloat16).cuda()

    best_phono_top1 = -1.0
    best_oracle_wer = float("inf")
    best_train_ctc = float("inf")
    load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'='*60}")
    print(" V300: EEG-video to Whisper latent bridge")
    print(" Exact topography + hierarchical VideoMAE + frozen Whisper decode")
    print(" FRESH START")
    print(f"{'='*60}\n")

    best_train_ctc, best_phono_top1, best_oracle_wer = run_phase1_curriculum(
        model, epochs_p1=epochs_p1,
        train_loader_pure=train_loader_pure,
        val_loader=val_loader,
        ckpt_prefix="/persist/v300",
        start_epoch=0,
        best_train_ctc=best_train_ctc,
        best_phono_top1=best_phono_top1,
        best_oracle_wer=best_oracle_wer,
    )

    best_p1 = "/persist/v300_phase1_best.pt"
    if Path(best_p1).exists():
        model.load_state_dict(torch.load(best_p1, map_location="cuda"))

    torch.save(model.state_dict(), "/persist/v300_final.pt")
    ckpt_vol.commit()
    print("\n✓ V300 complete. Saved /persist/v300_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 2, use_innerspeech: bool = False):
    if mode == "pipeline":
        print(f"Launching V300: Phase1={epochs_p1}ep inner={use_innerspeech}")
        run_pipeline.remote(epochs_p1=epochs_p1, use_innerspeech=use_innerspeech)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
