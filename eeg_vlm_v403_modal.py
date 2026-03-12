import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path


app = modal.App("eeg-vlm-v403-audio-slot")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v403", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "numpy",
        "scipy",
        "jiwer",
        "h5py",
        "mne",
        "pandas",
        "accelerate",
        "osfclient",
        "editdistance",
        "openneuro-py",
        "sentencepiece",
    ])
    .run_function(_download_models)
)

# V403: grounded audio-family branch with slot decoding.
# EEG timeline -> conv frontend -> temporal transformer -> pseudo-mel
# -> Whisper encoder -> fixed wordpiece slot decoder + wp CTC auxiliary.

WHISPER_PROCESSOR = None
CHAR_LM = None
SP_MODEL = None
CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)
WP_BLANK_ID = 0
WP_VOCAB_SIZE = 0
WP_SLOT_EOS_ID = 0
WP_SLOT_PAD_ID = 0
TIME_STEPS = 1024
EEG_DIM = 192
LATENT_DIM = 256
ACCUM_STEPS = 4
MAX_SLOTS = 40


def get_whisper_processor():
    global WHISPER_PROCESSOR
    if WHISPER_PROCESSOR is None:
        from transformers import WhisperProcessor

        WHISPER_PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    return WHISPER_PROCESSOR


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            return b"MATLAB" in f.read(128)
    except Exception:
        return False


def compute_cer(pred, ref):
    import editdistance

    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / max(len(ref), 1), 2.0)


def compute_wer(pred, ref):
    import editdistance

    pred_w = pred.strip().split()
    ref_w = ref.strip().split()
    if not ref_w:
        return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


def build_sentencepiece_model(texts, model_prefix="/persist/v403_wordpiece", vocab_size=192):
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


def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in normalize_text(text) if c in CHAR_TO_ID]


def text_to_wp_ids(text):
    if SP_MODEL is None:
        return []
    return [tok + 1 for tok in SP_MODEL.encode(normalize_text(text), out_type=int)]


def decode_wp_ids(ids):
    if SP_MODEL is None:
        return ""
    core = [int(i) - 1 for i in ids if int(i) > 0]
    if not core:
        return ""
    return normalize_text(SP_MODEL.decode(core)).strip()


def build_slot_targets(wp_ids):
    seq = list(wp_ids)[: MAX_SLOTS - 1]
    seq.append(WP_SLOT_EOS_ID)
    if len(seq) < MAX_SLOTS:
        seq.extend([WP_SLOT_PAD_ID] * (MAX_SLOTS - len(seq)))
    return seq


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


def beam_ctc_decode_nbest(logits, beam_width=12, nbest=5, blank_penalty=0.2, char_lm=None, lm_weight=0.35, diversity_weight=0.16):
    log_probs = F.log_softmax(logits, dim=-1)
    t, bsz, vocab = log_probs.shape
    results = []
    for b in range(bsz):
        beams = [([], "", -1, 0.0, 0.0)]
        for step in range(t):
            frame = log_probs[step, b].clone()
            frame[0] -= blank_penalty
            vals, ids = frame.topk(min(beam_width * 3, vocab))
            new_beams = {}
            for seq, text, last_tok, ctc_score, lm_score in beams:
                for val, tok_id in zip(vals.tolist(), ids.tolist()):
                    new_ctc = ctc_score + val
                    if tok_id == 0:
                        key = tuple(seq)
                        cand = (seq, text, 0, new_ctc, lm_score)
                    elif tok_id == last_tok:
                        key = tuple(seq)
                        cand = (seq, text, tok_id, new_ctc, lm_score)
                    else:
                        new_seq = seq + [tok_id]
                        ch = ID_TO_CHAR.get(tok_id, "")
                        new_text = text + ch
                        new_lm = lm_score + (char_lm.log_prob(text, ch) if char_lm is not None else 0.0)
                        key = tuple(new_seq)
                        cand = (new_seq, new_text, tok_id, new_ctc, new_lm)
                    if key not in new_beams or (new_beams[key][3] + lm_weight * new_beams[key][4]) < (cand[3] + lm_weight * cand[4]):
                        new_beams[key] = cand
            candidates = sorted(new_beams.values(), key=lambda x: -(x[3] + lm_weight * x[4]))[: max(beam_width * 2, nbest)]
            selected = []
            remaining = list(candidates)
            while remaining and len(selected) < max(beam_width, nbest):
                best_idx = 0
                best_score = None
                for idx, cand in enumerate(remaining):
                    overlap = max((_common_prefix_ratio(cand[1], s[1]) for s in selected), default=0.0)
                    score = cand[3] + lm_weight * cand[4] - diversity_weight * overlap
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = idx
                selected.append(remaining.pop(best_idx))
            beams = selected
        scores = torch.tensor([x[3] + lm_weight * x[4] for x in beams], dtype=torch.float32)
        probs = torch.softmax(scores - scores.max(), dim=0).tolist()
        results.append([(beam[1], float(prob)) for beam, prob in zip(beams[:nbest], probs[:nbest])])
    return results


def beam_ctc_decode_wp_nbest(logits, beam_width=12, nbest=5, blank_penalty=0.15, diversity_weight=0.12):
    log_probs = F.log_softmax(logits, dim=-1)
    t, bsz, vocab = log_probs.shape
    results = []
    for b in range(bsz):
        beams = [([], -1, 0.0)]
        for step in range(t):
            frame = log_probs[step, b].clone()
            frame[WP_BLANK_ID] -= blank_penalty
            vals, ids = frame.topk(min(beam_width * 3, vocab))
            new_beams = {}
            for seq, last_tok, ctc_score in beams:
                for val, tok_id in zip(vals.tolist(), ids.tolist()):
                    new_ctc = ctc_score + val
                    if tok_id == WP_BLANK_ID or tok_id == last_tok:
                        key = tuple(seq)
                        cand = (seq, tok_id, new_ctc)
                    else:
                        new_seq = seq + [tok_id]
                        key = tuple(new_seq)
                        cand = (new_seq, tok_id, new_ctc)
                    if key not in new_beams or new_beams[key][2] < cand[2]:
                        new_beams[key] = cand
            candidates = sorted(new_beams.values(), key=lambda x: -x[2])[: max(beam_width * 2, nbest)]
            selected = []
            remaining = list(candidates)
            while remaining and len(selected) < max(beam_width, nbest):
                best_idx = 0
                best_score = None
                for idx, cand in enumerate(remaining):
                    text = decode_wp_ids(cand[0])
                    overlap = max((_common_prefix_ratio(text, decode_wp_ids(s[0])) for s in selected), default=0.0)
                    score = cand[2] - diversity_weight * overlap
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = idx
                selected.append(remaining.pop(best_idx))
            beams = selected
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
        probs = torch.softmax(scores - scores.max(), dim=0).tolist()
        results.append([(text, float(prob), tok_len) for (text, (tok_len, _)), prob in zip(ranked, probs)])
    return results


def merge_char_wp_candidates(char_beams, wp_beams, nbest=5):
    merged = {}
    for text, prob in char_beams:
        norm = normalize_text(text).strip()
        if norm:
            merged[norm] = max(merged.get(norm, -1e9), np.log(max(prob, 1e-8)))
    for text, prob, tok_len in wp_beams:
        norm = normalize_text(text).strip()
        if not norm:
            continue
        lm_bonus = 0.0
        if CHAR_LM is not None:
            hist = ""
            for ch in norm:
                if ch in CHAR_TO_ID and ch != "_":
                    lm_bonus += 0.1 * CHAR_LM.log_prob(hist, ch)
                    hist += ch
        score = 1.15 * np.log(max(prob, 1e-8)) + lm_bonus - 0.02 * abs(tok_len - max(1, len(norm.split())))
        merged[norm] = max(merged.get(norm, -1e9), score)
    ranked = sorted(merged.items(), key=lambda x: -x[1])[:nbest]
    if not ranked:
        return [("", 1.0)]
    scores = torch.tensor([score for _, score in ranked], dtype=torch.float32)
    probs = torch.softmax(scores - scores.max(), dim=0).tolist()
    return [(text, float(prob)) for (text, _), prob in zip(ranked, probs)]


def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    for d in (zuco_v1, zuco_v2):
        d.mkdir(parents=True, exist_ok=True)

    expected = {
        zuco_v1 / "resultsZAB_SR.mat",
        zuco_v1 / "resultsZDM_SR.mat",
        zuco_v1 / "resultsZAB_NR.mat",
        zuco_v1 / "resultsZDM_NR.mat",
        zuco_v2 / "resultsYAC_NR.mat",
        zuco_v2 / "resultsYAG_NR.mat",
        zuco_v2 / "resultsYAK_NR.mat",
    }
    if all(verify_mat_file(path) for path in expected):
        print(f"[ZuCo] cache ready: {len(expected)}/{len(expected)} .mat files")
        return

    def fetch(project_id, target_dir, names):
        res = __import__("subprocess").run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True
        )
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if verify_mat_file(local):
                continue
            remote = next((p for p in paths if p.strip().endswith(name)), None)
            if remote:
                print(f"  Fetching {name}...")
                __import__("subprocess").run(
                    ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                    timeout=900,
                )

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"])
    vol.commit()


def load_mat_any(path):
    import scipy.io as sio
    import h5py

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data["sentenceData"]
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for s in sentences:
            if hasattr(s, "rawData") and hasattr(s, "content") and isinstance(s.rawData, np.ndarray):
                yield s.rawData, str(s.content)
    except Exception:
        try:
            with h5py.File(path, "r") as f:
                if "sentenceData" in f:
                    for key in f["sentenceData"].keys():
                        s = f["sentenceData"][key]
                        if "rawData" in s and "content" in s:
                            yield np.array(s["rawData"]).T, "hdf5_content_placeholder"
        except Exception:
            pass


def normalize_eeg(eeg, target_ch=64, target_t=TIME_STEPS):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None
    ch, t = eeg.shape
    if ch > t:
        eeg = eeg.T
        ch, t = eeg.shape
    if ch > target_ch:
        eeg = eeg[:target_ch, :]
    elif ch < target_ch:
        eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:
        eeg = eeg[:, :target_t]
    elif t < target_t:
        eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    mu = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt

    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo / 64.0, hi / 64.0], btype="bandpass")
            env = np.abs(filtfilt(b, a, data, axis=1))
            return np.convolve(env.flatten(), np.ones(5) / 5, mode="same").reshape(env.shape)
        except Exception:
            return np.zeros_like(data)

    theta = band_env(4, 8, eeg)
    alpha = band_env(8, 13, eeg)
    gamma = band_env(30, 80, eeg)
    combined = np.concatenate([theta.T, alpha.T, gamma.T], axis=1)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.samples = []
        p = Path(base_path)
        skipped = 0
        for zp in [p / "ZuCo_v1", p / "ZuCo_v2"]:
            for mat in zp.rglob("*.mat"):
                for raw, text in load_mat_any(mat):
                    if not text or "placeholder" in text.lower():
                        skipped += 1
                        continue
                    normed = normalize_eeg(raw)
                    if normed is not None:
                        self.samples.append({"eeg": normed, "text": normalize_text(text)})
        print(f"[Dataset] total={len(self.samples)} skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wp_ids = text_to_wp_ids(s["text"])
        return s["eeg"], s["text"], text_to_char_ids(s["text"]), wp_ids, build_slot_targets(wp_ids)


def collate_fn(batch):
    eegs, texts, char_ids_list, wp_ids_list, slot_targets = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(x) for x in char_ids_list], dtype=torch.long)
    max_char = max(char_lens) if max(char_lens) > 0 else 1
    char_pad = torch.zeros(len(batch), max_char, dtype=torch.long)
    for i, ids in enumerate(char_ids_list):
        if ids:
            char_pad[i, :len(ids)] = torch.tensor(ids)
    wp_lens = torch.tensor([len(x) for x in wp_ids_list], dtype=torch.long)
    max_wp = max(wp_lens) if max(wp_lens) > 0 else 1
    wp_pad = torch.zeros(len(batch), max_wp, dtype=torch.long)
    for i, ids in enumerate(wp_ids_list):
        if ids:
            wp_pad[i, :len(ids)] = torch.tensor(ids)
    slot_pad = torch.tensor(np.stack(slot_targets), dtype=torch.long)
    return eegs, list(texts), char_pad, char_lens, wp_pad, wp_lens, slot_pad


class SlotDecoder(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, MAX_SLOTS, dim) * 0.02)
        self.cross1 = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.cross2 = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, memory):
        q = self.queries.expand(memory.shape[0], -1, -1)
        attn, _ = self.cross1(q, memory, memory, need_weights=False)
        q = self.norm1(q + attn)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm2(q + self_out)
        attn2, _ = self.cross2(q, memory, memory, need_weights=False)
        q = self.norm3(q + attn2)
        q = q + self.ff(q)
        return self.out(q)


class EEGAudioWhisperSlot(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import WhisperForConditionalGeneration

        self.frontend = nn.Sequential(
            nn.Conv1d(EEG_DIM, LATENT_DIM, kernel_size=7, padding=3),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
        )
        self.time_steps = TIME_STEPS // 4
        self.pos = nn.Parameter(torch.randn(1, self.time_steps, LATENT_DIM) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.to_mel = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 80),
        )
        whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.whisper_encoder = whisper.model.encoder
        self.wp_head = nn.Linear(self.whisper_encoder.config.d_model, WP_VOCAB_SIZE)
        self.wp_loss_fn = nn.CTCLoss(blank=WP_BLANK_ID, zero_infinity=True)
        self.slot_decoder = SlotDecoder(self.whisper_encoder.config.d_model, WP_VOCAB_SIZE + 2)
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=WP_SLOT_PAD_ID)

    def encode_states(self, traj):
        x = traj.transpose(1, 2)
        x = self.frontend(x).transpose(1, 2)
        x = self.temporal(x + self.pos[:, : x.shape[1]])
        mel = self.to_mel(x).transpose(1, 2)
        mel = F.interpolate(mel, size=3000, mode="linear", align_corners=False)
        enc = self.whisper_encoder(input_features=mel).last_hidden_state
        return enc

    def forward(self, traj, wp_ids, wp_lens, slot_targets):
        states = self.encode_states(traj)
        wp_logits = self.wp_head(states).transpose(0, 1)
        slot_logits = self.slot_decoder(states)
        input_lens = torch.full((traj.shape[0],), states.shape[1], dtype=torch.long, device=traj.device)
        loss_wp = self.wp_loss_fn(F.log_softmax(wp_logits, dim=-1), wp_ids, input_lens, wp_lens)
        loss_slot = self.slot_loss_fn(slot_logits.reshape(-1, slot_logits.shape[-1]), slot_targets.reshape(-1))
        return {"loss_wp": loss_wp, "loss_slot": loss_slot}

    @torch.no_grad()
    def decode(self, traj, nbest=5):
        states = self.encode_states(traj.to(next(self.parameters()).dtype))
        wp_logits = self.wp_head(states).transpose(0, 1)
        slot_logits = self.slot_decoder(states)
        wp_groups = beam_ctc_decode_wp_nbest(wp_logits, beam_width=12, nbest=nbest)
        slot_ids = slot_logits.argmax(dim=-1).cpu().tolist()
        slot_texts = []
        for seq in slot_ids:
            toks = []
            for tok in seq:
                if tok == WP_SLOT_EOS_ID:
                    break
                if tok == WP_SLOT_PAD_ID:
                    continue
                if tok > 0 and tok <= WP_VOCAB_SIZE - 1:
                    toks.append(tok)
            slot_texts.append(decode_wp_ids(toks))
        merged = []
        for slot_text, wp_beams in zip(slot_texts, wp_groups):
            merged.append(merge_char_wp_candidates([(slot_text, 0.9)], wp_beams, nbest=nbest))
        return merged, slot_texts, wp_groups


def set_trainable(model, freeze_whisper_body):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.frontend.parameters():
        p.requires_grad = True
    for p in model.temporal.parameters():
        p.requires_grad = True
    for p in model.to_mel.parameters():
        p.requires_grad = True
    for p in model.wp_head.parameters():
        p.requires_grad = True
    for p in model.slot_decoder.parameters():
        p.requires_grad = True
    if not freeze_whisper_body:
        for p in model.whisper_encoder.layers[-2:].parameters():
            p.requires_grad = True


def make_optimizer(model, steps_per_epoch, epochs, freeze_whisper_body):
    set_trainable(model, freeze_whisper_body)
    groups = []

    def add(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01})

    add(model.frontend.parameters(), 3e-4)
    add(model.temporal.parameters(), 2e-4)
    add(model.to_mel.parameters(), 2e-4)
    add(model.wp_head.parameters(), 2e-4)
    add(model.slot_decoder.parameters(), 2e-4)
    if not freeze_whisper_body:
        add(model.whisper_encoder.layers[-2:].parameters(), 1.2e-5)

    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,
    )
    return optimizer, scheduler


def validate(model, val_loader, epoch, ckpt_prefix, best_wer):
    device = next(model.parameters()).device
    model.eval()
    cer_scores, wer_scores, oracle_wer_scores = [], [], []
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for eegs, texts, _, _, _, _, _ in val_loader:
            merged, slot_texts, wp_groups = model.decode(eegs.to(device).to(torch.bfloat16), nbest=5)
            for merged_beams, slot_text, wp_beams, ref in zip(merged, slot_texts, wp_groups, texts):
                top = merged_beams[0][0] if merged_beams else ""
                cer = compute_cer(top, ref)
                wer = compute_wer(top, ref)
                oracle = min(compute_wer(text, ref) for text, _ in merged_beams) if merged_beams else 2.0
                cer_scores.append(cer)
                wer_scores.append(wer)
                oracle_wer_scores.append(oracle)
                if shown < 4:
                    print(f"  REF: '{ref[:120]}'")
                    print(f"  TOP1: '{top[:120]}'  CER={cer:.2f} WER={wer:.2f} O-WER={oracle:.2f}")
                    print(f"  SLO: '{slot_text[:120]}'")
                    print(f"  WPC: '{wp_beams[0][0][:120] if wp_beams else ''}'")
                    shown += 1
            if shown >= 4:
                break
    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    mean_wer = float(np.mean(wer_scores)) if wer_scores else float("inf")
    mean_oracle_wer = float(np.mean(oracle_wer_scores)) if oracle_wer_scores else float("inf")
    print(f"  Mean CER: {mean_cer:.3f}  Mean WER: {mean_wer:.3f}  Oracle WER: {mean_oracle_wer:.3f}")
    torch.save(model.state_dict(), f"{ckpt_prefix}_ep{epoch}.pt")
    ckpt_vol.commit()
    if mean_wer < best_wer:
        best_wer = mean_wer
        torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
        ckpt_vol.commit()
        print(f"  ✓ New best checkpoint: WER={best_wer:.3f}")
    return best_wer


def train_stage(model, epochs, train_loader, val_loader, ckpt_prefix, start_epoch, best_wer, freeze_whisper_body, stage_name):
    device = next(model.parameters()).device
    optimizer, scheduler = make_optimizer(model, len(train_loader), epochs, freeze_whisper_body)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"── {stage_name} ──")
    print(f"  Trainable params: {trainable:.1f}M")

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        tot_char = tot_wp = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            eegs, _, _, _, wp_ids, wp_lens, slot_targets = batch
            eegs = eegs.to(device).to(torch.bfloat16)
            wp_ids = wp_ids.to(device)
            wp_lens = wp_lens.to(device)
            slot_targets = slot_targets.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(eegs, wp_ids, wp_lens, slot_targets)
                loss = (2.0 * out["loss_wp"] + 3.0 * out["loss_slot"]) / ACCUM_STEPS
            loss.backward()
            tot_wp += out["loss_wp"].item()
            tot_char += out["loss_slot"].item()
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
        print(f"Epoch {epoch:3d} | WPC:{tot_wp/n:.3f} SLOT:{tot_char/n:.3f} | LR0:{lr:.2e}")
        best_wer = validate(model, val_loader, epoch, ckpt_prefix, best_wer)
    return best_wer


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol},
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(epochs_p1: int = 2):
    download_zuco(Path("/data/EEG_Text"), data_vol)
    get_whisper_processor()

    ds = EEGDataset("/data/EEG_Text")
    n_train = int(0.9 * len(ds))
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    global SP_MODEL, WP_VOCAB_SIZE, CHAR_LM, WP_SLOT_EOS_ID, WP_SLOT_PAD_ID
    train_texts = [ds.samples[i]["text"] for i in train_idx]
    CHAR_LM = CharNGramLM(train_texts, order=3)
    SP_MODEL = build_sentencepiece_model(train_texts, model_prefix="/persist/v403_wordpiece", vocab_size=192)
    WP_VOCAB_SIZE = SP_MODEL.get_piece_size() + 1
    WP_SLOT_EOS_ID = WP_VOCAB_SIZE
    WP_SLOT_PAD_ID = WP_VOCAB_SIZE + 1

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    mkloader = lambda d, shuf: torch.utils.data.DataLoader(d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader = mkloader(train_ds, True)
    val_loader = mkloader(val_ds, False)

    model = EEGAudioWhisperSlot().to(torch.bfloat16).cuda()

    print("\n============================================================")
    print(" V403: EEG timeline -> audio frontend -> Whisper encoder -> slot decoder")
    print("       + grounded wordpiece CTC auxiliary")
    print(" FRESH START")
    print("============================================================\n")

    best_wer = float("inf")
    stage_a = min(1, epochs_p1)
    if stage_a > 0:
        best_wer = train_stage(
            model, stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v403",
            start_epoch=0,
            best_wer=best_wer,
            freeze_whisper_body=True,
            stage_name="Stage A: frozen Whisper encoder",
        )
    if epochs_p1 > stage_a:
        best_wer = train_stage(
            model, epochs_p1 - stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v403",
            start_epoch=stage_a,
            best_wer=best_wer,
            freeze_whisper_body=False,
            stage_name="Stage B: unfreeze late Whisper layers",
        )

    best_path = Path("/persist/v403_best.pt")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cuda"))
    torch.save(model.state_dict(), "/persist/v403_final.pt")
    ckpt_vol.commit()
    print("\n✓ V403 complete. Saved /persist/v403_final.pt")


@app.local_entrypoint()
def main(epochs_p1: int = 2):
    print(f"Launching V403: Phase1={epochs_p1}ep")
    run_pipeline.remote(epochs_p1=epochs_p1)
