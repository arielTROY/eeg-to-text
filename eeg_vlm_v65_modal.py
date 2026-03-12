"""
V65: Phonological-Fingerprint Contrastive Phase 1 + ByT5 Denoiser Phase 2
=========================================================================
What changed vs V61/V64:

[P1-1] DATASET CACHE  – normalized EEG tensors saved to /persist/v65_data_cache.pt
        on first run so subsequent runs skip mat-file parsing (~3-4 min saved).

[P1-2] PHONOLOGICAL FINGERPRINT LOSS (NEW, main novelty)
        For each training text, compute a 7-dim place-of-articulation (POA)
        profile: proportions of bilabial / labio-dental / alveolar / palatal /
        velar / glottal / vowel characters.
        Train an InfoNCE contrastive loss: EEG embedding must be closer to its
        own POA fingerprint than to other sentences' fingerprints.
        Hypothesis: if inner speech or reading activates articulatory cortex,
        the encoder should learn to separate sounds vs. just semantic clusters.

[P1-3] FROZEN VideoMAE BACKBONE – only last 2 of 12 encoder blocks are trainable.
        V61 trained all VideoMAE at 5e-5 with 1363 samples → overfit.
        Frozen backbone forces gradient into rasterizer, adapter, and heads.

[P1-4] WARM-START from V61 Phase 1 best checkpoint (best oracle WER baseline).

[P2]   ByT5-small denoiser (from V64) replaces Qwen: smaller, byte-level,
        better suited to short noisy sequences. Precomputes CTC beam pairs
        and caches them so restarts don't redo inference.

Validation reports: CER, WER, Oracle WER (beam of 16), consonant recall.
"""

import glob
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal infrastructure ───────────────────────────────────────────────────────

app      = modal.App("eeg-vlm-v65")
ckpt_vol = modal.Volume.from_name("bt-checkpoints-v65", create_if_missing=True)
v61_vol  = modal.Volume.from_name("bt-checkpoints-v61", create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data",     create_if_missing=True)


def _download_models():
    from transformers import VideoMAEModel, AutoModelForSeq2SeqLM, AutoTokenizer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    AutoTokenizer.from_pretrained("google/byt5-small")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "openneuro-py",
    ])
    .run_function(_download_models)
)

# ── Vocabulary ─────────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for c, i in CHAR_TO_ID.items()}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS      = set("aeiou")
CONSONANTS  = set("bcdfghjklmnpqrstvwxyz")

CLASS_BLANK = 0; CLASS_VOWEL = 1; CLASS_CONSONANT = 2
CLASS_DIGIT = 3; CLASS_OTHER = 4; CLASS_SPACE = 5
CLASS_VOCAB_SIZE = 6

NUM_EEG_BANDS  = 6
NUM_PHONO_DIMS = 7     # POA fingerprint dims
ACCUM_STEPS    = 8     # effective batch = 4 × 8 = 32

# Place-of-articulation groups (character-level approximation of phonology)
# Groups: bilabial, labio-dental, alveolar, palatal, velar, glottal, vowel
_POA_GROUPS = [
    set("bpmw"),          # 0 bilabial / labial
    set("fv"),            # 1 labio-dental
    set("dtnszlrc"),      # 2 alveolar  (c often /s/ or /k/, compromise)
    set("jy"),            # 3 palatal
    set("gkxq"),          # 4 velar
    set("h"),             # 5 glottal
    VOWELS,               # 6 vowel
]


def compute_phono_fingerprint(text: str) -> np.ndarray:
    """7-dim normalized POA profile for a text string."""
    counts = np.zeros(NUM_PHONO_DIMS, dtype=np.float32)
    total = 0
    for c in text.lower():
        if c not in CHAR_TO_ID:
            continue
        for i, grp in enumerate(_POA_GROUPS):
            if c in grp:
                counts[i] += 1
                total += 1
                break
    return counts / max(total, 1)

# ── CTC utilities ──────────────────────────────────────────────────────────────

def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]


def text_to_class_ids(text):
    ids, prev = [], None
    for c in text.lower():
        if c not in CHAR_TO_ID:
            continue
        if   c == " ":        cls = CLASS_SPACE
        elif c in VOWELS:     cls = CLASS_VOWEL
        elif c in CONSONANTS: cls = CLASS_CONSONANT
        elif c.isdigit():     cls = CLASS_DIGIT
        else:                 cls = CLASS_OTHER
        if cls != prev:
            ids.append(cls); prev = cls
    return ids


def greedy_ctc_decode(logits):
    ids = logits.argmax(dim=-1)
    results = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out, prev = "", -1
        for i in seq:
            if i != prev and i != 0:
                out += ID_TO_CHAR.get(i, "")
            prev = i
        results.append(out)
    return results


def beam_ctc_candidates(logits, beam_width=8, topk=4):
    log_probs = F.log_softmax(logits, dim=-1)
    T, B, V   = log_probs.shape
    results   = []
    for b in range(B):
        beams = [([], -1, 0.0)]
        for t in range(T):
            new_beams = {}
            frame_lp  = log_probs[t, b]
            top_vals, top_ids = frame_lp.topk(min(beam_width * 2, V))
            for seq, last_tok, score in beams:
                for val, tok_id in zip(top_vals.tolist(), top_ids.tolist()):
                    ns = score + val
                    if tok_id == 0:
                        key = tuple(seq)
                        if key not in new_beams or new_beams[key][2] < ns:
                            new_beams[key] = (seq, 0, ns)
                    elif tok_id == last_tok:
                        key = tuple(seq)
                        if key not in new_beams or new_beams[key][2] < ns:
                            new_beams[key] = (seq, tok_id, ns)
                    else:
                        new_seq = seq + [tok_id]
                        key = tuple(new_seq)
                        if key not in new_beams or new_beams[key][2] < ns:
                            new_beams[key] = (new_seq, tok_id, ns)
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:beam_width]
        top = []
        for seq, _, score in beams[:topk]:
            txt = "".join(ID_TO_CHAR.get(i, "") for i in seq).strip() or "..."
            top.append((txt, float(score)))
        while len(top) < topk:
            top.append(top[-1] if top else ("...", 0.0))
        results.append(top)
    return results


def beam_ctc_decode(logits, beam_width=8):
    return [c[0][0] for c in beam_ctc_candidates(logits, beam_width=beam_width, topk=1)]


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)


def compute_wer(pred, ref):
    import editdistance
    pw, rw = pred.strip().split(), ref.strip().split()
    if not rw: return 0.0 if not pw else 1.0
    return min(editdistance.eval(pw, rw) / len(rw), 2.0)

# ── EEG preprocessing ─────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False


def download_zuco(base_path, vol):
    import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"
    for d in [zuco_v1, zuco_v2]: d.mkdir(parents=True, exist_ok=True)

    def fetch(pid, tdir, names):
        missing = [n for n in names if not verify_mat_file(tdir / n)]
        if not missing:
            print(f"  Cached {tdir.name}"); return
        res   = subprocess.run(["osf", "-p", pid, "list"],
                               capture_output=True, text=True, timeout=120)
        paths = res.stdout.splitlines()
        for name in names:
            loc = tdir / name
            if verify_mat_file(loc): continue
            rem = next((p for p in paths if p.strip().endswith(name)), None)
            if rem:
                print(f"  Fetching {name}…")
                subprocess.run(["osf", "-p", pid, "fetch", rem.strip(), str(loc)],
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
        if not isinstance(sentences, (np.ndarray, list)): sentences = [sentences]
        for s in sentences:
            if (hasattr(s, 'rawData') and hasattr(s, 'content')
                    and isinstance(s.rawData, np.ndarray)):
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
        except: return np.zeros_like(data)

    alpha = band_env(8,  13,  eeg)
    beta  = band_env(13, 30,  eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gd    = np.diff(gamma, axis=1, prepend=gamma[:, :1])

    def rz(x):
        m = x.mean(axis=1, keepdims=True); s = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - m) / s

    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T, rz(delta).T, rz(gd).T], axis=1)
    return combined.astype(np.float32)  # (1024, 384)

# ── Dataset caching (NEW – avoids 3-4 min mat re-parse on every run) ──────────

def load_or_build_cache(base_path, vol):
    cache_path = Path("/persist/v65_data_cache.pt")
    if cache_path.exists():
        print("Loading cached EEG dataset …")
        return torch.load(cache_path, weights_only=False)

    print("Building EEG dataset cache from mat files …")
    eegs, texts, fingerprints = [], [], []
    skipped = 0
    for zp in [base_path / "ZuCo_v1", base_path / "ZuCo_v2"]:
        for mat in sorted(zp.rglob("*.mat")):
            for raw, text in load_mat_any(mat):
                if not text or "placeholder" in text.lower():
                    skipped += 1; continue
                normed = normalize_eeg(raw)
                if normed is None:
                    skipped += 1; continue
                eegs.append(normed)
                texts.append(text)
                fingerprints.append(compute_phono_fingerprint(text))

    print(f"[Cache] total={len(eegs)} skipped={skipped}")
    cache = {
        "eegs":         np.stack(eegs).astype(np.float32),
        "texts":        texts,
        "fingerprints": np.stack(fingerprints).astype(np.float32),
    }
    torch.save(cache, cache_path)
    vol.commit()
    print("Saved cache to /persist/v65_data_cache.pt")
    return cache

# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, eegs, texts, fingerprints, indices, augment=False):
        self.eegs = eegs; self.texts = texts; self.fps = fingerprints
        self.indices = indices; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        eeg = self.eegs[idx].copy()
        if self.augment:
            eeg = _augment(eeg)
        return (eeg, self.texts[idx],
                text_to_char_ids(self.texts[idx]),
                text_to_class_ids(self.texts[idx]),
                self.fps[idx])


def _augment(eeg):
    T, C = eeg.shape
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift] = 0
        else:         eeg[shift:] = 0
    eeg = eeg * np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * 0.04
    drop = np.random.choice(C, max(1, int(C * 0.08)), replace=False)
    eeg[:, drop] = 0
    for _ in range(2):
        span  = np.random.randint(16, 48)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0
    return eeg


def collate_phase1(batch):
    eegs, texts, char_ids_list, class_ids_list, fps = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    fps  = torch.from_numpy(np.stack(fps))
    char_lens  = torch.tensor([len(c) for c in char_ids_list],  dtype=torch.long)
    class_lens = torch.tensor([len(c) for c in class_ids_list], dtype=torch.long)
    mc  = max(char_lens).item()  or 1
    mc2 = max(class_lens).item() or 1
    char_pad  = torch.zeros(len(batch), mc,  dtype=torch.long)
    class_pad = torch.zeros(len(batch), mc2, dtype=torch.long)
    for i, (c, cl) in enumerate(zip(char_ids_list, class_ids_list)):
        if c:  char_pad[i, :len(c)]  = torch.tensor(c)
        if cl: class_pad[i, :len(cl)] = torch.tensor(cl)
    return eegs, list(texts), char_pad, char_lens, class_pad, class_lens, fps

# ── Model ──────────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    """Inverse-distance-weighted spatial interpolation from electrode signals."""
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands; self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w    = 1.0 / (dist + 1e-4) ** 2.0
        r    = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B * T, self.n_bands, self.n_electrodes)
        proj = xs @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B * T, C, H, W)).view(B, T, 3, H, W)


class EEGCTCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel
        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer    = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt      = ChannelAdapter(NUM_EEG_BANDS)
        self.video_enc     = VideoMAEModel(v_cfg)
        self.ctc_head      = nn.Linear(768, VOCAB_SIZE)
        self.class_ctc_head = nn.Linear(768, CLASS_VOCAB_SIZE)
        # Phonological fingerprint contrastive heads (NEW)
        self.phono_eeg_proj = nn.Sequential(
            nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
        self.phono_fp_proj  = nn.Sequential(
            nn.Linear(NUM_PHONO_DIMS, 32), nn.GELU(), nn.Linear(32, 64))
        self.ctc_loss_fn   = nn.CTCLoss(blank=0, zero_infinity=True)

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        # (B, 4096, 768) → (B, 256, 16, 768) → mean over 16 → (B, 256, 768)
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    def forward_phase1(self, traj, char_ids, char_lens, class_ids, class_lens, fingerprints):
        B, device = traj.shape[0], traj.device
        v_seq = self.encode(traj)  # (B, 256, 768)

        # ── Char CTC ──────────────────────────────────────────────────────────
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)   # (256, B, V)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)
        loss_ctc   = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)

        # ── Coarse class CTC ───────────────────────────────────────────────────
        cls_logits = self.class_ctc_head(v_seq).transpose(0, 1)
        loss_class = self.ctc_loss_fn(
            F.log_softmax(cls_logits, dim=-1), class_ids, input_lens, class_lens)

        # ── Diversity (blank-collapse prevention) ──────────────────────────────
        probs  = F.softmax(ctc_logits, dim=-1)
        mp     = probs.mean(dim=0)
        loss_div = -(mp * torch.log(mp + 1e-8)).sum(dim=-1).mean()

        # ── Phonological fingerprint InfoNCE (NEW) ────────────────────────────
        # Project mean EEG embedding and POA fingerprint to shared 64-D space.
        # InfoNCE: each sample is its own positive; others are negatives.
        v_mean  = v_seq.mean(dim=1)                                   # (B, 768)
        e_phono = F.normalize(self.phono_eeg_proj(v_mean), dim=-1)   # (B, 64)
        fp_emb  = F.normalize(
            self.phono_fp_proj(fingerprints.to(device, dtype=v_mean.dtype)),
            dim=-1)                                                   # (B, 64)
        # Temperature 0.15: not too sharp with small batches
        sims    = e_phono @ fp_emb.T / 0.15                          # (B, B)
        tgt     = torch.arange(B, device=device)
        loss_phono = (F.cross_entropy(sims, tgt) +
                      F.cross_entropy(sims.T, tgt)) / 2.0

        return loss_ctc, loss_class, loss_div, loss_phono

    @torch.no_grad()
    def decode_ctc(self, traj, beam_width=16, nbest=8):
        traj = traj.to(next(self.parameters()).dtype)
        v_seq  = self.encode(traj)
        logits = self.ctc_head(v_seq).transpose(0, 1)
        return beam_ctc_candidates(logits, beam_width=beam_width, topk=nbest)

# ── Warm-start ─────────────────────────────────────────────────────────────────

def freeze_videomae_except_last_n(video_enc, n_free=2):
    """Freeze all VideoMAE encoder layers except the last n_free."""
    layers = video_enc.encoder.layer
    for i, layer in enumerate(layers):
        freeze = (i < len(layers) - n_free)
        for p in layer.parameters():
            p.requires_grad = not freeze
    for p in video_enc.embeddings.parameters():
        p.requires_grad = False
    print(f"  VideoMAE: frozen layers 0-{len(layers)-n_free-1}, "
          f"trainable layers {len(layers)-n_free}-{len(layers)-1}")


def load_warm_start(model):
    for ckpt_path, label in [("/v61/v61_phase1_best.pt", "V6.1 Phase-1 best")]:
        if Path(ckpt_path).exists():
            print(f"Warm-starting from {label}: {ckpt_path}")
            src = torch.load(ckpt_path, map_location="cpu")
            dst = model.state_dict()
            prefixes = ("rasterizer.", "video_enc.", "ch_adapt.",
                        "ctc_head.", "class_ctc_head.")
            n_ok = n_skip = 0
            for k, v in src.items():
                if not any(k.startswith(p) for p in prefixes): continue
                if k in dst and dst[k].shape == v.shape:
                    dst[k] = v.clone(); n_ok += 1
                else:
                    n_skip += 1
            model.load_state_dict(dst, strict=False)
            print(f"  Loaded {n_ok} tensors, skipped {n_skip}")
            return True

    print("No V6.1 checkpoint found — pretrained VideoMAE only")
    from transformers import VideoMAEModel as _VMAE
    src = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    dst = model.video_enc.state_dict()
    n_ok = n_skip = 0
    for k, v in src.state_dict().items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v.clone(); n_ok += 1
        else: n_skip += 1
    model.video_enc.load_state_dict(dst, strict=False)
    print(f"  Transferred {n_ok} VideoMAE tensors"); return False

# ── Phase 1 training ──────────────────────────────────────────────────────────

def train_phase1(model, train_ds, val_ds, epochs, ckpt_prefix):
    device = next(model.parameters()).device

    # Freeze everything, then selectively enable
    for p in model.parameters(): p.requires_grad = False
    for m in [model.rasterizer, model.ch_adapt, model.ctc_head,
              model.class_ctc_head, model.phono_eeg_proj, model.phono_fp_proj]:
        for p in m.parameters(): p.requires_grad = True
    freeze_videomae_except_last_n(model.video_enc, n_free=2)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Phase 1 trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate_phase1, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=collate_phase1, num_workers=0)

    optimizer = torch.optim.AdamW([
        {"params": [p for m in [model.rasterizer, model.ch_adapt]
                    for p in m.parameters() if p.requires_grad], "lr": 3e-4},
        {"params": [p for m in [model.ctc_head, model.class_ctc_head,
                                 model.phono_eeg_proj, model.phono_fp_proj]
                    for p in m.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for p in model.video_enc.parameters()
                    if p.requires_grad], "lr": 2e-5},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[3e-4, 1e-4, 2e-5],
        steps_per_epoch=max(len(train_loader), 1),
        epochs=epochs, pct_start=0.1)

    best_ctc = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        tot = dict(ctc=0.0, cls=0.0, div=0.0, phn=0.0)
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eegs, texts, c_ids, c_lens, cl_ids, cl_lens, fps = batch
            eegs   = eegs.to(device, dtype=torch.bfloat16)
            c_ids  = c_ids.to(device);  c_lens  = c_lens.to(device)
            cl_ids = cl_ids.to(device); cl_lens = cl_lens.to(device)
            fps    = fps.to(device,  dtype=torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lc, lcl, ldiv, lphn = model.forward_phase1(
                    eegs, c_ids, c_lens, cl_ids, cl_lens, fps)
                # Weights: CTC x10, class CTC x3, phono contrastive x2, div x0.5
                loss = (10.0*lc + 3.0*lcl + 2.0*lphn + 0.5*ldiv) / ACCUM_STEPS

            loss.backward()
            tot["ctc"] += lc.item(); tot["cls"] += lcl.item()
            tot["div"] += ldiv.item(); tot["phn"] += lphn.item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        n   = max(len(train_loader), 1)
        lr0 = optimizer.param_groups[0]["lr"]
        print(f"Ep{epoch:2d} | CTC:{tot['ctc']/n:.3f} CLS:{tot['cls']/n:.3f} "
              f"PHN:{tot['phn']/n:.3f} DIV:{tot['div']/n:.3f} | LR:{lr0:.1e}")

        if tot["ctc"] / n < best_ctc:
            best_ctc = tot["ctc"] / n
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
            print(f"  ✓ New best CTC {best_ctc:.3f}")

        if epoch % 3 == 0 or epoch == epochs:
            _validate_phase1(model, val_loader, device, epoch)
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_ep{epoch}.pt")
            ckpt_vol.commit()


def _validate_phase1(model, val_loader, device, epoch):
    model.eval()
    cers, wers, oracle_wers = [], [], []
    v_correct = v_total = c_correct = c_total = 0
    print(f"── Val Epoch {epoch} ──")
    shown = 0
    with torch.no_grad():
        for batch in val_loader:
            eegs, texts, *_ = batch
            eegs = eegs.to(device, dtype=torch.bfloat16)
            beam_groups = model.decode_ctc(eegs, beam_width=16, nbest=8)
            for bgs, ref in zip(beam_groups, texts):
                ref_lower = ref.lower()
                top1 = bgs[0][0]
                cers.append(compute_cer(top1, ref_lower))
                wers.append(compute_wer(top1, ref_lower))
                oracle_wers.append(
                    min(compute_wer(b[0], ref_lower) for b in bgs))
                for c in ref_lower:
                    if   c in VOWELS:     v_total += 1; v_correct += int(c in top1)
                    elif c in CONSONANTS: c_total += 1; c_correct += int(c in top1)
                if shown < 4:
                    oracle_txt = min(bgs, key=lambda b: compute_wer(b[0], ref_lower))[0]
                    print(f"  REF: '{ref[:80]}'")
                    print(f"  TOP: '{top1[:80]}'  CER={cers[-1]:.2f} WER={wers[-1]:.2f}")
                    print(f"  ORA: '{oracle_txt[:80]}'  OracleWER={oracle_wers[-1]:.2f}")
                    shown += 1
    if cers:
        print(f"  Mean CER:{np.mean(cers):.3f}  WER:{np.mean(wers):.3f}  "
              f"OracleWER:{np.mean(oracle_wers):.3f}")
        print(f"  VowelRecall:{v_correct/max(v_total,1):.2f}  "
              f"ConsRecall:{c_correct/max(c_total,1):.2f}")
    model.train()

# ── ByT5 denoiser (Phase 2, from V64) ─────────────────────────────────────────

class ByT5Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/byt5-small", torch_dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask,
                 max_new_tokens=96, num_beams=6, num_return_sequences=4):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=4, repetition_penalty=1.20,
            length_penalty=0.55, early_stopping=True,
            do_sample=False, return_dict_in_generate=True, output_scores=True)


def ctc_like_corrupt(text, severity=1.0):
    out = []
    for c in text.lower():
        if c not in CHAR_TO_ID: continue
        r = np.random.random()
        if c in VOWELS:
            if r < 0.08 * severity: continue
            out.append(c)
            if np.random.random() < 0.04 * severity: out.append(c)
        elif c in CONSONANTS:
            if   r < 0.45 * severity: continue
            elif r < 0.60 * severity: out.append(np.random.choice(list(CONSONANTS)))
            else: out.append(c)
        elif c == " ":
            out.append(" " if r > 0.20 * severity else "  ")
        elif c in ".!?":
            if r > 0.50 * severity: out.append(".")
        else:
            if r > 0.35 * severity: out.append(c)
    return " ".join("".join(out).split()) or "..."


def format_beams(beams):
    """Format top-3 beam candidates as ByT5 input."""
    parts = []
    for i, (txt, score) in enumerate(beams[:3], start=1):
        delta = score - beams[0][1] if i > 1 else 0.0
        parts.append(f"<b{i}>{(txt or '...').strip()}</b{i}><s{i}>{delta:.1f}</s{i}>")
    return " ".join(parts)


class DenoiserPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, repeats=6, synth_prob=0.30):
        self.pairs = [p for p in pairs if p["text"].strip()]
        self.repeats = repeats; self.synth_prob = synth_prob

    def __len__(self): return len(self.pairs) * self.repeats

    def __getitem__(self, i):
        p = self.pairs[i % len(self.pairs)]
        beams = list(p["beams"]); scores = list(p.get("scores", [0.0, -0.5, -1.0]))
        if np.random.random() < self.synth_prob:
            s = np.random.uniform(0.75, 1.25)
            beams  = [ctc_like_corrupt(p["text"], s * (1 + 0.1*k)) for k in range(3)]
            scores = [0.0, -0.4, -0.8]
        src = format_beams(list(zip(beams, scores)))
        return src, p["text"]


def collate_denoiser(batch, tokenizer, max_len=192):
    sources, targets = zip(*batch)
    enc = tokenizer(list(sources), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    tgt = tokenizer(text_target=list(targets), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    labs = tgt.input_ids.clone()
    labs[labs == tokenizer.pad_token_id] = -100
    return enc.input_ids, enc.attention_mask, labs, list(targets), list(sources)


def precompute_pairs(ctc_model, eegs, texts, indices, label):
    device = next(ctc_model.parameters()).device
    pairs  = []
    for i in range(0, len(indices), 4):
        idx_batch = indices[i:i+4]
        eeg_batch = torch.from_numpy(eegs[idx_batch]).to(device, dtype=torch.bfloat16)
        beam_groups = ctc_model.decode_ctc(eeg_batch, beam_width=8, nbest=3)
        for bgs, idx in zip(beam_groups, idx_batch):
            pairs.append({
                "text":   texts[idx],
                "beams":  [b[0] for b in bgs],
                "scores": [b[1] for b in bgs],
            })
        if (i // 4) % 50 == 0:
            print(f"  [{label}] {i//4}/{len(indices)//4} batches")
    return pairs


def train_denoiser(denoiser, tokenizer, train_pairs, val_pairs, ckpt_prefix, epochs, start_epoch=0):
    device = next(denoiser.parameters()).device

    train_ds = DenoiserPairDataset(train_pairs, repeats=6, synth_prob=0.30)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_denoiser(b, tokenizer), num_workers=0)

    # Val: no augmentation, single pass
    val_ds = DenoiserPairDataset(val_pairs, repeats=1, synth_prob=0.0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=lambda b: collate_denoiser(b, tokenizer), num_workers=0)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_gen_wer = float("inf")

    for local_ep in range(1, epochs + 1):
        epoch = start_epoch + local_ep
        denoiser.train(); total_loss = 0.0; optimizer.zero_grad()

        for step, (inp, mask, labs, _, _) in enumerate(train_loader, 1):
            inp  = inp.to(device); mask = mask.to(device); labs = labs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = denoiser(inp, mask, labs)
                (out.loss / 4).backward()
            total_loss += out.loss.item()
            if step % 4 == 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
        if len(train_loader) % 4:
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()
        print(f"Ep{epoch:2d} | LM:{total_loss/max(len(train_loader),1):.3f}")

        # Validation
        denoiser.eval()
        raw_wers, gen_wers = [], []
        shown = 0
        with torch.no_grad():
            for inp, mask, _, targets, sources in val_loader:
                gen = denoiser.generate(
                    inp.to(device), mask.to(device),
                    num_beams=6, num_return_sequences=4)
                preds = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
                for i, (ref, src) in enumerate(zip(targets, sources)):
                    ref_lower = ref.lower()
                    cands     = preds[i*4:(i+1)*4]
                    # Top-1 is the fair metric; oracle for reference
                    top1 = (cands[0] if cands else src).strip().lower()
                    gen_wers.append(compute_wer(top1, ref_lower))
                    # Raw CTC hint = first beam tag
                    hint_end = src.find("</b1>")
                    raw_hint = src[:hint_end].replace("<b1>", "").strip().lower()
                    raw_wers.append(compute_wer(raw_hint, ref_lower))
                    if shown < 3:
                        print(f"  REF: '{ref[:80]}'")
                        print(f"  RAW: '{raw_hint[:60]}'  GEN: '{top1[:60]}'  WER={gen_wers[-1]:.2f}")
                        shown += 1

        m_raw = np.mean(raw_wers); m_gen = np.mean(gen_wers)
        print(f"  RawWER:{m_raw:.3f}  GenWER:{m_gen:.3f}")
        torch.save(denoiser.state_dict(), f"{ckpt_prefix}_denoiser_ep{epoch}.pt")
        if m_gen < best_gen_wer:
            best_gen_wer = m_gen
            torch.save(denoiser.state_dict(), f"{ckpt_prefix}_denoiser_best.pt")
            print(f"  ✓ New best GenWER {best_gen_wer:.3f}")
        ckpt_vol.commit()

# ── Modal pipeline ─────────────────────────────────────────────────────────────

@app.function(
    image=image, gpu="H100", timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol, "/v61": v61_vol},
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p1: int = 6, epochs_p2: int = 12):
    from transformers import AutoTokenizer

    download_zuco(Path("/data/EEG_Text"), data_vol)

    # ── Dataset (cached after first run) ──────────────────────────────────────
    cache = load_or_build_cache(Path("/data/EEG_Text"), data_vol)
    eegs, texts, fps_arr = cache["eegs"], cache["texts"], cache["fingerprints"]
    print(f"Dataset: {len(texts)} samples")

    np.random.seed(42)
    indices = np.random.permutation(len(texts)).tolist()
    n_train = int(0.9 * len(indices))
    train_idx = indices[:n_train]; val_idx = indices[n_train:]

    train_ds = EEGDataset(eegs, texts, fps_arr, train_idx, augment=True)
    val_ds   = EEGDataset(eegs, texts, fps_arr, val_idx,   augment=False)

    # ── Phase 1: phonological-contrastive CTC encoder ────────────────────────
    model  = EEGCTCEncoder().to(torch.bfloat16).cuda()
    load_warm_start(model)

    p1_done   = Path("/persist/v65_phase1_done")
    best_p1   = Path("/persist/v65_phase1_best.pt")
    p1_ckpts  = sorted(glob.glob("/persist/v65_phase1_ep*.pt"))
    p1_start  = 0

    if p1_done.exists():
        print("Phase 1 already complete, loading best checkpoint")
        model.load_state_dict(torch.load(best_p1, map_location="cuda"))
    elif p1_ckpts:
        last     = p1_ckpts[-1]
        p1_start = int(last.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last, map_location="cuda"))
        print(f"Resuming Phase 1 from epoch {p1_start}")

    print("\n" + "="*60)
    print(" V65: Phonological-Contrastive Phase 1 + ByT5 Denoiser")
    print(f" epochs_p1={epochs_p1}  epochs_p2={epochs_p2}")
    print("="*60 + "\n")

    if not p1_done.exists():
        remaining = epochs_p1 - p1_start
        if remaining > 0:
            train_phase1(model, train_ds, val_ds, remaining, "/persist/v65")
        p1_done.touch()
        ckpt_vol.commit()
        if best_p1.exists():
            model.load_state_dict(torch.load(best_p1, map_location="cuda"))

    # ── Phase 2: ByT5 denoiser ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    denoiser  = ByT5Denoiser().cuda()
    d_ckpts   = sorted(glob.glob("/persist/v65_denoiser_ep*.pt"))
    final_path = Path("/persist/v65_final.pt")

    if final_path.exists():
        print("V65 already complete."); return

    d_start = 0
    if d_ckpts:
        d_start = int(d_ckpts[-1].split("_ep")[-1].replace(".pt", ""))
        denoiser.load_state_dict(torch.load(d_ckpts[-1], map_location="cuda"))
        print(f"Resuming denoiser from epoch {d_start}")

    # Precompute and cache CTC beam pairs
    pairs_cache = Path("/persist/v65_pairs.pt")
    if pairs_cache.exists():
        print("Loading cached CTC pairs …")
        pair_data   = torch.load(pairs_cache, weights_only=False)
        train_pairs = pair_data["train"]; val_pairs = pair_data["val"]
    else:
        print("Precomputing CTC beam pairs …")
        model.eval()
        train_pairs = precompute_pairs(model, eegs, texts, train_idx, "train")
        val_pairs   = precompute_pairs(model, eegs, texts, val_idx,   "val")
        torch.save({"train": train_pairs, "val": val_pairs}, pairs_cache)
        ckpt_vol.commit()
        print(f"Saved {len(train_pairs)} train / {len(val_pairs)} val pairs")

    del model; torch.cuda.empty_cache()

    remaining_p2 = epochs_p2 - d_start
    if remaining_p2 > 0:
        train_denoiser(denoiser, tokenizer, train_pairs, val_pairs,
                       "/persist/v65", remaining_p2, start_epoch=d_start)

    best_d = Path("/persist/v65_denoiser_best.pt")
    if best_d.exists():
        denoiser.load_state_dict(torch.load(best_d, map_location="cuda"))
    torch.save(denoiser.state_dict(), final_path)
    ckpt_vol.commit()
    print("\n✓ V65 complete. Saved /persist/v65_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 6, epochs_p2: int = 12):
    if mode == "pipeline":
        print(f"Launching V65: P1={epochs_p1}ep  P2={epochs_p2}ep")
        run_pipeline.remote(epochs_p1=epochs_p1, epochs_p2=epochs_p2)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
