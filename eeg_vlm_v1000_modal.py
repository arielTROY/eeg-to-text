"""
V1000: Two-Stage Training + Diversity Loss on VideoMAE Output
=============================================================
ROOT CAUSE DIAGNOSIS (V900):
  V900 div_loss computed on q = F.normalize(sem_proj(v_mean)) — a 128-dim L2-norm vector.
  Training showed div=1.000 every step: all batch elements had identical q (collapsed sem_proj).
  sem_proj is a linear layer — it CANNOT produce different outputs for the same input.
  VideoMAE was outputting near-identical v_mean for all EEG signals (the real collapse).
  The sem_proj bottleneck masked the actual diversity of enc (256×768 position-wise features),
  which is WHY validation diversity (23.4%) was higher than training div suggested.

V1000 FIXES:
  1. div_loss computed on v_mean (raw VideoMAE pooled output, 768-dim) — directly penalizes
     VideoMAE from collapsing, no bottleneck. Gradient flows: div_loss → v_mean → VideoMAE.
  2. Two-stage training:
     Stage 1 (ep 1-15): lm=0.0, align=3.0, div_enc=2.0, sent=0.3, len=0.1
       → VideoMAE forced to be discriminative BEFORE T5 memorizes the distribution
     Stage 2 (ep 16+):  lm=0.15, align=1.5, div_enc=1.0, sent=0.2, len=0.1
       → LM added gently; discriminative features already established
  3. Warm-start from V700 ep5 (best diversity ever: 30.2%) — NOT V800/V900 (already collapsed)

ARCHITECTURE (unchanged from V700/V800/V900):
  EEG (1024, 384)
    → MultiScaleRasterizer (topographic video 6-band, 64×64)
    → ChannelAdapter (6→3 RGB)
    → VideoMAE-base (12 layers, differential LR)
    → v_seq (B, 256, 768)

  Path A — alignment:
    v_mean = v_seq.mean(1) → sem_proj (768→128 L2) → InfoNCE vs SBERT
    v_mean normalized → div_loss (NEW: penalize VideoMAE collapse directly)
    q → sentiment_head, length_head

  Path B — generation:
    v_seq → eeg_to_t5 (768→768) → Frozen Flan-T5-base cross-attention at all layers

STAGE 1 RATIONALE (SemKey insight):
  With 1363 samples, the LM decoder memorizes the training distribution in <5 epochs.
  Once memorized, EEG becomes irrelevant (GPT-style prior dominates).
  Stage 1 trains VideoMAE alignment WITHOUT letting the decoder interfere.
  After 15 epochs of InfoNCE + div_enc, VideoMAE features are maximally discriminative.
  Stage 2 then connects these features to text generation with a small LM weight.
"""

from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ──────────────────────────────────────────────────────────────────────

app      = modal.App("eeg-vlm-v1000")
ckpt_vol = modal.Volume.from_name("bt-checkpoints-v1000", create_if_missing=True)
v61_vol  = modal.Volume.from_name("bt-checkpoints-v61",   create_if_missing=False)
v600_vol = modal.Volume.from_name("bt-checkpoints-v600",  create_if_missing=False)
v700_vol = modal.Volume.from_name("bt-checkpoints-v700",  create_if_missing=False)
v800_vol = modal.Volume.from_name("bt-checkpoints-v800",  create_if_missing=False)
v900_vol = modal.Volume.from_name("bt-checkpoints-v900",  create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data",        create_if_missing=True)


def _download_models():
    from transformers import VideoMAEModel, T5ForConditionalGeneration, T5Tokenizer
    from sentence_transformers import SentenceTransformer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    T5Tokenizer.from_pretrained("google/flan-t5-base")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "openneuro-py",
        "sentence-transformers", "vaderSentiment",
    ])
    .run_function(_download_models)
)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_EEG_BANDS = 6
ACCUM_STEPS   = 8
QUEUE_SIZE    = 4096
SEM_DIM       = 128
SBERT_DIM     = 768
T5_DIM        = 768      # flan-t5-base d_model
EEG_SEQ_LEN   = 256      # VideoMAE output after reshape+mean
MAX_TEXT_LEN  = 80

STAGE2_START  = 15       # switch from Stage 1 → Stage 2 after this epoch

# ── EEG preprocessing (identical to V65/V100/V600 — reuses cache) ─────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False


def download_zuco(base_path, vol):
    import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"
    if (zuco_v1.exists() and any(zuco_v1.rglob("*.mat")) and
            zuco_v2.exists() and any(zuco_v2.rglob("*.mat"))):
        print("ZuCo already downloaded."); return
    print("Downloading ZuCo via osf …")
    for cmd in [
        ["osf", "-p", "q3zws", "clone", str(zuco_v1)],
        ["osf", "-p", "2urht", "clone", str(zuco_v2)],
    ]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0: print(f"osf warn: {r.stderr[:200]}")
    vol.commit()


def load_mat_any(path):
    try:
        import scipy.io as sio
        raw = sio.loadmat(str(path), simplify_cells=True)
    except Exception:
        try:
            import h5py
            with h5py.File(str(path), 'r') as f:
                raw = {k: np.array(f[k]) for k in f.keys()}
        except Exception:
            return []
    results = []
    for key in raw:
        if not isinstance(raw[key], dict): continue
        block = raw[key]
        eeg_keys = [k for k in block if 'rawData' in k or k == 'rawData']
        word_keys = [k for k in block if 'word' in k.lower() or 'content' in k.lower()]
        if not eeg_keys or not word_keys: continue
        eeg   = np.array(block[eeg_keys[0]], dtype=np.float32)
        words = block[word_keys[0]]
        if isinstance(words, np.ndarray):
            words = [str(w) for w in words.flatten() if w]
        elif isinstance(words, str):
            words = [words]
        else:
            words = [str(words)]
        text = " ".join(words).strip()
        if len(text) < 5: continue
        results.append((eeg, text))
    return results


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if eeg.ndim == 3: eeg = eeg.mean(0)
    if eeg.ndim != 2: return None
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    T, C = eeg.shape
    if C > target_ch:
        eeg = eeg[:, :target_ch]
    elif C < target_ch:
        pad = np.zeros((T, target_ch - C), dtype=np.float32)
        eeg = np.concatenate([eeg, pad], axis=1)
    if T > target_t:
        eeg = eeg[:target_t]
    elif T < target_t:
        pad = np.zeros((target_t - T, target_ch), dtype=np.float32)
        eeg = np.concatenate([eeg, pad], axis=0)
    bands = []
    try:
        import mne
        info = mne.create_info(target_ch, 500., 'eeg')
        raw  = mne.io.RawArray(eeg.T, info, verbose=False)
        raw.filter(0.5, 40., fir_design='firwin', verbose=False)
        data = raw.get_data().T
        bands.append(data)
        for lo, hi in [(8,13),(13,30),(30,45),(1,4),(30,45)]:
            filt = raw.copy()
            filt.filter(lo, hi, fir_design='firwin', verbose=False)
            d = filt.get_data().T
            if hi == 30 and lo == 1:
                d = np.diff(d, axis=0, prepend=d[:1])
            elif hi == 45 and lo == 30 and len(bands) > 4:
                d = np.diff(d, axis=0, prepend=d[:1])
            bands.append(d)
        result = np.concatenate(bands, axis=1).astype(np.float32)
    except Exception:
        result = np.tile(eeg, (1, NUM_EEG_BANDS)).astype(np.float32)
    mu = result.mean(0, keepdims=True); sig = result.std(0, keepdims=True) + 1e-6
    return ((result - mu) / sig).astype(np.float32)


def load_or_build_cache(base_path, vol):
    for cp in ["/persist/v65_data_cache.pt", "/v600_ckpt/v65_data_cache.pt"]:
        cache_path = Path(cp)
        if cache_path.exists():
            print(f"EEG cache hit at {cp} …")
            return torch.load(cache_path, weights_only=False)
    cache_path = Path("/persist/v65_data_cache.pt")
    print("Building EEG cache …")
    eegs, texts, skipped = [], [], 0
    for zp in [base_path / "ZuCo_v1", base_path / "ZuCo_v2"]:
        for mat in sorted(zp.rglob("*.mat")):
            for raw, text in load_mat_any(mat):
                if not text or "placeholder" in text.lower():
                    skipped += 1; continue
                n = normalize_eeg(raw)
                if n is None: skipped += 1; continue
                eegs.append(n); texts.append(text)
    cache = {"eegs": np.stack(eegs).astype(np.float32), "texts": texts,
             "fingerprints": np.zeros((len(eegs), 7), dtype=np.float32)}
    torch.save(cache, cache_path); vol.commit()
    print(f"Cache: {len(eegs)} samples, {skipped} skipped")
    return cache


# ── Augmentation ───────────────────────────────────────────────────────────────

def _augment(eeg):
    T, C = eeg.shape
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift] = 0
        else:         eeg[shift:] = 0
    eeg *= np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)
    eeg += np.random.randn(*eeg.shape).astype(np.float32) * 0.04
    eeg[:, np.random.choice(C, max(1, int(C * 0.10)), replace=False)] = 0
    for _ in range(3):
        span  = np.random.randint(16, 64)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0
    return eeg


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_labels(texts, t5_tokenizer):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiments = np.array(
            [1 if sia.polarity_scores(t)['compound'] >= 0.05 else 0 for t in texts],
            dtype=np.float32)
    except Exception:
        sentiments = np.zeros(len(texts), dtype=np.float32)
    lengths = np.array(
        [np.log1p(len(t5_tokenizer(t, return_tensors='pt').input_ids[0]))
         for t in texts],
        dtype=np.float32)
    return sentiments, lengths


class EEGTextDataset(torch.utils.data.Dataset):
    def __init__(self, eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                 sentiments, lengths, indices, augment=False):
        self.eegs           = eegs
        self.texts          = texts
        self.sbert_np       = sbert_np
        self.input_ids_list = input_ids_list
        self.attn_mask_list = attn_mask_list
        self.sentiments     = sentiments
        self.lengths        = lengths
        self.indices        = indices
        self.augment        = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        gi  = self.indices[i]
        eeg = self.eegs[gi].copy()
        if self.augment: eeg = _augment(eeg)
        return (eeg, self.texts[gi], self.sbert_np[gi],
                self.input_ids_list[gi], self.attn_mask_list[gi],
                self.sentiments[gi], self.lengths[gi])


def collate_fn(batch):
    eegs, texts, sberts, ids_list, mask_list, sents, lens = zip(*batch)
    eegs    = torch.from_numpy(np.stack(eegs))
    sberts  = torch.from_numpy(np.stack(sberts))
    sents   = torch.tensor(sents, dtype=torch.float32)
    lens    = torch.tensor(lens,  dtype=torch.float32)
    max_len = max(ids.shape[0] for ids in ids_list)
    B = len(batch)
    padded_ids  = torch.full((B, max_len), -100, dtype=torch.long)
    padded_attn = torch.zeros(B, max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = ids.shape[0]
        padded_ids[i, :L]  = ids
        padded_attn[i, :L] = mask
    return eegs, list(texts), sberts, padded_ids, padded_attn, sents, lens


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
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
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B * T, self.n_bands, self.n_electrodes)
        return (xs @ self.W.T.to(x.device, x.dtype)).reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_ch=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B*T, C, H, W)).view(B, T, 3, H, W)


class EEGV1000(nn.Module):
    """
    V1000: Two-stage training + diversity loss directly on VideoMAE output.

    KEY CHANGE vs V900:
      div_loss computed on v_mean = v_seq.mean(1)  (raw 768-dim VideoMAE pool)
      NOT on q = sem_proj(v_mean) (collapsed 128-dim bottleneck)
      → gradient flows directly into VideoMAE weights

    Trainable:
      - VideoMAE (all 12 layers, differential LR)
      - ChannelAdapter, rasterizer (adapter LR)
      - sem_proj, txt_proj (alignment, adapter LR)
      - eeg_to_t5 (VideoMAE→T5 projection, adapter LR)
      - sentiment_head, length_head (auxiliary, adapter LR)
    Frozen:
      - Flan-T5-base encoder + decoder (100%)
    """
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, T5ForConditionalGeneration

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)

        # Path A: alignment + auxiliary
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256), nn.GELU(),
            nn.Linear(256, SEM_DIM))
        self.txt_proj = nn.Sequential(
            nn.Linear(SBERT_DIM, 256), nn.GELU(),
            nn.Linear(256, SEM_DIM))

        # MoCo memory bank
        self.register_buffer("queue",
            F.normalize(torch.randn(QUEUE_SIZE, SEM_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.sentiment_head = nn.Linear(SEM_DIM, 1)
        self.length_head    = nn.Linear(SEM_DIM, 1)

        # Path B: generation (T5 cross-attention)
        self.eeg_to_t5 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, T5_DIM),
            nn.GELU(),
            nn.Linear(T5_DIM, T5_DIM))

        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        for p in self.t5.parameters():
            p.requires_grad = False

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state  # (B, 4096, 768)
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)   # (B, 256, 768)

    def get_eeg_enc(self, traj):
        v_seq  = self.encode(traj)                                # (B, 256, 768)
        v_mean = v_seq.mean(1)                                    # (B, 768) raw pool
        q      = F.normalize(self.sem_proj(v_mean), dim=-1)      # (B, 128) L2-norm
        enc    = self.eeg_to_t5(v_seq)                            # (B, 256, 768)
        return q, enc, v_mean                                     # also return raw v_mean

    @torch.no_grad()
    def _enqueue(self, k):
        B   = k.shape[0]
        ptr = int(self.queue_ptr.item())
        end = ptr + B
        if end <= QUEUE_SIZE:
            self.queue.data[ptr:end] = k.detach().float()
        else:
            p1 = QUEUE_SIZE - ptr
            self.queue.data[ptr:]   = k[:p1].detach().float()
            self.queue.data[:B-p1] = k[p1:].detach().float()
        self.queue_ptr.data[0] = end % QUEUE_SIZE

    def forward(self, traj, sbert_embs, target_ids, target_mask,
                sentiment_labels, length_labels):
        """
        Returns individual losses (unweighted). Training loop applies stage-dependent weights.
        """
        from transformers.modeling_outputs import BaseModelOutput

        B, device = traj.shape[0], traj.device
        dtype = next(p for p in self.parameters() if p.requires_grad).dtype
        traj  = traj.to(device, dtype)

        q, enc, v_mean = self.get_eeg_enc(traj)   # q: (B,128), enc: (B,256,768), v_mean: (B,768)

        # ── InfoNCE alignment ──────────────────────────────────────────────────
        k = F.normalize(
            self.txt_proj(sbert_embs.to(device, dtype)), dim=-1)    # (B, 128)
        pos        = (q * k).sum(-1, keepdim=True) / 0.07
        negs       = q @ self.queue.T.detach() / 0.07
        loss_align = F.cross_entropy(
            torch.cat([pos, negs], dim=1),
            torch.zeros(B, dtype=torch.long, device=device))
        self._enqueue(k)

        # ── Auxiliary heads ───────────────────────────────────────────────────
        sent_logit = self.sentiment_head(q).squeeze(-1)
        len_pred   = self.length_head(q).squeeze(-1)
        loss_sent  = F.binary_cross_entropy_with_logits(
            sent_logit, sentiment_labels.to(device))
        loss_len   = F.mse_loss(len_pred, length_labels.to(device))

        # ── Diversity loss on raw VideoMAE output (NOT on q) ──────────────────
        # v_mean is the L2-space before any bottleneck projection.
        # Gradient from loss_div flows directly into VideoMAE layer weights.
        # This is the KEY FIX vs V900 where div_loss on q was ineffective.
        if B > 1:
            v_norm   = F.normalize(v_mean, dim=-1)               # (B, 768) L2-norm
            cos_mat  = v_norm @ v_norm.T                          # (B, B) cosine sims
            off_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            loss_div = cos_mat[off_mask].clamp(min=0).mean()     # penalize sim > 0
        else:
            loss_div = torch.zeros(1, device=device, dtype=dtype).squeeze()

        # ── T5 forward (only matters in Stage 2; Stage 1 has lm_weight=0) ────
        enc_hs   = enc                                            # (B, 256, 768) bfloat16
        enc_attn = torch.ones(B, EEG_SEQ_LEN, dtype=torch.long, device=device)
        labels   = target_ids.to(device).clone()
        labels[target_mask.to(device) == 0] = -100

        enc_out  = BaseModelOutput(last_hidden_state=enc_hs)
        out      = self.t5(encoder_outputs=enc_out, attention_mask=enc_attn, labels=labels)
        loss_lm  = out.loss

        # Return individual losses — training loop applies stage-dependent weights
        return loss_lm, loss_align, loss_div, loss_sent, loss_len

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_new_tokens=80, num_beams=5):
        from transformers.modeling_outputs import BaseModelOutput

        traj   = traj.to(next(self.parameters()).dtype)
        device = next(self.parameters()).device
        B      = traj.shape[0]

        _, enc, _ = self.get_eeg_enc(traj)                       # (B, 256, 768)
        enc_hs    = enc
        enc_attn  = torch.ones(B, EEG_SEQ_LEN, dtype=torch.long, device=device)

        results = []
        for b in range(B):
            eo   = BaseModelOutput(last_hidden_state=enc_hs[b:b+1])
            attn = enc_attn[b:b+1]
            out  = self.t5.generate(
                encoder_outputs=eo,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.20,
                length_penalty=0.70,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            results.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return results


# ── Warm-start ─────────────────────────────────────────────────────────────────

def load_warm_start(model):
    """
    Priority:
      1. V700 ep5 (best diversity: 30.2%) — fresh discriminative VideoMAE state
      2. V700 best other checkpoint
      3. V61 VideoMAE only
    We intentionally SKIP V800/V900 — they were already in collapsed states.
    V700 ep5 + Stage 1 alignment-only training = maximum chance to escape collapse.
    """
    import glob

    # Priority 1: V700 ep5 specifically
    v700_dir = Path("/v700_ckpt")
    v700_ep5 = sorted(glob.glob(str(v700_dir / "v700_ep5*.pt")))
    if v700_ep5:
        src = v700_ep5[0]
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state", ckpt)
        own  = model.state_dict()
        loaded, skipped = [], []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own, strict=False)
        print(f"Warm-start from V700 ep5: {len(loaded)} params, {len(skipped)} skipped. src={src}")
        return

    # Priority 2: Any V700 checkpoint
    v700_ckpts = sorted(glob.glob(str(v700_dir / "v700_ep*.pt")))
    if v700_ckpts:
        src = v700_ckpts[0]
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state", ckpt)
        own  = model.state_dict()
        loaded, skipped = [], []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own, strict=False)
        print(f"Warm-start from V700: {len(loaded)} params, {len(skipped)} skipped. src={src}")
        return

    # Fallback: V61 VideoMAE weights
    v61_dir = Path("/v61_ckpt")
    ckpts   = sorted(glob.glob(str(v61_dir / "*.pt")))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state", ckpt)
        own  = model.state_dict()
        loaded, skipped = [], []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own, strict=False)
        print(f"Warm-start from V61: {len(loaded)} params, {len(skipped)} skipped.")
        return

    print("No checkpoint found, training from scratch.")


def make_optimizer(model, steps_per_epoch, epochs):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    emb_p, early_p, mid_p, top_p, adapt_p = [], [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "video_enc.embeddings" in n:         emb_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(0, 4)):
            early_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(4, 8)):
            mid_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(8, 12)):
            top_p.append(p)
        else:
            adapt_p.append(p)

    param_groups = [
        {"params": emb_p,   "lr": 5e-7},
        {"params": early_p, "lr": 1e-6},
        {"params": mid_p,   "lr": 5e-6},
        {"params": top_p,   "lr": 1e-5},
        {"params": adapt_p, "lr": 2e-4},
    ]
    opt = AdamW(param_groups, weight_decay=1e-4)
    sched = OneCycleLR(opt,
        max_lr=[5e-7, 1e-6, 5e-6, 1e-5, 2e-4],
        total_steps=steps_per_epoch * epochs,
        pct_start=0.10, anneal_strategy="cos")
    return opt, sched


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_wer(pred, ref):
    import editdistance
    p_words = pred.split(); r_words = ref.split()
    if not r_words: return 1.0
    return editdistance.eval(p_words, r_words) / len(r_words)


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 1.0
    return editdistance.eval(pred, ref) / len(ref)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(model, val_loader, device, epoch, tokenizer):
    model.eval()
    wers, cers, all_preds, all_refs = [], [], [], []
    n_show = 5

    for batch in val_loader:
        eegs, texts, sberts, tgt_ids, tgt_mask, sents, lens = batch
        eegs = eegs.to(device)
        with torch.no_grad():
            preds = model.generate(eegs, tokenizer, max_new_tokens=80, num_beams=5)
        for pred, ref in zip(preds, texts):
            wers.append(compute_wer(pred.lower(), ref.lower()))
            cers.append(compute_cer(pred.lower(), ref.lower()))
            all_preds.append(pred); all_refs.append(ref)

    mean_wer = float(np.mean(wers))
    mean_cer = float(np.mean(cers))
    diversity = len(set(all_preds)) / max(1, len(all_preds))

    def cons_recall(preds, refs):
        hits = 0; total = 0
        for p, r in zip(preds, refs):
            rw = set(r.lower().split()); pw = set(p.lower().split())
            hits += len(rw & pw); total += max(1, len(rw))
        return hits / total
    cr = cons_recall(all_preds, all_refs)

    stage = "S1" if epoch <= STAGE2_START else "S2"
    print(f"\n[Epoch {epoch}/{stage}] WER={mean_wer:.3f}  CER={mean_cer:.3f}  "
          f"Diversity={diversity:.3f}  ConsRecall={cr:.3f}")
    print("── Sample outputs ──")
    for p, r in zip(all_preds[:n_show], all_refs[:n_show]):
        print(f"  REF : {r[:80]}")
        print(f"  PRED: {p[:80]}")
        print()

    model.train()
    return mean_wer, diversity, cr


# ── Training loop ──────────────────────────────────────────────────────────────

def train(model, train_ds, val_ds, tokenizer, device, epochs=50):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    opt, sched      = make_optimizer(model, steps_per_epoch, epochs)

    ckpt_dir       = Path("/persist"); ckpt_dir.mkdir(exist_ok=True)
    best_score     = -1.0
    best_wer       = 1e9
    low_div_streak = 0
    EARLY_STOP_DIV    = 0.10   # slightly more permissive in Stage 1 (no LM = lower diversity)
    EARLY_STOP_STREAK = 5      # allow more patience for two-stage approach

    model.train()
    for epoch in range(1, epochs + 1):
        # ── Stage-dependent loss weights ──────────────────────────────────────
        if epoch <= STAGE2_START:
            # Stage 1: No LM. Force VideoMAE to be discriminative.
            w_lm, w_align, w_div, w_sent, w_len = 0.0, 3.0, 2.0, 0.30, 0.10
            stage_label = "S1"
        else:
            # Stage 2: Gentle LM. Discriminative features established.
            w_lm, w_align, w_div, w_sent, w_len = 0.15, 1.5, 1.0, 0.20, 0.10
            stage_label = "S2"

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0; n_accum = 0

        for step, batch in enumerate(train_loader, 1):
            eegs, texts, sberts, tgt_ids, tgt_mask, sents, lens = batch
            eegs = eegs.to(device, torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lm, align, div_l, sent_l, len_l = model(
                    eegs, sberts, tgt_ids, tgt_mask, sents, lens)
                total = (w_lm * lm + w_align * align + w_div * div_l
                         + w_sent * sent_l + w_len * len_l)
                loss  = total / ACCUM_STEPS

            loss.backward()
            n_accum += 1

            if n_accum % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
                total_loss += total.item()

                if (n_accum // ACCUM_STEPS) % 10 == 0:
                    print(f"  ep={epoch}[{stage_label}] step={n_accum//ACCUM_STEPS} "
                          f"lm={lm.item():.3f} align={align.item():.3f} "
                          f"div_enc={div_l.item():.3f} sent={sent_l.item():.3f}")

        print(f"\nEpoch {epoch}[{stage_label}]: "
              f"avg_loss={total_loss / max(1, n_accum//ACCUM_STEPS):.3f}")

        wer, div, cr = validate(model, val_loader, device, epoch, tokenizer)
        score = div * cr

        if score > best_score or wer < best_wer:
            best_score = max(best_score, score)
            best_wer   = min(best_wer, wer)
            ckpt_path  = ckpt_dir / (
                f"v1000_ep{epoch}_{stage_label}_div{div:.3f}_cr{cr:.3f}_wer{wer:.3f}.pt")
            torch.save({
                "epoch": epoch, "stage": stage_label,
                "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
                "opt_state":   opt.state_dict(),
            }, str(ckpt_path))
            ckpt_vol.commit()
            print(f"  ✓ Best checkpoint: {ckpt_path.name}")

        if div < EARLY_STOP_DIV:
            low_div_streak += 1
            print(f"  ⚠ Low diversity ({div:.3f}) streak={low_div_streak}/{EARLY_STOP_STREAK}")
            if low_div_streak >= EARLY_STOP_STREAK:
                print(f"  ✗ Early stop: div < {EARLY_STOP_DIV} for {EARLY_STOP_STREAK} epochs")
                break
        else:
            low_div_streak = 0


# ── Modal entrypoint ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={
        "/persist":    ckpt_vol,
        "/v61_ckpt":   v61_vol,
        "/v600_ckpt":  v600_vol,
        "/v700_ckpt":  v700_vol,
        "/v800_ckpt":  v800_vol,
        "/v900_ckpt":  v900_vol,
        "/data":       data_vol,
    },
)
def run_training(epochs: int = 50):
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import T5Tokenizer

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    base_path = Path("/data")
    eeg_cache_exists = any(
        Path(p).exists()
        for p in ["/persist/v65_data_cache.pt", "/v600_ckpt/v65_data_cache.pt"])
    if not eeg_cache_exists:
        download_zuco(base_path, data_vol)
    else:
        print("EEG cache found — skipping ZuCo download.")
    cache = load_or_build_cache(base_path, data_vol)

    eegs  = cache["eegs"]
    texts = cache["texts"]
    N     = len(texts)
    print(f"Dataset: {N} samples")

    sbert_cache = None
    for sp in ["/persist/v100_sbert_embs.pt", "/v600_ckpt/v100_sbert_embs.pt"]:
        if Path(sp).exists():
            sbert_cache = Path(sp); break
    if sbert_cache:
        sbert_np = torch.load(sbert_cache, weights_only=False)
        print(f"SBERT cache hit at {sbert_cache}")
    else:
        sbert_cache = Path("/persist/v100_sbert_embs.pt")
        print("Computing SBERT embeddings …")
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        sbert_np    = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                         convert_to_numpy=True)
        torch.save(sbert_np, sbert_cache); ckpt_vol.commit()
    print(f"SBERT: {sbert_np.shape}")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    print("Tokenizing targets …")
    input_ids_list, attn_mask_list = [], []
    for text in texts:
        enc = tokenizer(text, max_length=MAX_TEXT_LEN, truncation=True,
                        padding=False, return_tensors='pt')
        input_ids_list.append(enc.input_ids[0])
        attn_mask_list.append(enc.attention_mask[0])

    print("Computing auxiliary labels …")
    sentiments, lengths = build_labels(texts, tokenizer)

    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    split = int(0.85 * N)
    train_idx = idx[:split].tolist()
    val_idx   = idx[split:].tolist()

    train_ds = EEGTextDataset(eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                              sentiments, lengths, train_idx, augment=True)
    val_ds   = EEGTextDataset(eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                              sentiments, lengths, val_idx, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print("Building model …")
    model = EEGV1000().to(device)
    load_warm_start(model)
    model = model.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.1f}M  Frozen: {frozen/1e6:.1f}M")

    print(f"Two-stage training: Stage 1 (ep 1-{STAGE2_START}) = align+div_enc only, "
          f"Stage 2 (ep {STAGE2_START+1}+) = +LM")

    train(model, train_ds, val_ds, tokenizer, device, epochs=epochs)
    print("Training complete.")


@app.local_entrypoint()
def main():
    run_training.remote(epochs=50)
