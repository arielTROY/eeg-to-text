"""
V1400: Self-Supervised EEG Foundation Encoder (SimCLR on PhysioNet EEGBCI)
===========================================================================
DATA: PhysioNet EEGBCI (EEG Motor Movement/Imagery Dataset)
  - 109 subjects, 64 EEG channels, 160 Hz
  - 14 runs per subject (rest + 5 motor imagery tasks × 3 runs)
  - ~50 hours total EEG, freely downloadable via MNE

METHOD: SimCLR with masked VideoMAE encoder
  - FFT-based spectral decomposition: 6 frequency bands (fast, vectorized)
  - Preprocess → same (1024, 384) format as ZuCo pipeline
  - Two augmented views per window → NT-Xent contrastive loss
  - 90% tube masking → 410/4096 patches seen (memory-efficient)
  - ~42,000 windows total (30× ZuCo training size)

WHY THIS HELPS:
  The V1200 pipeline starts VideoMAE from random init or V1000 (15 epochs on 1363
  pairs). V1400 gives the encoder 42,000× 10 = 420,000 EEG views before any text
  alignment. The encoder learns universal EEG spatiotemporal structure
  (frequency band dynamics, spatial smoothness, temporal oscillations) before
  being asked to also align with sentence meanings.

  Analogy: wav2vec2 pre-trained on 960h unlabeled speech → fine-tune on 10h
  labeled speech. We pre-train on 50h unlabeled EEG → fine-tune on 3h labeled EEG.

EXPECTED OUTCOME:
  V1500 (ZuCo fine-tune from V1400) should reach Content Recall 20-30%
  (vs 11.2% current best in V1200) with same training budget.
"""

from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ──────────────────────────────────────────────────────────────────────

app      = modal.App("eeg-vlm-v1400-foundation")
ckpt_vol = modal.Volume.from_name("bt-checkpoints-v1400", create_if_missing=True)
data_vol = modal.Volume.from_name("eeg-foundation-data",  create_if_missing=True)


def _download_models():
    from transformers import VideoMAEConfig, VideoMAEModel
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "mne>=1.6.0",
    ])
    .run_function(_download_models)
)

# ── Constants (match V1200 pipeline exactly) ───────────────────────────────────

NUM_EEG_BANDS = 6
N_ELEC        = 64
TRAJ_T        = 1024     # time steps
WINDOW_SEC    = 8.0      # seconds per window
STRIDE_SEC    = 4.0      # 50% overlap
SRATE_EEGBCI  = 160      # Hz

# SimCLR hyper-parameters
BATCH_SIZE    = 32
LR            = 3e-4
TEMPERATURE   = 0.07
MASK_RATIO    = 0.90     # 90% tube masking (only 10% patches visible)

# ── Fast EEG Preprocessing (FFT-based, vectorized across all channels) ─────────

def fft_bandpass_segment(data_ch_time, srate, n_target=TRAJ_T):
    """
    data_ch_time: (n_ch, n_samples) raw EEG
    Returns:      (n_target, NUM_EEG_BANDS * n_ch) — matches V1200 format exactly
    """
    from scipy.fft import rfft, irfft
    from scipy.signal import resample

    n_ch, n_samp = data_ch_time.shape
    freqs = np.fft.rfftfreq(n_samp, 1.0 / srate)

    # Single FFT across all channels
    X = rfft(data_ch_time.astype(np.float64), axis=1)  # (n_ch, n_fft)

    bands = [
        (0.5, 40.),   # broadband
        (8.,  13.),   # alpha
        (13., 30.),   # beta
        (30., 45.),   # gamma
        (1.,   4.),   # delta
        (30., 45.),   # high-gamma (will apply diff below)
    ]

    result_bands = []
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi)
        Xf   = X.copy()
        Xf[:, ~mask] = 0.0
        filtered = irfft(Xf, n=n_samp, axis=1).real  # (n_ch, n_samp)

        if i == 5:   # high-gamma: temporal derivative suppresses DC drift
            filtered = np.diff(filtered, axis=1, prepend=filtered[:, :1])

        # Resample to TRAJ_T if needed
        if n_samp != n_target:
            filtered = resample(filtered, n_target, axis=1)

        result_bands.append(filtered.astype(np.float32))  # (n_ch, n_target)

    # Stack → (6, n_ch, n_target) → transpose → (n_target, 6, n_ch) → (n_target, 384)
    stacked = np.stack(result_bands, axis=0)              # (6, n_ch, n_target)
    result  = stacked.transpose(2, 0, 1).reshape(n_target, -1)  # (n_target, 384)

    # Z-score per feature (matches V1200 normalize_eeg)
    mu  = result.mean(0, keepdims=True)
    sig = result.std(0,  keepdims=True) + 1e-6
    return ((result - mu) / sig).astype(np.float32)


def segment_raw(raw_data, srate, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    """
    raw_data: (n_ch, n_samples)
    Returns list of (n_target, 384) windows.
    """
    n_ch, n_samp = raw_data.shape
    window = int(window_sec * srate)
    stride = int(stride_sec * srate)
    n_ch_use = min(n_ch, N_ELEC)

    windows = []
    for start in range(0, n_samp - window + 1, stride):
        seg = raw_data[:n_ch_use, start:start + window]
        # Pad channels if fewer than N_ELEC
        if n_ch_use < N_ELEC:
            pad = np.zeros((N_ELEC - n_ch_use, window), dtype=np.float32)
            seg = np.vstack([seg, pad])
        try:
            traj = fft_bandpass_segment(seg, srate)
            windows.append(traj)
        except Exception:
            continue
    return windows


def download_and_preprocess_eegbci(data_dir, max_subjects=109):
    """
    Download EEGBCI via MNE and preprocess to (N, 1024, 384).
    Cached to {data_dir}/v1400_windows.npy — reused on re-runs.
    """
    import mne
    from mne.datasets import eegbci

    mne.set_config('MNE_DATA', str(data_dir))
    mne.set_log_level('ERROR')

    cache_path = Path(data_dir) / "v1400_windows.npy"
    if cache_path.exists():
        arr = np.load(str(cache_path))
        print(f"Loaded cached EEGBCI windows: {arr.shape}")
        return arr

    all_runs = list(range(1, 15))
    all_windows = []

    for sub in range(1, max_subjects + 1):
        sub_count = 0
        try:
            raw_fnames = eegbci.load_data(
                sub, all_runs, path=str(data_dir),
                update_path=False, verbose=False)

            for fname in raw_fnames:
                try:
                    raw = mne.io.read_raw_edf(
                        fname, preload=True, stim_channel='auto', verbose=False)
                    eegbci.standardize(raw)
                    raw.pick_types(eeg=True, verbose=False)
                    raw_data = raw.get_data().astype(np.float32)
                    srate    = int(raw.info['sfreq'])

                    wins = segment_raw(raw_data, srate)
                    all_windows.extend(wins)
                    sub_count += len(wins)
                except Exception:
                    continue

        except Exception as e:
            print(f"  sub-{sub:02d}: download failed ({e})")
            continue

        if sub % 10 == 0 or sub <= 5:
            print(f"  sub-{sub:02d}/{max_subjects}: +{sub_count} windows "
                  f"| total {len(all_windows)}")

    arr = np.stack(all_windows, axis=0).astype(np.float32)  # (N, 1024, 384)
    np.save(str(cache_path), arr)
    print(f"Saved {arr.shape} → {cache_path}")
    return arr


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment(eeg: np.ndarray) -> np.ndarray:
    """
    eeg: (T, C) — two independent calls produce two augmented views.
    Matches V1200 augmentations + adds frequency-band masking.
    """
    T, C = eeg.shape
    eeg  = eeg.copy()

    # Time shift (±6.25% of T)
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift]  = 0
        else:         eeg[shift:]  = 0

    # Amplitude scale
    eeg *= np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)

    # Gaussian noise
    eeg += (np.random.randn(T, C) * 0.04).astype(np.float32)

    # Channel dropout (10%)
    n_drop = max(1, C // 10)
    drop   = np.random.choice(C, n_drop, replace=False)
    eeg[:, drop] = 0

    # Temporal masking (3 random spans)
    for _ in range(3):
        span  = np.random.randint(16, 80)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0

    # Frequency-band masking (zero one of the 6 bands)
    band_ch = C // NUM_EEG_BANDS    # 64 channels per band
    bidx    = np.random.randint(0, NUM_EEG_BANDS)
    eeg[:, bidx * band_ch:(bidx + 1) * band_ch] = 0

    return eeg


class EEGContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = windows  # (N, 1024, 384) float32

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        w  = self.windows[i]
        v1 = torch.from_numpy(augment(w))
        v2 = torch.from_numpy(augment(w))
        return v1, v2


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    """Circular IDW rasterizer — identical to V1200."""
    def __init__(self, size=64, n_electrodes=N_ELEC, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands; self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        w    = 1.0 / (dist + 1e-4) ** 2.0
        r    = torch.sqrt(px ** 2 + py ** 2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)   # (H*W, n_elec)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B * T, self.n_bands, self.n_electrodes)
        return (xs @ self.W.T.to(x.device, x.dtype)).reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    """6 frequency bands → 3 RGB channels — identical to V1200."""
    def __init__(self, in_ch=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B * T, C, H, W)).view(B, T, 3, H, W)


class EEGFoundationSimCLR(nn.Module):
    """
    SimCLR-style EEG foundation encoder.

    Architecture:
      - Rasterizer + ChannelAdapter: same as V1200
      - VideoMAE-base encoder (init from HuggingFace weights)
      - 90% tube masking → only 410/4096 patches encoded
      - Projection head: 768 → 512 GELU → 128 (L2 normalized)
      - NT-Xent contrastive loss on projected representations

    After pre-training, only video_enc / rasterizer / ch_adapt weights are
    transferred to V1500 (projector is discarded).
    """

    def __init__(self, mask_ratio=MASK_RATIO):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=TRAJ_T, tubelet_size=4, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)
        self.mask_ratio = mask_ratio

        # Total patches: (1024/4) × (64/16) × (64/16) = 256 × 4 × 4 = 4096
        T_p = TRAJ_T // 4; H_p = 64 // 16; W_p = 64 // 16
        self.n_patches = T_p * H_p * W_p  # 4096

        # Projection head (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 128),
        )

    def encode(self, x):
        """x: (B, T, 384) → (B, 768) mean-pooled masked encoder output."""
        B = x.shape[0]
        imgs = self.rasterizer(x)    # (B, T, 6, 64, 64)
        imgs = self.ch_adapt(imgs)   # (B, T, 3, 64, 64)

        # Random tube mask — different per item in batch
        n_mask     = int(self.n_patches * self.mask_ratio)
        noise      = torch.rand(B, self.n_patches, device=x.device)
        ids        = noise.argsort(dim=1)
        bool_mask  = torch.zeros(B, self.n_patches, dtype=torch.bool, device=x.device)
        bool_mask.scatter_(1, ids[:, :n_mask], True)  # True = masked (not visible)

        enc_out  = self.video_enc(pixel_values=imgs, bool_masked_pos=bool_mask)
        features = enc_out.last_hidden_state   # (B, n_visible, 768)
        return features.mean(1)               # (B, 768)

    def forward(self, x1, x2):
        """
        x1, x2: two augmented views of same EEG segments, shape (B, T, 384).
        Returns NT-Xent loss.
        """
        h1 = self.encode(x1); h2 = self.encode(x2)
        z1 = F.normalize(self.projector(h1), dim=-1)  # (B, 128)
        z2 = F.normalize(self.projector(h2), dim=-1)  # (B, 128)

        B   = z1.shape[0]
        # Build 2B × 2B similarity matrix
        z   = torch.cat([z1, z2], dim=0)              # (2B, 128)
        sim = z @ z.T / TEMPERATURE                   # (2B, 2B)

        # Mask self-similarities on diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=x1.device)
        sim  = sim.masked_fill(mask, -1e9)

        # Labels: z1[i] should match z2[i] (offset by B) and vice versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=x1.device),
            torch.arange(0, B,     device=x1.device),
        ])
        loss = F.cross_entropy(sim, labels)
        return loss


# ── Checkpoint loading from HuggingFace VideoMAE ──────────────────────────────

def load_hf_videomae(model):
    """Initialize VideoMAE from pretrained HuggingFace weights (good start)."""
    from transformers import VideoMAEModel
    hf = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    src = hf.state_dict()
    own = model.video_enc.state_dict()
    loaded = []
    for k in src:
        if k in own and own[k].shape == src[k].shape:
            own[k] = src[k]; loaded.append(k)
    model.video_enc.load_state_dict(own, strict=False)
    print(f"Loaded HuggingFace VideoMAE-base: {len(loaded)}/{len(src)} keys")
    del hf


# ── Training ──────────────────────────────────────────────────────────────────

def train_simclr(model, windows, device, epochs=10):
    from torch.utils.data import DataLoader

    ds = EEGContrastiveDataset(windows)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=epochs * len(dl),
        pct_start=0.05, anneal_strategy='cos')

    ckpt_dir = Path("/persist"); ckpt_dir.mkdir(exist_ok=True)

    print(f"\nSimCLR training: {len(ds)} windows, {len(dl)} steps/epoch, {epochs} epochs")
    print(f"Batch={BATCH_SIZE}, Temp={TEMPERATURE}, MaskRatio={MASK_RATIO}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0; n_steps = 0

        for step, (v1, v2) in enumerate(dl, 1):
            v1 = v1.to(device, torch.bfloat16)
            v2 = v2.to(device, torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(v1, v2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            total_loss += loss.item(); n_steps += 1

            if step % 100 == 0:
                print(f"  ep={epoch} step={step}/{len(dl)} "
                      f"loss={total_loss/n_steps:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

        avg = total_loss / max(1, n_steps)
        print(f"\nEpoch {epoch}: avg_loss={avg:.4f}")

        ckpt_path = ckpt_dir / f"v1400_ep{epoch:02d}_loss{avg:.4f}.pt"
        torch.save({
            "epoch": epoch, "loss": avg,
            "model_state": model.state_dict(),
        }, str(ckpt_path))
        ckpt_vol.commit()
        print(f"  ✓ Saved {ckpt_path.name}")

    print("\nV1400 pre-training complete!")
    print("Encoder weights are ready for V1500 fine-tuning on ZuCo.")


# ── Modal entrypoint ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={
        "/persist": ckpt_vol,
        "/data":    data_vol,
    },
)
def run_v1400(epochs: int = 10, max_subjects: int = 109):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Phase 1: Data ──────────────────────────────────────────────────────────
    print("\n=== Phase 1: EEGBCI download + FFT preprocessing ===")
    data_dir = Path("/data/mne_eegbci"); data_dir.mkdir(exist_ok=True)
    windows  = download_and_preprocess_eegbci(str(data_dir), max_subjects=max_subjects)
    data_vol.commit()
    print(f"Dataset: {windows.shape[0]} windows × {windows.shape[1:]} "
          f"= {windows.nbytes / 1e9:.1f} GB")

    # ── Phase 2: Build model ───────────────────────────────────────────────────
    print("\n=== Phase 2: Build EEGFoundationSimCLR ===")
    model = EEGFoundationSimCLR(mask_ratio=MASK_RATIO).to(device)
    load_hf_videomae(model)
    model = model.to(torch.bfloat16)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_total/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    # ── Phase 3: SimCLR training ───────────────────────────────────────────────
    print("\n=== Phase 3: SimCLR pre-training ===")
    train_simclr(model, windows, device, epochs=epochs)


@app.local_entrypoint()
def main():
    run_v1400.remote(epochs=10, max_subjects=109)
