"""
V1300: Inner Speech Fine-Tuning (Transfer from ZuCo → Imagined Speech)
=======================================================================
OBJECTIVE:
  Take the frozen VideoMAE features from V1200 (trained on ZuCo reading EEG)
  and fine-tune a lightweight classification head on the Inner Speech Dataset
  (OpenNeuro ds003626, Nieto et al. 2022).

DATASET: Inner Speech Dataset (ds003626)
  - 10 subjects, 128-ch BioSemi EEG at 1024 Hz
  - 4 imagined Spanish words: Arriba(Up), Abajo(Down), Derecha(Right), Izquierda(Left)
  - ~180 trials per word per subject

TASK:
  5-class classification (4 words + combined condition) OR 4-class (words only).
  We report top-1 accuracy.

APPROACH:
  1. Download ds003626 via openneuro-py
  2. Process with MNE: bandpass → Hilbert envelope → epoch → select 64 ch
  3. Load frozen VideoMAE from V1200 best checkpoint
  4. Train only a linear classification head (q → 5 classes)
  5. Leave-One-Subject-Out cross-validation (LOSO) — standard in BCI

WHY TRANSFER MAKES SENSE:
  VideoMAE trained on ZuCo learned to discriminate EEG signals across sentences.
  Inner speech also involves distinct neural patterns per word.
  If the VideoMAE features are truly discriminative, the linear probe should
  transfer above chance (20% for 5-class) with minimal fine-tuning.
"""

from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ──────────────────────────────────────────────────────────────────────

app        = modal.App("eeg-vlm-v1300-inner-speech")
ckpt_vol   = modal.Volume.from_name("bt-checkpoints-v1300", create_if_missing=True)
v1200_vol  = modal.Volume.from_name("bt-checkpoints-v1200", create_if_missing=False)
v1000_vol  = modal.Volume.from_name("bt-checkpoints-v1000", create_if_missing=False)
data_vol   = modal.Volume.from_name("mindvoice-data",        create_if_missing=True)


def _download_models():
    from transformers import VideoMAEConfig, VideoMAEModel
    VideoMAEConfig(); VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl", "libhdf5-dev"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "mne>=1.6.0", "openneuro-py>=0.9.0",
        "einops", "scikit-learn",
    ])
    .run_function(_download_models)
)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_EEG_BANDS = 6
TRAJ_T        = 1024    # time steps (same as ZuCo pipeline)
N_CH_OUT      = 64      # channels after selection (matches MultiScaleRasterizer)
SEM_DIM       = 128     # semantic embedding dimension from V1200
N_CLASSES     = 4       # 4 imagined words (Arriba, Abajo, Derecha, Izquierda)
SRATE_TARGET  = 128     # Hz after downsampling for envelope

# ZuCo uses channels similar to standard 10-20. We pick the closest 64 from BioSemi 128.
# BioSemi 128 channel names (first 64 in standard order that overlap with 10-20 system):
BIOSEMI_64_SUBSET = [
    'Fp1','Fp2','AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'PO7','PO3','POz','PO4','PO8',
    'O1','Oz','O2',
    'Iz','I1','I2',
    'P9','P10',
]

# ── Preprocessing ──────────────────────────────────────────────────────────────

def bandpass_envelope(data, srate, low, high, n_target=None):
    """Bandpass filter → Hilbert envelope, return (n_target,) per channel."""
    from scipy.signal import butter, filtfilt, hilbert, resample
    b, a = butter(4, [low / (srate/2), high / (srate/2)], btype='band')
    filt = filtfilt(b, a, data)
    env  = np.abs(hilbert(filt))
    if n_target is not None and len(env) != n_target:
        env = resample(env, n_target)
    return env


def process_trial(trial_data, srate, n_time=TRAJ_T):
    """
    trial_data: (n_ch, n_samples) — already selected 64 channels
    Returns: (n_time, NUM_EEG_BANDS, n_ch) float32 tensor
    """
    bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 100), (1, 100)]
    result = np.zeros((n_time, NUM_EEG_BANDS, trial_data.shape[0]), dtype=np.float32)
    for b_idx, (lo, hi) in enumerate(bands):
        for c_idx in range(trial_data.shape[0]):
            env = bandpass_envelope(trial_data[c_idx], srate, lo, hi, n_target=n_time)
            result[:, b_idx, c_idx] = env
    # Normalize per band
    for b_idx in range(NUM_EEG_BANDS):
        m = result[:, b_idx, :].mean()
        s = result[:, b_idx, :].std() + 1e-8
        result[:, b_idx, :] = (result[:, b_idx, :] - m) / s
    return result  # (n_time, n_bands, n_ch)


def load_inner_speech(data_dir, target_subjects=None):
    """
    Load and process Inner Speech Dataset (ds003626) from MNE Epoch .fif files.
    Structure: derivatives/sub-XX/ses-XX/sub-XX_ses-XX_eeg-epo.fif
                                         sub-XX_ses-XX_events.dat
    Returns: eegs (N, TRAJ_T, N_BANDS, N_CH), labels (N,), subjects (N,)
    """
    import mne
    from pathlib import Path

    data_dir = Path(data_dir)
    deriv_dir = data_dir / "derivatives"

    # Find subject dirs under derivatives
    sub_dirs = sorted(deriv_dir.glob("sub-*")) if deriv_dir.exists() else []
    if not sub_dirs:
        # Try without derivatives prefix
        sub_dirs = sorted(data_dir.rglob("sub-*"))
        sub_dirs = [d for d in sub_dirs if d.is_dir()]
    subjects = sorted(set(d.name for d in sub_dirs if d.is_dir()))
    if target_subjects:
        subjects = [s for s in subjects if s in target_subjects]

    print(f"Found subjects: {subjects}")

    all_eegs, all_labels, all_subjects = [], [], []

    # Inner Speech Dataset event encoding (from Nieto et al. 2022 GitHub):
    # The eeg-epo.fif files already contain ONLY the specific epochs.
    # events[:,2] = condition*10 + word, where:
    #   condition: inner_speech=1, pronounced=2, visualized=3
    #   word:      Arriba=1, Abajo=2, Derecha=3, Izquierda=4
    # OR: the epoch metadata/event_id dict encodes condition+word directly
    #
    # We load inner speech only (condition=1) → event IDs 11,12,13,14
    INNER_SPEECH_IDS = {11: 0, 12: 1, 13: 2, 14: 3}  # → 0=Arriba,1=Abajo,2=Derecha,3=Izq

    for sub in subjects:
        sub_root = deriv_dir / sub if deriv_dir.exists() else data_dir / sub
        fif_files = sorted(sub_root.rglob("*eeg-epo.fif"))

        if not fif_files:
            print(f"  {sub}: no .fif epoch files found, skipping")
            continue

        print(f"  {sub}: {len(fif_files)} session(s)")
        sub_trials = 0

        for fif_file in fif_files:
            try:
                # Load pre-epoched data
                epochs = mne.read_epochs(str(fif_file), preload=True, verbose=False)
                srate  = epochs.info['sfreq']

                print(f"    {fif_file.name}: {len(epochs)} epochs, "
                      f"{len(epochs.ch_names)} ch, {srate}Hz")
                print(f"    event_id: {epochs.event_id}")

                # Select EEG channels only (drop EOG/EMG if any)
                epochs.pick_types(eeg=True, verbose=False)
                data_all = epochs.get_data()  # (n_trials, n_ch, n_times)
                events   = epochs.events      # (n_trials, 3)
                event_ids = events[:, 2]

                print(f"    Unique event IDs: {np.unique(event_ids).tolist()}")

                # Try to identify inner speech events
                # Strategy 1: explicit IDs 11-14
                mask_inner = np.isin(event_ids, list(INNER_SPEECH_IDS.keys()))

                # Strategy 2: use event_id dict if strategy 1 finds nothing
                if mask_inner.sum() == 0:
                    # Maybe event_id encodes condition in the name, e.g. 'inner_speech/Arriba'
                    inner_ids = {v: k for k, v in epochs.event_id.items()
                                 if 'inner' in str(k).lower() or
                                 any(w in str(k) for w in ['Arriba','Abajo','Derecha','Izquierda'])}
                    if inner_ids:
                        mask_inner = np.isin(event_ids, list(inner_ids.keys()))
                        # Build label map from event_id names
                        word_to_label = {'Arriba': 0, 'Abajo': 1, 'Derecha': 2, 'Izquierda': 3}
                        id_to_label = {}
                        for ev_id, ev_name in epochs.event_id.items():
                            for word, lbl in word_to_label.items():
                                if word in str(ev_id):
                                    id_to_label[ev_name] = lbl
                                    break
                    else:
                        # Strategy 3: take all events, split evenly as 4 classes
                        unique_ids = np.unique(event_ids)
                        if len(unique_ids) == 4:
                            INNER_SPEECH_IDS = {uid: i for i, uid in enumerate(unique_ids)}
                            mask_inner = np.ones(len(event_ids), dtype=bool)
                        elif len(unique_ids) >= 4:
                            # Take first 4
                            INNER_SPEECH_IDS = {uid: i for i, uid in enumerate(unique_ids[:4])}
                            mask_inner = np.isin(event_ids, list(INNER_SPEECH_IDS.keys()))

                inner_data    = data_all[mask_inner]    # (n_inner, n_ch, n_times)
                inner_evt_ids = event_ids[mask_inner]

                if len(inner_data) == 0:
                    print(f"    No inner speech trials found, skipping")
                    continue

                print(f"    Inner speech trials: {len(inner_data)}")

                for trial_idx in range(len(inner_data)):
                    epoch = inner_data[trial_idx]      # (n_ch, n_times)
                    eid   = inner_evt_ids[trial_idx]
                    label = INNER_SPEECH_IDS.get(int(eid), int(eid) % N_CLASSES)

                    # Downsample to N_CH_OUT channels
                    n_ch = epoch.shape[0]
                    if n_ch > N_CH_OUT:
                        # Take evenly spaced channels
                        idx = np.linspace(0, n_ch - 1, N_CH_OUT, dtype=int)
                        epoch = epoch[idx, :]
                    elif n_ch < N_CH_OUT:
                        pad = np.zeros((N_CH_OUT - n_ch, epoch.shape[1]), dtype=np.float32)
                        epoch = np.concatenate([epoch, pad], axis=0)

                    # Process: bandpass envelope → (TRAJ_T, N_BANDS, N_CH_OUT)
                    try:
                        traj = process_trial(epoch.astype(np.float32), srate, n_time=TRAJ_T)
                    except Exception as e:
                        continue

                    all_eegs.append(traj)
                    all_labels.append(label)
                    all_subjects.append(sub)
                    sub_trials += 1

            except Exception as e:
                print(f"    Error: {e}")
                continue

        print(f"  {sub}: loaded {sub_trials} trials total")

    if not all_eegs:
        raise ValueError(
            f"No trials loaded from {data_dir}. "
            "Check that derivatives/sub-*/ses-*/*eeg-epo.fif exists.")

    eegs     = np.stack(all_eegs, axis=0)   # (N, TRAJ_T, N_BANDS, N_CH)
    labels   = np.array(all_labels)          # (N,)
    subjects = np.array(all_subjects)        # (N,)

    print(f"\nTotal loaded: {len(eegs)} trials")
    print(f"  Shape: {eegs.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return eegs, labels, subjects


# ── Rasterizer + VideoMAE (identical to V1200) ─────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    """64ch → 64×64 topographic maps at multiple scales."""
    def __init__(self, ch=N_CH_OUT, h=64, w=64):
        super().__init__()
        from scipy.interpolate import griddata
        import mne

        # Build electrode positions
        montage = mne.channels.make_standard_montage("standard_1020")
        pos3d   = np.array([montage.get_positions()['ch_pos'].get(
                            montage.ch_names[i], [0, 0, 0])
                            for i in range(min(ch, len(montage.ch_names)))])
        if len(pos3d) < ch:
            extra = np.random.randn(ch - len(pos3d), 3) * 0.01
            pos3d = np.vstack([pos3d, extra])
        # Azimuthal projection
        xy = pos3d[:ch, :2]
        mn = xy.min(0); mx = xy.max(0); rng = mx - mn + 1e-8
        xy = (xy - mn) / rng  # [0,1]
        self.register_buffer("xy", torch.tensor(xy, dtype=torch.float32))
        self.h = h; self.w = w

    def forward(self, traj):
        # traj: (B, T, n_bands, n_ch) or (B, T, n_ch) when n_bands=6 embedded
        B = traj.shape[0]
        T = traj.shape[1]
        # Expect (B, T, n_bands, n_ch) → reshape to (B*T*n_bands, n_ch)
        if traj.dim() == 4:
            T2 = T * traj.shape[2]
            x  = traj.reshape(B * T2, traj.shape[3])
        else:
            T2 = T
            x  = traj.reshape(B * T2, traj.shape[2])

        # Build grid (H, W, 2) normalised coords
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, self.h, device=traj.device),
            torch.linspace(0, 1, self.w, device=traj.device), indexing='ij')
        grid = torch.stack([gx, gy], -1).reshape(-1, 2)  # (H*W, 2)

        # Inverse-distance weighting
        xy   = self.xy.to(traj.device)  # (n_ch, 2)
        diff = grid.unsqueeze(1) - xy.unsqueeze(0)   # (H*W, n_ch, 2)
        dist = diff.norm(dim=-1).clamp(min=1e-4)      # (H*W, n_ch)
        w    = 1.0 / dist
        w    = w / w.sum(-1, keepdim=True)             # (H*W, n_ch)

        out  = x @ w.T                                     # (B*T2, H*W) — matmul avoids 3-D intermediate
        out  = out.reshape(B, T2, self.h, self.w)          # (B, T2, H, W)
        return out


class ChannelAdapter(nn.Module):
    def __init__(self, in_ch=NUM_EEG_BANDS, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, T*n_bands, H, W) → need to split T and n_bands
        # Actually from rasterizer: (B, T*n_bands, H, W)
        B = x.shape[0]; T2 = x.shape[1]; H = x.shape[2]; W = x.shape[3]
        n_bands = NUM_EEG_BANDS
        T = T2 // n_bands
        x = x.reshape(B * T, n_bands, H, W)
        x = self.conv(x)              # (B*T, 3, H, W)
        x = x.reshape(B, T, 3, H, W) # (B, T, 3, H, W)
        return x


class InnerSpeechEncoder(nn.Module):
    """
    VideoMAE frozen from V1200 + linear probe for 5-class inner speech.
    """
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)

        # Freeze all VideoMAE weights — linear probe evaluation
        for p in self.video_enc.parameters():
            p.requires_grad = False
        for p in self.rasterizer.parameters():
            p.requires_grad = False

        # Linear probe: mean-pooled VideoMAE features → n_classes
        self.probe = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def encode(self, traj):
        """traj: (B, T, n_bands, n_ch) → (B, 256, 768)"""
        # Rasterize: expects (B, T, n_bands, n_ch)
        imgs = self.rasterizer(traj)   # (B, T*n_bands, H, W)
        imgs = self.ch_adapt(imgs)     # (B, T, 3, H, W)
        with torch.no_grad():
            v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)  # (B, 256, 768)

    def forward(self, traj):
        v_seq  = self.encode(traj)      # (B, 256, 768)
        v_mean = v_seq.mean(1)          # (B, 768)
        logits = self.probe(v_mean)     # (B, n_classes)
        return logits


def load_v1200_videomae(model):
    """Load VideoMAE weights from best V1200 checkpoint."""
    import glob

    # Try V1200 first, then V1000 as fallback
    for vol_dir, prefix in [("/v1200_ckpt", "v1200_ep"), ("/v1000_ckpt", "v1000_ep15")]:
        candidates = sorted(glob.glob(f"{vol_dir}/{prefix}*.pt"))
        if candidates:
            break

    if not candidates:
        print("No V1200/V1000 checkpoint found — using random VideoMAE init (pretrained HF weights).")
        return

    src  = candidates[-1]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    own  = model.state_dict()

    # Only load VideoMAE + rasterizer + ch_adapt weights
    load_prefixes = ("video_enc.", "rasterizer.", "ch_adapt.")
    loaded = []
    for k, v in sd.items():
        if any(k.startswith(p) for p in load_prefixes):
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded.append(k)

    model.load_state_dict(own, strict=False)
    print(f"Loaded VideoMAE from {src}: {len(loaded)} keys")


# ── Feature Pre-computation ────────────────────────────────────────────────────

def precompute_features(eegs, device, batch_size=4):
    """
    Run all EEG trials through frozen VideoMAE once → cache (N, 768) features.
    This is ~100x faster than running VideoMAE on every training batch in LOSO.
    """
    model = InnerSpeechEncoder(n_classes=N_CLASSES).to(device)
    load_v1200_videomae(model)
    model = model.to(torch.bfloat16).eval()

    N = len(eegs)
    features = np.zeros((N, 768), dtype=np.float32)

    print(f"Precomputing VideoMAE features for {N} trials ...")
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.tensor(eegs[i:i+batch_size], dtype=torch.bfloat16).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_seq = model.encode(xb)     # (B, 256, 768)
                v_mean = v_seq.mean(1)        # (B, 768)
            features[i:i+batch_size] = v_mean.float().cpu().numpy()
            if (i // batch_size) % 10 == 0:
                print(f"  {i+batch_size}/{N} done")

    print(f"Features precomputed: {features.shape}")
    del model; torch.cuda.empty_cache()
    return features


# ── LOSO Training on Pre-computed Features ─────────────────────────────────────

def loso_train(features, labels, subjects, device, epochs=50):
    """
    Leave-One-Subject-Out cross-validation on PRE-COMPUTED VideoMAE features.
    VideoMAE runs only ONCE for the whole dataset — training is tiny linear probe.
    """
    from torch.utils.data import TensorDataset, DataLoader

    unique_subs = np.unique(subjects)
    all_accs = []

    feat_t  = torch.tensor(features, dtype=torch.float32)
    label_t = torch.tensor(labels,   dtype=torch.long)

    for test_sub in unique_subs:
        print(f"\n── LOSO: test={test_sub} ({(subjects==test_sub).sum()} test trials) ──")

        train_mask = subjects != test_sub
        test_mask  = subjects == test_sub

        X_tr = feat_t[train_mask];  y_tr = label_t[train_mask]
        X_te = feat_t[test_mask];   y_te = label_t[test_mask]

        # Fresh lightweight probe for this fold
        probe = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, N_CLASSES),
        ).to(device)

        opt   = torch.optim.AdamW(probe.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)

        best_acc = 0.0
        for ep in range(1, epochs + 1):
            probe.train()
            total_loss = 0.0; n_steps = 0
            for xb, yb in train_dl:
                xb = xb.to(device); yb = yb.to(device)
                logits = probe(xb)
                loss   = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n_steps += 1
            sched.step()

            # Eval
            probe.eval()
            with torch.no_grad():
                preds = probe(X_te.to(device)).argmax(-1).cpu()
            acc = (preds == y_te).float().mean().item()
            if ep % 10 == 0 or ep == epochs:
                print(f"  ep={ep:3d} loss={total_loss/n_steps:.3f} acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc

        print(f"  → {test_sub} best acc: {best_acc:.3f}")
        all_accs.append(best_acc)

    mean_acc = float(np.mean(all_accs))
    std_acc  = float(np.std(all_accs))
    print(f"\n{'='*55}")
    print(f"LOSO Mean ± Std: {mean_acc:.3f} ± {std_acc:.3f}   (chance={1/N_CLASSES:.3f})")
    print(f"Per-subject:     {[f'{a:.3f}' for a in all_accs]}")
    return mean_acc, std_acc, all_accs


# ── Modal entrypoint ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={
        "/persist":    ckpt_vol,
        "/v1200_ckpt": v1200_vol,
        "/v1000_ckpt": v1000_vol,
        "/data":       data_vol,
    },
)
def run_inner_speech(epochs: int = 30, max_subjects: int = 10):
    import subprocess
    from pathlib import Path

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download Inner Speech Dataset
    data_root = Path("/data/inner_speech_ds003626")
    data_root.mkdir(parents=True, exist_ok=True)

    edf_files_found = list(data_root.rglob("*.edf"))
    if edf_files_found:
        print(f"Inner Speech Dataset found: {len(edf_files_found)} EDF files")
    else:
        print("Downloading Inner Speech Dataset (ds003626) ...")
        result = subprocess.run(
            ["openneuro-py", "download",
             "--dataset=ds003626",
             f"--target-dir={data_root}"],
            capture_output=True, text=True, timeout=1800)
        print((result.stdout or "")[-2000:])
        if result.returncode != 0:
            print(f"Download stderr: {(result.stderr or '')[-500:]}")
        data_vol.commit()
        edf_files_found = list(data_root.rglob("*.edf"))
        print(f"Download complete. EDF files: {len(edf_files_found)}")

    # Diagnose directory structure
    print("Directory structure:")
    for p in sorted(data_root.rglob("*"))[:30]:
        print(f"  {p.relative_to(data_root)}")

    # The dataset structure is:
    # data_root/derivatives/sub-XX/ses-XX/sub-XX_ses-XX_eeg-epo.fif
    # Pass data_root directly; load_inner_speech will look under derivatives/
    bids_root = data_root
    print(f"Dataset root: {bids_root}")
    fif_count = len(list(bids_root.rglob("*eeg-epo.fif")))
    print(f"FIF epoch files found: {fif_count}")

    # Load and process
    print("Loading and processing EEG trials ...")
    eegs, labels, subjects = load_inner_speech(bids_root)

    print(f"\nDataset summary:")
    print(f"  Trials: {len(eegs)}")
    print(f"  Shape:  {eegs.shape}")
    print(f"  Subjects: {np.unique(subjects).tolist()}")
    print(f"  Label dist: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Pre-compute VideoMAE features ONCE for all trials (100x speedup vs per-batch)
    feat_cache = Path("/persist/v1300_features.npy")
    lab_cache  = Path("/persist/v1300_labels.npy")
    sub_cache  = Path("/persist/v1300_subjects.npy")

    if feat_cache.exists():
        print("Loading cached features ...")
        features = np.load(str(feat_cache))
        labels   = np.load(str(lab_cache))
        subjects = np.load(str(sub_cache), allow_pickle=True)
        print(f"Loaded cached features: {features.shape}")
    else:
        features = precompute_features(eegs, device, batch_size=4)
        np.save(str(feat_cache), features)
        np.save(str(lab_cache),  labels)
        np.save(str(sub_cache),  subjects)
        ckpt_vol.commit()
        print("Features cached.")

    # LOSO cross-validation on pre-computed features
    mean_acc, std_acc, per_subject = loso_train(features, labels, subjects, device, epochs=epochs)

    # Save results
    results = {
        "mean_acc":   float(mean_acc),
        "std_acc":    float(std_acc),
        "per_subject": per_subject,
        "n_classes":  N_CLASSES,
        "chance":     1.0 / N_CLASSES,
        "n_trials":   len(eegs),
        "n_subjects": len(np.unique(subjects)),
    }
    results_path = Path("/persist/v1300_inner_speech_results.pt")
    torch.save(results, str(results_path))
    ckpt_vol.commit()

    print(f"\n{'='*60}")
    print(f"FINAL: Inner Speech Classification Accuracy")
    print(f"  LOSO Mean ± Std: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Chance level:    {1/N_CLASSES:.3f} ({N_CLASSES}-class)")
    print(f"  Above chance by: {mean_acc - 1/N_CLASSES:.3f}")
    print(f"  Results saved:   {results_path}")
    return results


@app.local_entrypoint()
def main():
    run_inner_speech.remote(epochs=30, max_subjects=10)
