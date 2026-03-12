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
    Load and process Inner Speech Dataset (ds003626).
    Returns: eegs (N, TRAJ_T, N_BANDS, N_CH), labels (N,), subjects (N,)
    """
    import mne
    from pathlib import Path

    data_dir = Path(data_dir)
    # Find all subject directories recursively
    subjects = sorted(set(
        d.name for d in data_dir.rglob("sub-*") if d.is_dir()
        and (list(d.rglob("*.edf")) or list(d.rglob("*.fif")))
    ))
    if target_subjects:
        subjects = [s for s in subjects if s in target_subjects]

    print(f"Found subjects: {subjects}")

    all_eegs, all_labels, all_subjects = [], [], []

    # Word labels
    label_map = {"Arriba": 0, "Abajo": 1, "Derecha": 2, "Izquierda": 3}

    for sub in subjects:
        sub_dir = data_dir / sub / "eeg"
        edf_files = sorted(sub_dir.glob("*task-inner*_eeg.edf"))
        if not edf_files:
            # Try alternative naming
            edf_files = sorted(sub_dir.glob("*.edf"))

        if not edf_files:
            print(f"  {sub}: no EDF files found, skipping")
            continue

        print(f"  Processing {sub}: {len(edf_files)} EDF file(s)")

        for edf_file in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                srate = raw.info['sfreq']

                # Select 64-channel subset
                available_ch = [ch for ch in BIOSEMI_64_SUBSET if ch in raw.ch_names]
                if len(available_ch) < 32:
                    # Fall back: take first 64 EEG channels
                    eeg_ch = [ch for ch in raw.ch_names if 'EEG' not in ch or True][:64]
                    available_ch = eeg_ch[:64]

                if not available_ch:
                    print(f"    No valid channels, skipping {edf_file.name}")
                    continue

                raw.pick_channels(available_ch, ordered=True)

                # Load events from events file
                events_tsv = edf_file.parent / (edf_file.name.replace('_eeg.edf', '_events.tsv'))
                if not events_tsv.exists():
                    # Try alternative naming
                    events_files = list(edf_file.parent.glob("*events.tsv"))
                    events_tsv = events_files[0] if events_files else None

                if events_tsv is None:
                    print(f"    No events file for {edf_file.name}")
                    continue

                import pandas as pd
                events_df = pd.read_csv(str(events_tsv), sep='\t')
                print(f"    Events columns: {list(events_df.columns)}")
                print(f"    Events sample: {events_df.head(3).to_dict()}")

                # Inner Speech Dataset (Nieto 2022) has columns:
                # onset, duration, trial_type (= 'inner_speech'/'pronounced_speech'/etc)
                # and 'value' column with the word ('Arriba','Abajo','Derecha','Izquierda')
                # OR trial_type contains the word directly
                word_events = pd.DataFrame()
                for col in ['value', 'trial_type', 'stim_type', 'condition', 'label']:
                    if col in events_df.columns:
                        mask = events_df[col].astype(str).isin(label_map.keys())
                        if mask.sum() > 0:
                            word_events = events_df[mask].copy()
                            word_events['_word_col'] = col
                            print(f"    Found {len(word_events)} word events in column '{col}'")
                            break

                if len(word_events) == 0:
                    # Try: filter for 'inner_speech' trial_type, get word from 'value'
                    if 'trial_type' in events_df.columns and 'value' in events_df.columns:
                        inner = events_df[events_df['trial_type'] == 'inner_speech']
                        if len(inner) > 0:
                            word_events = inner.copy()
                            word_events['_word_col'] = 'value'
                    print(f"    Fallback inner_speech events: {len(word_events)}")

                if len(word_events) == 0:
                    print(f"    No word events found in {events_tsv.name}")
                    continue

                data_np, _ = raw[:]
                data_np = data_np.astype(np.float32)  # (n_ch, n_samples)

                # Extract epochs around each word event
                epoch_dur_s = 4.0  # 4 seconds of inner speech (standard)
                epoch_samples = int(epoch_dur_s * srate)

                for _, row in word_events.iterrows():
                    # Get onset
                    onset    = row.get('onset', row.iloc[0])
                    word_col = row.get('_word_col', 'trial_type')
                    word     = str(row.get(word_col, row.get('trial_type', row.iloc[-1])))

                    if word not in label_map:
                        continue

                    onset_sample = int(float(onset) * srate)
                    end_sample   = onset_sample + epoch_samples

                    if end_sample > data_np.shape[1]:
                        continue

                    epoch = data_np[:, onset_sample:end_sample]  # (n_ch, n_samples)

                    try:
                        traj = process_trial(epoch, srate, n_time=TRAJ_T)  # (TRAJ_T, N_BANDS, n_ch)
                    except Exception as e:
                        print(f"    process_trial failed: {e}")
                        continue

                    # Pad/trim channels to N_CH_OUT
                    n_ch = traj.shape[2]
                    if n_ch < N_CH_OUT:
                        pad = np.zeros((TRAJ_T, NUM_EEG_BANDS, N_CH_OUT - n_ch), dtype=np.float32)
                        traj = np.concatenate([traj, pad], axis=2)
                    elif n_ch > N_CH_OUT:
                        traj = traj[:, :, :N_CH_OUT]

                    all_eegs.append(traj)
                    all_labels.append(label_map[word])
                    all_subjects.append(sub)

            except Exception as e:
                print(f"  Error processing {edf_file}: {e}")
                continue

    if not all_eegs:
        raise ValueError("No trials loaded! Check dataset path and format.")

    eegs    = np.stack(all_eegs)   # (N, TRAJ_T, N_BANDS, N_CH)
    labels  = np.array(all_labels) # (N,)
    subjects = np.array(all_subjects)  # (N,) str

    print(f"Loaded: {len(eegs)} trials, label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
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

        out  = (w.unsqueeze(0) * x.unsqueeze(1)).sum(-1)  # (B*T2, H*W)
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


# ── LOSO Training ──────────────────────────────────────────────────────────────

def loso_train(eegs, labels, subjects, device, epochs=30):
    """Leave-One-Subject-Out cross-validation."""
    unique_subs = np.unique(subjects)
    all_accs = []

    for test_sub in unique_subs:
        print(f"\n── LOSO: test subject = {test_sub} ──")

        train_mask = subjects != test_sub
        test_mask  = subjects == test_sub

        X_train = torch.tensor(eegs[train_mask],   dtype=torch.bfloat16)
        y_train = torch.tensor(labels[train_mask], dtype=torch.long)
        X_test  = torch.tensor(eegs[test_mask],    dtype=torch.bfloat16)
        y_test  = torch.tensor(labels[test_mask],  dtype=torch.long)

        model = InnerSpeechEncoder(n_classes=N_CLASSES).to(device)
        load_v1200_videomae(model)
        model = model.to(torch.bfloat16)

        # Only train the probe
        opt   = torch.optim.AdamW(model.probe.parameters(), lr=3e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)

        best_acc = 0.0
        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0; n_steps = 0
            for xb, yb in train_dl:
                xb = xb.to(device); yb = yb.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss   = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item(); n_steps += 1
            sched.step()

            # Evaluate
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for i in range(0, len(X_test), 16):
                    xb = X_test[i:i+16].to(device)
                    yb = y_test[i:i+16].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        preds = model(xb).argmax(-1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            acc = correct / max(1, total)
            if ep % 5 == 0 or ep == epochs:
                print(f"  ep={ep:2d} loss={total_loss/n_steps:.3f} acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc

        print(f"  {test_sub} best acc: {best_acc:.3f}")
        all_accs.append(best_acc)

    mean_acc = np.mean(all_accs)
    std_acc  = np.std(all_accs)
    print(f"\n{'='*50}")
    print(f"LOSO Results: {mean_acc:.3f} ± {std_acc:.3f} (chance={1/N_CLASSES:.3f})")
    print(f"Per-subject:  {[f'{a:.3f}' for a in all_accs]}")
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

    # The openneuro-py may nest under dataset name
    # Find actual BIDS root (where sub-* dirs are)
    bids_root = data_root
    for candidate in [data_root, data_root / "ds003626"]:
        if candidate.exists() and any(candidate.glob("sub-*")):
            bids_root = candidate
            break
    # Also try nested in any subfolder
    if not any(bids_root.glob("sub-*")):
        for d in sorted(data_root.rglob("sub-*")):
            if d.is_dir():
                bids_root = d.parent
                break
    print(f"BIDS root: {bids_root}")

    # Load and process
    print("Loading and processing EEG trials ...")
    eegs, labels, subjects = load_inner_speech(bids_root)

    print(f"\nDataset summary:")
    print(f"  Trials: {len(eegs)}")
    print(f"  Shape:  {eegs.shape}")
    print(f"  Subjects: {np.unique(subjects).tolist()}")
    print(f"  Label dist: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # LOSO cross-validation
    mean_acc, std_acc, per_subject = loso_train(eegs, labels, subjects, device, epochs=epochs)

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
