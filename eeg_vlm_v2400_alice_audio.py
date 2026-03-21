#!/usr/bin/env python3
"""
V2400: Alice EEG Pretrain → FC5 ZUNA Finetune → EEG→Audio→Whisper WER
=======================================================================
ACCOUNT: admin-1711
GPU: A100-40GB

PARADIGM: EEG → Mel Spectrogram → Griffin-Lim Audio → Whisper ASR → Text
  Contrast with V1xxx: EEG → Whisper features (TinyLatentAdapter) → Whisper decoder

PHASE 1 — Pretrain on Brennan Alice (49 subjects, 61ch, 512Hz, naturalistic listening)
  Input: 61ch EEG windows (5s, 256Hz after downsample) → (61, 1280)
  Target: mel spectrogram of paired Alice audio → (80, 500)

PHASE 2 — Finetune on FC5 ZUNA (32ch upscaled EEG, speaking)
  Input: 32ch EEG (from fc5_zuna_work/3_pt_output/*.pt) → (32, 1280)
  Target: mel spectrogram of FC5 audio → (80, 500)
  Split: train=29 sentences, val=5, held=8

EVAL — held set: EEG → mel → Griffin-Lim audio → Whisper → WER

AUTORESEARCH: modify hyperparams below, run modal run eeg_vlm_v2400_alice_audio.py
Output: held_window_asr_wer: X.XXXX
"""
from __future__ import annotations
import hashlib, io, json, math, os, random, re, time, warnings, zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def autocast_ctx(device: torch.device):
    if device.type == "cuda" and USE_BFLOAT16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)

# ── HYPERPARAMETERS (autoresearch modifies these) ──────────────────────────────
HIDDEN_DIM = _env_int("EEG_HIDDEN_DIM", 128)
N_LAYERS = _env_int("EEG_N_LAYERS", 2)
USE_ATTENTION = _env_bool("EEG_USE_ATTENTION", False)
PRETRAIN_EPOCHS = _env_int("EEG_PRETRAIN_EPOCHS", 40)
PRETRAIN_LR = _env_float("EEG_PRETRAIN_LR", 2e-4)
FINETUNE_EPOCHS = _env_int("EEG_FINETUNE_EPOCHS", 60)
FINETUNE_LR = _env_float("EEG_FINETUNE_LR", 0.0002)
FREEZE_ENCODER = _env_bool("EEG_FREEZE_ENCODER", True)
FREEZE_WARMUP_EPOCHS = _env_int("EEG_FREEZE_WARMUP_EPOCHS", 12)
BACKBONE_LR_SCALE = _env_float("EEG_BACKBONE_LR_SCALE", 0.25)
EARLY_STOP_PATIENCE = _env_int("EEG_EARLY_STOP_PATIENCE", 12)
PRETRAIN_EARLY_STOP_PATIENCE = _env_int("EEG_PRETRAIN_EARLY_STOP_PATIENCE", 8)
MEL_L1_W = 1
MFCC_W = 0.6
DELTA_W = 0.2
FREQ_DELTA_W = 0.15
HIGH_FREQ_W = 0.25
ENERGY_W = 0.15
VOICE_W = 0.15
SUBJECT_LIMIT = _env_int("EEG_SUBJECT_LIMIT", 10)
NOISE_AUG = _env_float("EEG_NOISE_AUG", 0.01)
N_AUG_FC5 = _env_int("EEG_N_AUG_FC5", 3)
FC5_GAIN_JITTER_STD = _env_float("EEG_FC5_GAIN_JITTER_STD", 0.15)
FC5_CHANNEL_DROPOUT_P = _env_float("EEG_FC5_CHANNEL_DROPOUT_P", 0.06)
FC5_MEL_SHIFT_FRAMES = _env_int("EEG_FC5_MEL_SHIFT_FRAMES", 8)
USE_ALICE_PRETRAIN = _env_bool("EEG_USE_ALICE_PRETRAIN", True)
RESUME_FROM_ALICE_CKPT = _env_bool("EEG_RESUME_FROM_ALICE_CKPT", False)
RUN_TAG = os.environ.get("EEG_RUN_TAG", "v2400")
ARTIFACT_ROOT = os.environ.get("EEG_ARTIFACT_ROOT", "/ckpts")
ALICE_CKPT_PATH = os.environ.get("EEG_ALICE_CKPT_PATH", f"{ARTIFACT_ROOT}/{RUN_TAG}_alice_pretrain.pt")
FC5_CKPT_PATH = os.environ.get("EEG_FC5_CKPT_PATH", f"{ARTIFACT_ROOT}/{RUN_TAG}_fc5_finetune.pt")
RUN_PRETRAIN_IF_RESUME_MISSING = _env_bool("EEG_RUN_PRETRAIN_IF_RESUME_MISSING", True)
STRICT_SPLIT_AUDIT = _env_bool("EEG_STRICT_SPLIT_AUDIT", True)
HASH_DECIMALS = 4
USE_MEL_GAN = False
USE_GAN_PRETRAIN = False
USE_GAN_FINETUNE = False
GAN_D_BASE = 32
GAN_D_LR = 5e-5
GAN_FM_W = 0.35
GAN_G_ADV_W_PRETRAIN = 0.02
GAN_G_ADV_W_FINETUNE = 0.015
GAN_WARMUP_EPOCHS_PRETRAIN = 8
GAN_WARMUP_EPOCHS_FINETUNE = 3
GAN_RAMP_EPOCHS = 10
SAVE_WAV_GRIFFIN_ITERS = 128
EVAL_GRIFFIN_ITERS = 96
QUICK_FINETUNE = _env_bool("EEG_QUICK_FINETUNE", False)
QUICK_FINETUNE_EPOCHS = _env_int("EEG_QUICK_FINETUNE_EPOCHS", 20)
EVAL_ALICE_CKPT_ONLY = _env_bool("EEG_EVAL_ALICE_CKPT_ONLY", False)
ALICE_EVAL_LIMIT = _env_int("EEG_ALICE_EVAL_LIMIT", 12)
ALICE_RAW_GRIFFIN_ITERS = 96
ALICE_CLARIFIED_GRIFFIN_ITERS = 192
ALICE_CLARIFY_HI_BOOST = 0.12
USE_HIFIGAN_VOCODER = _env_bool("EEG_USE_HIFIGAN_VOCODER", True)
USE_VOICE_ENHANCER = _env_bool("EEG_USE_VOICE_ENHANCER", True)
VOICE_ENHANCE_STRENGTH = 0.18
LATENT_ADAPTER_DIM = _env_int("EEG_LATENT_ADAPTER_DIM", 48)
LATENT_ADAPTER_DROPOUT = _env_float("EEG_LATENT_ADAPTER_DROPOUT", 0.05)
TOP_BLOCKS_UNFREEZE = _env_int("EEG_TOP_BLOCKS_UNFREEZE", 2)
ADAPTER_LR = _env_float("EEG_ADAPTER_LR", 3e-4)
TOP_BLOCK_LR_SCALE = _env_float("EEG_TOP_BLOCK_LR_SCALE", 0.12)
USE_BFLOAT16 = _env_bool("EEG_USE_BFLOAT16", True)
USE_TORCH_COMPILE = _env_bool("EEG_USE_TORCH_COMPILE", False)
SPEECH_AUX_W = _env_float("EEG_SPEECH_AUX_W", 0.35)
SPEECH_DIM = 48
SPEECH_HIDDEN_LAYER = 6
UNIT_AUX_W = _env_float("EEG_UNIT_AUX_W", 0.20)
UNIT_LABEL_SMOOTHING = _env_float("EEG_UNIT_LABEL_SMOOTHING", 0.0)
UNIT_TEMPORAL_W = _env_float("EEG_UNIT_TEMPORAL_W", 0.0)
UNIT_CHANGE_W = _env_float("EEG_UNIT_CHANGE_W", 0.0)
UNIT_DECODE_SMOOTH_PASSES = _env_int("EEG_UNIT_DECODE_SMOOTH_PASSES", 0)
UNIT_MIN_HOLD_FRAMES = _env_int("EEG_UNIT_MIN_HOLD_FRAMES", 1)
UNIT_VOCAB_SIZE = 1024
UNIT_CODEBOOKS = 8
ENCODEC_BANDWIDTH = 6.0
ENCODEC_SR = 24000
UNIT_FRAMES = 375  # Encodec 24kHz @ 6.0 kbps produces 75 frames/sec -> 375 frames for 5s
# ──────────────────────────────────────────────────────────────────────────────

# ── Modal setup ───────────────────────────────────────────────────────────────
app       = modal.App("eeg-vlm-v2400-alice-audio")
alice_vol = modal.Volume.from_name("alice-eeg-v2400", create_if_missing=True)
ckpt_vol  = modal.Volume.from_name("bt-ckpts-v2400",  create_if_missing=True)

FC5_LOCAL = Path("/Users/ariel/.gemini/antigravity/scratch")
ALICE_LOCAL = Path("/Users/ariel/alice_v2")

def _dl_whisper():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
    WhisperProcessor.from_pretrained("openai/whisper-small.en")


def _dl_vocoder():
    from transformers import SpeechT5HifiGan
    SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


def _dl_wav2vec():
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def _dl_encodec():
    from transformers import EncodecModel
    EncodecModel.from_pretrained("facebook/encodec_24khz")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "curl"])
    .pip_install([
        "torch>=2.4.0", "torchaudio>=2.4.0",
        "numpy", "scipy", "librosa>=0.10", "soundfile",
        "mne>=1.6.0", "jiwer", "transformers>=4.45.0",
        "cloudscraper", "requests", "accelerate",
    ])
    .run_function(_dl_whisper)
    .run_function(_dl_vocoder)
    .run_function(_dl_wav2vec)
    .run_function(_dl_encodec)
    # Embed local FC5 data + Alice audio directly into image
    .add_local_dir(
        FC5_LOCAL / "fc5_zuna_work/3_pt_output",
        remote_path="/fc5/zuna",
    )
    .add_local_file(
        FC5_LOCAL / "fc5_audio_20260317_215751.wav",
        remote_path="/fc5/audio.wav",
    )
    .add_local_file(
        FC5_LOCAL / "fc5_session_20260317_215751.json",
        remote_path="/fc5/session.json",
    )
    .add_local_file(
        FC5_LOCAL / "alice_v2/audio.zip",
        remote_path="/alice_audio/audio.zip",
    )
    .add_local_dir(
        ALICE_LOCAL,
        remote_path="/alice_v2",
    )
)

# ── EEG / Mel constants ───────────────────────────────────────────────────────
EEG_SR       = 256      # downsampled EEG rate
AUDIO_SR     = 16000    # target audio sample rate
N_MELS       = 80
HOP_LENGTH   = 160      # 10ms at 16kHz → 100fps
N_FFT        = 400
WIN_SAMPLES  = 1280     # 5s EEG window at 256Hz
MEL_FRAMES   = 500      # 5s × 100fps
WIN_SECONDS  = WIN_SAMPLES / EEG_SR
ALICE_VAL_SUBJECTS = 5
ALICE_TEST_SUBJECTS = 5

# ── Model ─────────────────────────────────────────────────────────────────────

class DepthwiseTemporal(nn.Module):
    def __init__(self, ch: int, k: int = 7):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, k, padding=k // 2, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, ch), ch)
    def forward(self, x):
        return F.gelu(self.norm(self.pw(self.dw(x)))) + x


def sample_arrays(sample):
    if isinstance(sample, dict):
        return sample["eeg"], sample["mel"]
    return sample


def sample_speech_target(sample) -> Optional[np.ndarray]:
    if isinstance(sample, dict):
        return sample.get("speech")
    return None


def sample_unit_target(sample) -> Optional[np.ndarray]:
    if isinstance(sample, dict):
        return sample.get("units")
    return None


def sample_manifest(sample: dict) -> dict:
    keep = [
        "dataset", "split", "subject_id", "segment_idx", "window_start_sec",
        "window_end_sec", "vhdr", "epoch_idx", "center_sec", "t_start",
        "t_end", "text",
    ]
    return {k: sample[k] for k in keep if k in sample}


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def split_alice_subjects(subject_ids: List[str]) -> Tuple[set[str], set[str], set[str]]:
    ids = sorted(subject_ids)
    n_test = min(ALICE_TEST_SUBJECTS, max(1, len(ids) // 10))
    n_val = min(ALICE_VAL_SUBJECTS, max(1, len(ids) // 10))
    test_ids = set(ids[-n_test:])
    val_ids = set(ids[-(n_test + n_val):-n_test] if n_test < len(ids) else [])
    train_ids = set(ids) - val_ids - test_ids
    return train_ids, val_ids, test_ids


def load_checkpoint_flexible(model: nn.Module, ckpt_path: Path, device: torch.device, label: str) -> bool:
    if not ckpt_path.exists():
        print(f"[ckpt] {label}: missing at {ckpt_path}")
        return False
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[ckpt] {label}: loaded {ckpt_path} "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[ckpt] {label}: first_missing={missing[:8]}")
    if unexpected:
        print(f"[ckpt] {label}: first_unexpected={unexpected[:8]}")
    return True


def _array_fingerprint(arr: np.ndarray, decimals: int = HASH_DECIMALS) -> str:
    rounded = np.round(np.asarray(arr, dtype=np.float32), decimals=decimals)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def _collect_fingerprints(samples: List[dict], key: str) -> dict[str, List[int]]:
    out: dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict) or key not in sample:
            continue
        fp = _array_fingerprint(sample[key])
        out.setdefault(fp, []).append(i)
    return out


def _count_overlap(a: dict[str, List[int]], b: dict[str, List[int]]) -> int:
    return len(set(a.keys()) & set(b.keys()))


def shift_mel_frames(mel: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(mel)
    if shift == 0:
        return mel.copy()
    if shift > 0:
        out[shift:] = mel[:-shift]
    else:
        s = -shift
        out[:-s] = mel[s:]
    return out


def ensure_mel_frames(mel: np.ndarray, frames: int = MEL_FRAMES) -> np.ndarray:
    """Force mel shape to (frames, n_mels) via crop/pad to avoid 500/501 mismatches."""
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError(f"mel must be 2D, got shape={mel.shape}")
    cur = mel.shape[0]
    if cur == frames:
        return mel
    if cur > frames:
        return mel[:frames]
    out = np.zeros((frames, mel.shape[1]), dtype=np.float32)
    out[:cur] = mel
    return out


def ensure_label_frames(labels: np.ndarray, frames: int = MEL_FRAMES) -> np.ndarray:
    """Force 1D discrete labels to a fixed frame count via nearest-neighbor resize."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape={labels.shape}")
    cur = labels.shape[0]
    if cur == frames:
        return labels.astype(np.int64, copy=False)
    if cur <= 0:
        return np.zeros((frames,), dtype=np.int64)
    if cur == 1:
        return np.full((frames,), int(labels[0]), dtype=np.int64)
    src = np.arange(cur, dtype=np.float32)
    dst = np.linspace(0, cur - 1, frames, dtype=np.float32)
    idx = np.clip(np.round(dst).astype(np.int64), 0, cur - 1)
    return labels[idx].astype(np.int64, copy=False)


def ensure_unit_frames(units: np.ndarray, frames: int = UNIT_FRAMES) -> np.ndarray:
    """Force 2D discrete unit labels to (frames, codebooks) via nearest-neighbor resize."""
    units = np.asarray(units)
    if units.ndim != 2:
        raise ValueError(f"units must be 2D, got shape={units.shape}")
    cur, n_codebooks = units.shape
    if cur == frames:
        return units.astype(np.int64, copy=False)
    if cur <= 0:
        return np.zeros((frames, n_codebooks), dtype=np.int64)
    if cur == 1:
        return np.repeat(units.astype(np.int64), frames, axis=0)
    dst = np.linspace(0, cur - 1, frames, dtype=np.float32)
    idx = np.clip(np.round(dst).astype(np.int64), 0, cur - 1)
    return units[idx].astype(np.int64, copy=False)


def augment_fc5_sample(sample: dict) -> dict:
    eeg = sample["eeg"].copy().astype(np.float32)
    mel = sample["mel"].copy().astype(np.float32)
    speech = sample_speech_target(sample)
    units = sample_unit_target(sample)
    speech = None if speech is None else speech.copy().astype(np.float32)
    units = None if units is None else units.copy().astype(np.int64)

    # Simulate session-to-session gain drift from live FC5 capture.
    global_gain = float(np.exp(np.random.randn() * FC5_GAIN_JITTER_STD))
    eeg *= global_gain
    ch_gain = np.exp(
        np.random.randn(eeg.shape[0], 1).astype(np.float32) * (FC5_GAIN_JITTER_STD * 0.35)
    )
    eeg *= ch_gain

    if FC5_CHANNEL_DROPOUT_P > 0:
        drop = np.random.rand(eeg.shape[0]) < FC5_CHANNEL_DROPOUT_P
        if np.any(drop):
            eeg[drop] = np.random.randn(int(drop.sum()), eeg.shape[1]).astype(np.float32) * 0.05

    if NOISE_AUG > 0:
        eeg += np.random.randn(*eeg.shape).astype(np.float32) * NOISE_AUG

    # Mild timing jitter between EEG and audio alignment.
    eeg_shift = int(np.random.randint(-32, 33))
    if eeg_shift != 0:
        eeg = np.roll(eeg, shift=eeg_shift, axis=1)
        if eeg_shift > 0:
            eeg[:, :eeg_shift] = 0.0
        else:
            eeg[:, eeg_shift:] = 0.0

    mel_shift = int(np.random.randint(-FC5_MEL_SHIFT_FRAMES, FC5_MEL_SHIFT_FRAMES + 1))
    if mel_shift != 0:
        mel = shift_mel_frames(mel, mel_shift)
        if speech is not None:
            speech = shift_mel_frames(speech, mel_shift)
        if units is not None:
            units = np.roll(units, mel_shift, axis=0)
            fill = units[0].copy() if mel_shift < 0 else units[-1].copy()
            if mel_shift > 0:
                units[:mel_shift] = fill
            else:
                units[mel_shift:] = fill
    mel = ensure_mel_frames(mel)
    if speech is not None:
        speech = ensure_mel_frames(speech, frames=MEL_FRAMES)
    if units is not None:
        units = ensure_unit_frames(units, frames=UNIT_FRAMES)

    aug = dict(sample)
    aug["eeg"] = eeg.astype(np.float32)
    aug["mel"] = mel.astype(np.float32)
    if speech is not None:
        aug["speech"] = speech.astype(np.float32)
    if units is not None:
        aug["units"] = units.astype(np.int64)
    return aug


def build_fc5_augmented_train_epoch(train_pairs: List[dict]) -> List[dict]:
    out = list(train_pairs)
    for sample in train_pairs:
        for _ in range(N_AUG_FC5):
            out.append(augment_fc5_sample(sample))
    return out


def make_finetune_optimizer(model: nn.Module, base_lr: float, full_unfrozen: bool):
    adapter_params = []
    top_params = []
    decoder_params = []
    frozen_backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            name.startswith("ch_proj")
            or name.startswith("latent_adapter")
            or name.startswith("voice_head")
            or name.startswith("speech_head")
            or name.startswith("unit_head")
        ):
            adapter_params.append(p)
        elif name.startswith("bigru"):
            top_params.append(p)
        elif (
            name.startswith("pre_mel")
            or name.startswith("mel_head")
            or name.startswith("up1")
            or name.startswith("up2")
            or name.startswith("skip_proj")
            or name.startswith("skip_norm")
            or name.startswith("temporal_refine")
            or name.startswith("enc_down")
            or name.startswith("pool_down")
        ):
            decoder_params.append(p)
        else:
            frozen_backbone_params.append(p)

    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": ADAPTER_LR})
    if top_params:
        groups.append({"params": top_params, "lr": base_lr * TOP_BLOCK_LR_SCALE})
    if full_unfrozen and decoder_params:
        groups.append({"params": decoder_params, "lr": base_lr * BACKBONE_LR_SCALE})
    if full_unfrozen and frozen_backbone_params:
        groups.append({"params": frozen_backbone_params, "lr": base_lr * (BACKBONE_LR_SCALE * 0.5)})

    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=1e-4)


def audit_fc5_split_integrity(train: List[dict], val: List[dict], held: List[dict], strict: bool = True) -> dict:
    report: dict[str, object] = {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
    }

    # Epoch index disjointness check.
    train_idx = {int(s["epoch_idx"]) for s in train if "epoch_idx" in s}
    val_idx = {int(s["epoch_idx"]) for s in val if "epoch_idx" in s}
    held_idx = {int(s["epoch_idx"]) for s in held if "epoch_idx" in s}
    overlap_tv = sorted(train_idx & val_idx)
    overlap_th = sorted(train_idx & held_idx)
    overlap_vh = sorted(val_idx & held_idx)
    report["epoch_overlap"] = {
        "train_val": overlap_tv,
        "train_held": overlap_th,
        "val_held": overlap_vh,
    }

    # Tensor fingerprint overlap check.
    t_eeg = _collect_fingerprints(train, "eeg")
    v_eeg = _collect_fingerprints(val, "eeg")
    h_eeg = _collect_fingerprints(held, "eeg")
    t_mel = _collect_fingerprints(train, "mel")
    v_mel = _collect_fingerprints(val, "mel")
    h_mel = _collect_fingerprints(held, "mel")
    report["fingerprint_overlap"] = {
        "eeg_train_val": _count_overlap(t_eeg, v_eeg),
        "eeg_train_held": _count_overlap(t_eeg, h_eeg),
        "eeg_val_held": _count_overlap(v_eeg, h_eeg),
        "mel_train_val": _count_overlap(t_mel, v_mel),
        "mel_train_held": _count_overlap(t_mel, h_mel),
        "mel_val_held": _count_overlap(v_mel, h_mel),
    }

    train_texts = {s.get("text", "").strip().lower() for s in train}
    val_texts = {s.get("text", "").strip().lower() for s in val}
    held_texts = {s.get("text", "").strip().lower() for s in held}
    report["text_overlap"] = {
        "train_val": len(train_texts & val_texts),
        "train_held": len(train_texts & held_texts),
        "val_held": len(val_texts & held_texts),
    }

    bad = bool(overlap_tv or overlap_th or overlap_vh)
    bad = bad or any(v > 0 for v in report["fingerprint_overlap"].values())
    report["strict_pass"] = not bad
    if strict and bad:
        raise RuntimeError(f"FC5 split leakage audit failed: {json.dumps(report, indent=2)}")
    return report


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        h = out_ch // 4
        self.k3  = nn.Sequential(nn.Conv1d(in_ch, h, 3,  padding=1,  bias=False), nn.GELU())
        self.k7  = nn.Sequential(nn.Conv1d(in_ch, h, 7,  padding=3,  bias=False), nn.GELU())
        self.k15 = nn.Sequential(nn.Conv1d(in_ch, h, 15, padding=7,  bias=False), nn.GELU())
        self.k31 = nn.Sequential(nn.Conv1d(in_ch, h, 31, padding=15, bias=False), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv1d(h * 4, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(), nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.fuse(torch.cat([self.k3(x), self.k7(x), self.k15(x), self.k31(x)], 1))


class FC5InputAdapter(nn.Module):
    """Higher-capacity 32ch adapter for transferring a 61ch Alice backbone to FC5."""

    def __init__(self, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(32, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.ms = MultiScaleBlock(out_ch, out_ch)
        self.res = DepthwiseTemporal(out_ch, 7)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.ms(x)
        x = self.res(x)
        return x * (0.75 + 0.25 * self.gate(x))


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.dw(y)
        y = self.bn(y)
        y = F.gelu(y)
        y = self.pw2(y).transpose(1, 2)
        return x + self.dropout(y)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            for cand in (8, 4, 2, 1):
                if dim % cand == 0:
                    heads = cand
                    break
        self.ff1 = FeedForwardBlock(dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.conv = ConvModule(dim, kernel_size=15, dropout=dropout)
        self.ff2 = FeedForwardBlock(dim, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ff1(x)
        y = self.attn_norm(x)
        y = self.attn(y, y, y, need_weights=False)[0]
        x = x + y
        x = self.conv(x)
        x = self.ff2(x)
        return self.out_norm(x)


class GatedSkipFuse(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.mix = nn.Conv1d(ch * 2, ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.gate = nn.Sequential(
            nn.Conv1d(ch * 2, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, skip], dim=1)
        fused = F.gelu(self.norm(self.mix(h)))
        gate = self.gate(h)
        return x + gate * fused


class LatentFiLMAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = LATENT_ADAPTER_DIM, dropout: float = LATENT_ADAPTER_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gb = self.net(x)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        return gamma * x + beta


class EEGToMelNet(nn.Module):
    """
    EEG (B, in_ch, WIN_SAMPLES) → log-mel (B, MEL_FRAMES, N_MELS)

    Architecture:
      channel/input adapter → multi-scale front-end → pooled BiGRU encoder
      → light U-Net decoder → mel head
    """
    def __init__(self, in_ch: int, hidden: int, n_layers: int,
                 use_attn: bool, n_mels: int = N_MELS,
                 mel_frames: int = MEL_FRAMES):
        super().__init__()
        self.mel_frames = mel_frames
        self.ch_proj = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.GELU(),
        )
        self.ms = MultiScaleBlock(hidden, hidden)
        self.pool64 = nn.AvgPool1d(8)
        self.bigru = nn.GRU(
            hidden, hidden // 2, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.25
        )
        self.enc_down1 = DepthwiseTemporal(hidden, 7)
        self.pool_down = nn.AvgPool1d(2)
        self.enc_down2 = DepthwiseTemporal(hidden, 7)
        self.latent_adapter = LatentFiLMAdapter(hidden)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            DepthwiseTemporal(hidden, 7),
        )
        self.skip_proj = nn.Conv1d(hidden * 2, hidden, 1, bias=False)
        self.skip_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            DepthwiseTemporal(hidden, 7),
        )
        self.temporal_refine = DepthwiseTemporal(hidden, 7)
        self.pre_mel = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mel_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_mels),
        )
        self.voice_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.speech_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, SPEECH_DIM),
        )
        self.unit_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, UNIT_CODEBOOKS * UNIT_VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self.ch_proj(x)
        x = self.ms(x)
        x = self.pool64(x)
        xt, _ = self.bigru(x.transpose(1, 2))
        xt = self.latent_adapter(xt)
        h = xt.transpose(1, 2)
        s1 = self.enc_down1(h)
        hd = self.enc_down2(self.pool_down(h))
        x = self.up1(hd)
        x = torch.cat([x, s1], dim=1)
        x = F.gelu(self.skip_norm(self.skip_proj(x)))
        x = self.up2(x)
        x = F.interpolate(x, size=self.mel_frames, mode="linear", align_corners=False)
        x = self.temporal_refine(x)
        xt = self.pre_mel(x.transpose(1, 2))
        mel = self.mel_head(xt)
        if not return_aux:
            return mel
        voice = self.voice_head(xt).squeeze(-1)
        speech = self.speech_head(xt)
        unit_xt = F.interpolate(x, size=UNIT_FRAMES, mode="linear", align_corners=False).transpose(1, 2)
        units = self.unit_head(unit_xt).view(unit_xt.shape[0], unit_xt.shape[1], UNIT_CODEBOOKS, UNIT_VOCAB_SIZE)
        return mel, {"voice_logits": voice, "speech": speech, "units": units}


class MelPatchDiscriminator(nn.Module):
    """Lightweight 2D discriminator on mel patches for adversarial clarity training."""

    def __init__(self, n_mels: int = N_MELS, base: int = GAN_D_BASE):
        super().__init__()
        chs = [1, base, base * 2, base * 4]
        self.blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(8, chs[i + 1]), chs[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=3, padding=1)

    def forward(self, mel_bt: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # mel_bt: (B, T, M)
        x = mel_bt.transpose(1, 2).unsqueeze(1)  # (B, 1, M, T)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        logits = self.head(x)
        return logits, feats


# ── Mel utilities ─────────────────────────────────────────────────────────────

def audio_to_mel(audio: np.ndarray, sr: int = AUDIO_SR,
                 n_mels: int = N_MELS, hop: int = HOP_LENGTH,
                 n_fft: int = N_FFT) -> np.ndarray:
    """Audio → log-mel (n_mels, T) in dB, normalized to [-1, 1]."""
    import librosa
    if sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
    mel = librosa.feature.melspectrogram(y=audio, sr=AUDIO_SR,
                                          n_mels=n_mels, hop_length=hop, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [-1, 1]
    mel_db = np.clip(mel_db, -80.0, 0.0) / 40.0 + 1.0  # [-1, 1]
    return mel_db.astype(np.float32)  # (n_mels, T)


def mel_to_audio(mel_db: np.ndarray, n_iter: int = 64,
                 hop: int = HOP_LENGTH, n_fft: int = N_FFT) -> np.ndarray:
    """Log-mel (n_mels, T) → audio waveform via Griffin-Lim."""
    import librosa
    # Denormalize
    mel_db_raw = (mel_db - 1.0) * 40.0  # [-80, 0]
    mel_power  = librosa.db_to_power(mel_db_raw)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=AUDIO_SR, n_fft=n_fft, hop_length=hop, n_iter=n_iter
    )
    return audio.astype(np.float32)


def mel_to_mfcc(mel_logdb: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Log-mel (n_mels, T) → MFCC (n_mfcc, T)."""
    import librosa
    mel_db_raw = (mel_logdb - 1.0) * 40.0
    mel_power  = librosa.db_to_power(mel_db_raw)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_power), n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def make_speech_feature_fn(device: torch.device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    @torch.no_grad()
    def extract(audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        inp = fe(x, sampling_rate=AUDIO_SR, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with autocast_ctx(device):
            out = model(**inp, output_hidden_states=True)
            hid = out.hidden_states[SPEECH_HIDDEN_LAYER][0].float()
        # Deterministic compression 768 -> SPEECH_DIM by averaging contiguous bands.
        t, d = hid.shape
        if d % SPEECH_DIM == 0:
            hid = hid.view(t, SPEECH_DIM, d // SPEECH_DIM).mean(dim=-1)
        else:
            hid = F.interpolate(
                hid.transpose(0, 1).unsqueeze(0),
                size=SPEECH_DIM,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        hid = F.interpolate(
            hid.transpose(0, 1).unsqueeze(0),
            size=max(1, int(round(len(x) / HOP_LENGTH))),
            mode="linear",
            align_corners=False,
        ).squeeze(0).transpose(0, 1)
        return hid.cpu().numpy().astype(np.float32)

    return extract


def make_encodec_unit_fn(device: torch.device):
    from transformers import EncodecModel
    import librosa

    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()

    @torch.no_grad()
    def extract(audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if AUDIO_SR != ENCODEC_SR:
            x = librosa.resample(x, orig_sr=AUDIO_SR, target_sr=ENCODEC_SR)
        wav = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
        with autocast_ctx(device):
            out = model.encode(wav, bandwidth=ENCODEC_BANDWIDTH)
        codes = out.audio_codes
        if isinstance(codes, list):
            codes = codes[0]
        # (1, 1, codebooks, frames) -> (frames, codebooks)
        codes = codes[0, 0].transpose(0, 1).detach().cpu().numpy().astype(np.int64)
        return ensure_unit_frames(codes, frames=UNIT_FRAMES)

    return extract


def load_encodec_decoder(device: torch.device):
    from transformers import EncodecModel

    return EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()


@torch.no_grad()
def units_to_waveform(
    unit_btq: np.ndarray,
    encodec_model,
    device: torch.device,
    enhance: bool = True,
) -> np.ndarray:
    import librosa

    units = np.asarray(unit_btq, dtype=np.int64)
    if units.ndim != 2:
        raise ValueError(f"unit_btq must be 2D, got shape={units.shape}")
    codes = torch.from_numpy(units.T).unsqueeze(0).unsqueeze(0).to(device)
    decoded = encodec_model.decode(audio_codes=codes, audio_scales=[None], last_frame_pad_length=0)
    audio = decoded.audio_values.squeeze().detach().float().cpu().numpy().astype(np.float32)
    if ENCODEC_SR != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=ENCODEC_SR, target_sr=AUDIO_SR)
    if enhance:
        audio = enhance_waveform(audio)
    return audio.astype(np.float32)


def logits_to_unit_codes(unit_logits: torch.Tensor, smooth_alpha: float = 0.7) -> np.ndarray:
    logits = unit_logits.detach().float().cpu()
    if logits.ndim != 3:
        raise ValueError(f"unit_logits must be 3D [T,Q,V], got shape={tuple(logits.shape)}")
    if logits.shape[0] > 1 and smooth_alpha > 0:
        prev = logits[:-1]
        curr = logits[1:]
        logits[1:] = (1.0 - smooth_alpha) * curr + smooth_alpha * prev
    conf = torch.softmax(logits, dim=-1).amax(dim=-1)  # [T,Q]
    codes = logits.argmax(dim=-1)  # [T,Q]
    if conf.shape[0] > 1:
        low_conf = conf < 0.22
        prev_codes = codes[:-1]
        codes[1:][low_conf[1:]] = prev_codes[low_conf[1:]]
    if codes.shape[0] >= 3 and UNIT_DECODE_SMOOTH_PASSES > 0:
        for _ in range(UNIT_DECODE_SMOOTH_PASSES):
            smoothed = codes.clone()
            smoothed[1:-1] = torch.where(
                codes[:-2] == codes[2:],
                codes[:-2],
                codes[1:-1],
            )
            codes = smoothed
    if codes.shape[0] > 1 and UNIT_MIN_HOLD_FRAMES > 1:
        min_hold = int(UNIT_MIN_HOLD_FRAMES)
        for q in range(codes.shape[1]):
            run_start = 0
            prev = int(codes[0, q])
            for t in range(1, codes.shape[0] + 1):
                curr = int(codes[t, q]) if t < codes.shape[0] else None
                if curr == prev:
                    continue
                run_len = t - run_start
                if run_len < min_hold and run_start > 0:
                    codes[run_start:t, q] = codes[run_start - 1, q]
                if t < codes.shape[0]:
                    run_start = t
                    prev = curr
    return codes.numpy().astype(np.int64)


def clarify_mel_for_listening(mel_bt: np.ndarray, hi_boost: float = ALICE_CLARIFY_HI_BOOST) -> np.ndarray:
    """Lightweight inference-only sharpening for listening diagnostics."""
    mel = np.asarray(mel_bt, dtype=np.float32).copy()
    freq_delta = mel[:, 1:] - mel[:, :-1]
    sharpen = np.zeros_like(mel)
    sharpen[:, 1:] += freq_delta
    sharpen[:, :-1] -= freq_delta
    mel = mel + 0.08 * sharpen
    hi0 = int(mel.shape[1] * 0.6)
    mel[:, hi0:] += hi_boost
    return np.clip(mel, -1.0, 1.0)


def enhance_waveform(audio: np.ndarray, strength: float = VOICE_ENHANCE_STRENGTH) -> np.ndarray:
    """Simple inference-time speech enhancer: preemphasis + mild denoise + compression."""
    x = np.asarray(audio, dtype=np.float32).copy()
    if x.size == 0:
        return x
    x = x - x.mean()
    if USE_VOICE_ENHANCER:
        pre = np.empty_like(x)
        pre[0] = x[0]
        pre[1:] = x[1:] - 0.97 * x[:-1]
        kernel = max(5, int(0.008 * AUDIO_SR))
        env = np.sqrt(np.convolve(pre ** 2, np.ones(kernel, dtype=np.float32) / kernel, mode="same") + 1e-8)
        floor = np.quantile(env, 0.15)
        gain = np.clip((env - floor) / (env + 1e-6), 0.15, 1.0)
        x = (1.0 - strength) * x + strength * (pre * gain)
        x = np.tanh(1.15 * x)
    peak = np.max(np.abs(x)) + 1e-8
    return (x / peak).astype(np.float32)


def normalized_mel_to_log(mel_bt: np.ndarray) -> np.ndarray:
    """Convert normalized [-1,1] mel-dB to log-power mel for SpeechT5 HiFi-GAN."""
    import librosa
    mel_db_raw = (np.asarray(mel_bt, dtype=np.float32).T - 1.0) * 40.0
    mel_pow = librosa.db_to_power(mel_db_raw)
    return np.log(np.maximum(mel_pow.T, 1e-9)).astype(np.float32)


@torch.no_grad()
def mel_to_waveform(
    mel_bt: np.ndarray,
    vocoder=None,
    device: Optional[torch.device] = None,
    griffin_iters: int = SAVE_WAV_GRIFFIN_ITERS,
    enhance: bool = True,
) -> np.ndarray:
    audio = None
    if USE_HIFIGAN_VOCODER and vocoder is not None:
        mel_in = torch.from_numpy(normalized_mel_to_log(mel_bt)).unsqueeze(0)
        if device is not None:
            mel_in = mel_in.to(device)
        try:
            audio = vocoder(mel_in).squeeze().detach().cpu().numpy().astype(np.float32)
        except Exception:
            audio = None
    if audio is None:
        audio = mel_to_audio(mel_bt.T, n_iter=griffin_iters)
    if enhance:
        audio = enhance_waveform(audio)
    return audio


# ── Loss ──────────────────────────────────────────────────────────────────────

def mel_loss(pred: torch.Tensor, target: torch.Tensor,
             l1_w: float = MEL_L1_W, mfcc_w: float = MFCC_W,
             delta_w: float = DELTA_W) -> torch.Tensor:
    """pred, target: (B, T, n_mels)."""
    loss = l1_w * F.l1_loss(pred, target)
    if delta_w > 0:
        d_pred   = pred[:, 1:] - pred[:, :-1]
        d_target = target[:, 1:] - target[:, :-1]
        loss = loss + delta_w * F.l1_loss(d_pred, d_target)
    if mfcc_w > 0:
        # Approximate MFCC via DCT on mel dim
        p_t = pred.transpose(1, 2)    # (B, n_mels, T)
        t_t = target.transpose(1, 2)
        n = p_t.shape[1]
        dct_mat = torch.tensor(
            [[math.cos(math.pi * k * (2 * i + 1) / (2 * n))
              for i in range(n)] for k in range(13)],
            dtype=pred.dtype, device=pred.device
        )  # (13, n_mels)
        p_mfcc = (dct_mat @ p_t).transpose(1, 2)[:, :, :13]  # (B, T, 13)
        t_mfcc = (dct_mat @ t_t).transpose(1, 2)[:, :, :13]
        loss = loss + mfcc_w * F.l1_loss(p_mfcc, t_mfcc)

    if FREQ_DELTA_W > 0:
        # Encourage stable spectral envelope transitions (across mel bins).
        f_pred = pred[:, :, 1:] - pred[:, :, :-1]
        f_tgt = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + FREQ_DELTA_W * F.l1_loss(f_pred, f_tgt)

    if HIGH_FREQ_W > 0:
        # Extra emphasis on high mel bins helps consonant sharpness / intelligibility.
        hi = slice(int(pred.shape[-1] * 0.65), pred.shape[-1])
        loss = loss + HIGH_FREQ_W * F.l1_loss(pred[:, :, hi], target[:, :, hi])
    if ENERGY_W > 0:
        loss = loss + ENERGY_W * F.l1_loss(pred.mean(dim=-1), target.mean(dim=-1))
    return loss


def voice_activity_target(mel: torch.Tensor) -> torch.Tensor:
    energy = mel.mean(dim=-1)
    thresh = energy.mean(dim=1, keepdim=True) + 0.05 * energy.std(dim=1, keepdim=True)
    return (energy > thresh).float()


def speech_aux_loss(pred: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    if target is None or SPEECH_AUX_W <= 0:
        return pred.new_tensor(0.0)
    return SPEECH_AUX_W * F.l1_loss(pred, target)


def unit_temporal_loss(logits: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    if target is None or logits.shape[1] <= 1 or (UNIT_TEMPORAL_W <= 0 and UNIT_CHANGE_W <= 0):
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=-1)
    loss = logits.new_tensor(0.0)
    if UNIT_TEMPORAL_W > 0:
        loss = loss + UNIT_TEMPORAL_W * F.l1_loss(probs[:, 1:], probs[:, :-1])
    if UNIT_CHANGE_W > 0:
        same_prob = (probs[:, 1:] * probs[:, :-1]).sum(dim=-1)
        pred_change = 1.0 - same_prob
        tgt_change = (target[:, 1:] != target[:, :-1]).float()
        loss = loss + UNIT_CHANGE_W * F.l1_loss(pred_change, tgt_change)
    return loss


def unit_aux_loss(logits: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    if target is None or UNIT_AUX_W <= 0:
        return logits.new_tensor(0.0)
    return UNIT_AUX_W * F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        label_smoothing=UNIT_LABEL_SMOOTHING,
    )


def gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    )


def gan_g_adv_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def feature_matching_loss(fake_feats: List[torch.Tensor], real_feats: List[torch.Tensor]) -> torch.Tensor:
    terms = []
    for ff, rf in zip(fake_feats, real_feats):
        terms.append(F.l1_loss(ff, rf.detach()))
    return sum(terms) / max(1, len(terms))


# ── Alice data loading ────────────────────────────────────────────────────────

def download_alice_subjects(alice_dir: Path, subject_limit: int = 0) -> List[Path]:
    """Download Alice BrainVision files to alice_dir. Returns list of .vhdr files."""
    import cloudscraper, re as _re
    alice_dir.mkdir(parents=True, exist_ok=True)

    BASE = "https://deepblue.lib.umich.edu"
    PAGE = f"{BASE}/data/concern/data_sets/bn999738r"
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )

    # Scrape file list
    print("[alice] Fetching file list...")
    html = scraper.get(PAGE, timeout=60).text
    # Find /data/downloads/<id> links
    links = list(dict.fromkeys(_re.findall(r'/data/downloads/[a-zA-Z0-9]+', html)))
    print(f"[alice] Found {len(links)} download links")

    # Filter for .vhdr/.eeg/.vmrk files
    vhdr_links, eeg_links, vmrk_links = [], [], []
    for link in links:
        resp = scraper.head(f"{BASE}{link}", timeout=30, allow_redirects=True)
        fname = Path(resp.url).name
        if fname.endswith(".vhdr"):
            vhdr_links.append((link, fname))
        elif fname.endswith(".eeg"):
            eeg_links.append((link, fname))
        elif fname.endswith(".vmrk"):
            vmrk_links.append((link, fname))

    # Sort and optionally limit
    subjects = sorted(vhdr_links, key=lambda x: x[1])
    if subject_limit > 0:
        subjects = subjects[:subject_limit]
    eeg_links  = sorted(eeg_links,  key=lambda x: x[1])[:len(subjects)]
    vmrk_links = sorted(vmrk_links, key=lambda x: x[1])[:len(subjects)]

    all_downloads = subjects + eeg_links + vmrk_links
    vhdr_files = []

    for link, fname in all_downloads:
        out_path = alice_dir / fname
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[alice] cached: {fname}")
            if fname.endswith(".vhdr"):
                vhdr_files.append(out_path)
            continue
        print(f"[alice] downloading {fname}...", end="", flush=True)
        t0 = time.time()
        r = scraper.get(f"{BASE}{link}", stream=True, timeout=300)
        with open(out_path, "wb") as fout:
            for chunk in r.iter_content(65536):
                fout.write(chunk)
        print(f" {out_path.stat().st_size/1e6:.1f}MB in {time.time()-t0:.0f}s")
        if fname.endswith(".vhdr"):
            vhdr_files.append(out_path)

    return vhdr_files


def load_alice_subject(vhdr_path: Path, audio_zip_path: Path,
                       target_sr: int = EEG_SR,
                       max_windows: int = 500,
                       speech_feature_fn=None,
                       unit_feature_fn=None) -> List[dict]:
    """Load one Alice subject into leak-safe window samples with metadata."""
    import mne, librosa, soundfile as sf

    # ── Load EEG ──
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    # Drop VEOG if present
    drops = [ch for ch in raw.ch_names if "veog" in ch.lower() or "eog" in ch.lower()]
    if drops:
        raw.drop_channels(drops)
    eeg_ch = len(raw.ch_names)
    eeg_data = raw.get_data().astype(np.float32)  # (ch, T)

    # Resample to target_sr
    if int(raw.info["sfreq"]) != target_sr:
        from scipy.signal import resample_poly
        eeg_data = resample_poly(eeg_data, up=target_sr,
                                 down=int(raw.info["sfreq"]), axis=1).astype(np.float32)

    # Z-score per channel
    mu = eeg_data.mean(axis=1, keepdims=True)
    sd = eeg_data.std(axis=1, keepdims=True) + 1e-8
    eeg_data = (eeg_data - mu) / sd

    # ── Load audio from zip, falling back to the mounted audio directory ──
    audio_segments = []
    audio_mode = "missing"
    if audio_zip_path.exists():
        with zipfile.ZipFile(audio_zip_path, "r") as zf:
            wav_names = sorted(
                [n for n in zf.namelist() if n.lower().endswith(".wav")
                 and not Path(n).name.startswith("._")],
                key=lambda n: int(m.group(1)) if (m := re.search(r"SoundFile(\d+)", n)) else 9999
            )
            for wn in wav_names:
                raw_bytes = io.BytesIO(zf.read(wn))
                aud, aud_sr = sf.read(raw_bytes, dtype="float32")
                if aud.ndim == 2:
                    aud = aud.mean(1)
                if aud_sr != AUDIO_SR:
                    aud = librosa.resample(aud, orig_sr=aud_sr, target_sr=AUDIO_SR)
                audio_segments.append(aud.astype(np.float32))
        if audio_segments:
            audio_mode = "zip"

    if not audio_segments:
        audio_dir = vhdr_path.parent / "audio"
        wav_paths = sorted(
            [p for p in audio_dir.glob("*.wav") if not p.name.startswith("._")],
            key=lambda p: int(m.group(1)) if (m := re.search(r"SoundFile(\d+)", p.name)) else 9999
        )
        for wav_path in wav_paths:
            aud, aud_sr = sf.read(str(wav_path), dtype="float32")
            if aud.ndim == 2:
                aud = aud.mean(1)
            if aud_sr != AUDIO_SR:
                aud = librosa.resample(aud, orig_sr=aud_sr, target_sr=AUDIO_SR)
            audio_segments.append(aud.astype(np.float32))
        if audio_segments:
            audio_mode = "dir"

    # ── Get chapter onset times from events ──
    raw_onsets = []
    if hasattr(raw, "annotations") and len(raw.annotations) > 0:
        raw_onsets = [
            float(ann["onset"])
            for ann in raw.annotations
            if "Stimulus" in str(ann.get("description", ""))
        ]

    if raw_onsets:
        segment_starts = raw_onsets[:len(audio_segments)]
        onset_mode = "annotations"
    else:
        # Brennan Alice files do not always expose "Stimulus" annotations cleanly in
        # Modal; align each chapter audio segment sequentially through the recording.
        segment_starts = []
        cursor_sec = 0.0
        for audio_seg in audio_segments:
            segment_starts.append(cursor_sec)
            cursor_sec += len(audio_seg) / AUDIO_SR
        onset_mode = "sequential"

    print(
        f"[alice] {vhdr_path.name}: annotations={len(getattr(raw, 'annotations', []))} "
        f"usable_onsets={len(raw_onsets)} audio_segments={len(audio_segments)} "
        f"audio_mode={audio_mode} mode={onset_mode}"
    )

    # ── Build (eeg_window, mel_window) pairs ──
    pairs = []
    stride_samples = WIN_SAMPLES // 4  # 25% stride = 1.25s
    subject_id = vhdr_path.stem
    win_audio_samples = int(WIN_SECONDS * AUDIO_SR)

    for seg_idx, (onset_sec, audio_seg) in enumerate(
        zip(segment_starts, audio_segments[:len(segment_starts)])
    ):
        onset_eeg = int(onset_sec * target_sr)
        audio_dur = len(audio_seg) / AUDIO_SR
        seg_eeg_len = min(int(audio_dur * target_sr),
                          eeg_data.shape[1] - onset_eeg)
        if seg_eeg_len < WIN_SAMPLES:
            continue

        seg_eeg = eeg_data[:, onset_eeg: onset_eeg + seg_eeg_len]
        mel_full = audio_to_mel(audio_seg)  # (n_mels, T_mel)
        speech_full = speech_feature_fn(audio_seg) if speech_feature_fn is not None else None
        unit_full = unit_feature_fn(audio_seg) if unit_feature_fn is not None else None
        unit_rate = (len(unit_full) / max(audio_dur, 1e-6)) if unit_full is not None else 0.0

        n_wins = (seg_eeg_len - WIN_SAMPLES) // stride_samples + 1
        for wi in range(n_wins):
            if len(pairs) >= max_windows:
                break
            eeg_st   = wi * stride_samples
            time_st  = eeg_st / target_sr
            time_en  = time_st + WIN_SAMPLES / target_sr  # +5s

            mel_st = int(time_st * AUDIO_SR / HOP_LENGTH)
            mel_en = mel_st + MEL_FRAMES
            if mel_en > mel_full.shape[1]:
                break
            aud_st = int(round(time_st * AUDIO_SR))
            aud_en = aud_st + win_audio_samples
            if aud_en > len(audio_seg):
                break

            eeg_win = seg_eeg[:, eeg_st: eeg_st + WIN_SAMPLES]
            mel_win = mel_full[:, mel_st: mel_en].T  # (MEL_FRAMES, n_mels)
            audio_win = audio_seg[aud_st: aud_en].astype(np.float32)
            speech_win = None
            if speech_full is not None:
                speech_win = ensure_mel_frames(speech_full[mel_st: mel_en], frames=MEL_FRAMES)
            unit_win = None
            if unit_full is not None:
                unit_st = int(round(time_st * unit_rate))
                unit_en = unit_st + UNIT_FRAMES
                if unit_en > len(unit_full):
                    break
                unit_win = ensure_unit_frames(unit_full[unit_st: unit_en], frames=UNIT_FRAMES)
            pairs.append({
                "dataset": "alice",
                "subject_id": subject_id,
                "segment_idx": int(seg_idx),
                "window_start_sec": float(onset_sec + time_st),
                "window_end_sec": float(onset_sec + time_en),
                "vhdr": vhdr_path.name,
                "eeg": eeg_win.astype(np.float32),
                "mel": ensure_mel_frames(mel_win).astype(np.float32),
                "speech": speech_win.astype(np.float32) if speech_win is not None else None,
                "units": unit_win.astype(np.int64) if unit_win is not None else None,
                "audio": audio_win,
            })

        if len(pairs) >= max_windows:
            break

    return pairs


# ── FC5 data loading ───────────────────────────────────────────────────────────

def load_fc5_data(zuna_dir: Path, audio_path: Path, session_path: Path,
                  speech_feature_fn=None,
                  unit_feature_fn=None,
                  ) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Load FC5 ZUNA epochs + audio → 5s window pairs aligned to the live decoder path.
    Returns (train_pairs, val_pairs, held_pairs).
    Each pair: {"eeg": (32, 1280), "mel": (MEL_FRAMES, n_mels), "audio": (80000,),
                "text": str, "center_sec": float}
    """
    import librosa, soundfile as sf

    # Load ZUNA epochs (32ch, 256Hz)
    pt_files = sorted(zuna_dir.glob("*.pt"))
    all_epochs: List[np.ndarray] = []
    for pt_f in pt_files:
        d = torch.load(pt_f, map_location="cpu", weights_only=False)
        epochs_list = d["data"]  # list of np arrays
        meta = d["metadata"]
        n_epochs = meta.get("n_epochs", len(epochs_list))
        for ep in epochs_list[:n_epochs]:
            arr = np.array(ep, dtype=np.float32)  # (32, 1280)
            if arr.shape != (32, WIN_SAMPLES):
                continue
            all_epochs.append(arr)
    print(f"[fc5] loaded {len(all_epochs)} ZUNA epochs (32ch, 5s each)")

    # Z-score each epoch per-channel
    norm_epochs = []
    for ep in all_epochs:
        mu = ep.mean(axis=1, keepdims=True)
        sd = ep.std(axis=1, keepdims=True) + 1e-8
        norm_epochs.append((ep - mu) / sd)
    all_epochs = norm_epochs

    # Load full audio
    audio, audio_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(1)
    if audio_sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=AUDIO_SR)
    mel_full = audio_to_mel(audio)  # (n_mels, T_mel)

    # Load session JSON → sentence events
    sess = json.loads(session_path.read_text())
    events = sess["events"]
    sentence_events = [e for e in events if e[1] == "sentence"]
    sentence_events.sort(key=lambda e: e[0])  # sort by time

    # Build epoch-centered pairs.
    rec_dur = sess.get("eeg_duration_s", len(all_epochs) * WIN_SECONDS)
    pairs = []
    epoch_dur = WIN_SECONDS
    win_audio_samples = int(WIN_SECONDS * AUDIO_SR)

    for epoch_idx, eeg_win in enumerate(all_epochs):
        t_center = (epoch_idx + 0.5) * epoch_dur
        active_idx = None
        for si, ev in enumerate(sentence_events):
            t_start = ev[0]
            t_end = (sentence_events[si + 1][0] if si + 1 < len(sentence_events) else rec_dur)
            if t_start <= t_center < t_end:
                active_idx = si
                break
        if active_idx is None:
            continue

        ev = sentence_events[active_idx]
        t_start = ev[0]
        t_end = (sentence_events[active_idx + 1][0] if active_idx + 1 < len(sentence_events) else rec_dur)
        text = ev[2].get("text", "")

        audio_center = int(t_center * AUDIO_SR)
        audio_st = max(0, audio_center - win_audio_samples // 2)
        audio_en = min(len(audio), audio_st + win_audio_samples)
        if audio_en - audio_st < win_audio_samples:
            audio_st = max(0, audio_en - win_audio_samples)
        audio_win = audio[audio_st:audio_en]
        if len(audio_win) < win_audio_samples:
            audio_win = np.pad(audio_win, (0, win_audio_samples - len(audio_win)))
        audio_win = audio_win.astype(np.float32)
        mel_win = ensure_mel_frames(audio_to_mel(audio_win).T.astype(np.float32))
        speech_win = None
        if speech_feature_fn is not None:
            speech_win = ensure_mel_frames(speech_feature_fn(audio_win), frames=MEL_FRAMES)
        unit_win = None
        if unit_feature_fn is not None:
            unit_win = ensure_unit_frames(unit_feature_fn(audio_win), frames=UNIT_FRAMES)

        pairs.append({
            "dataset": "fc5",
            "epoch_idx": int(epoch_idx),
            "center_sec": float(t_center),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "text": text,
            "eeg": eeg_win.astype(np.float32),
            "mel": mel_win,
            "speech": speech_win.astype(np.float32) if speech_win is not None else None,
            "units": unit_win.astype(np.int64) if unit_win is not None else None,
            "audio": audio_win,
        })

    # Split contiguously in time so evaluation matches future live-window deployment.
    n_total = len(pairs)
    n_held  = 8
    n_val   = 5
    n_train = n_total - n_held - n_val
    train = pairs[:n_train]
    val   = pairs[n_train: n_train + n_val]
    held  = pairs[n_train + n_val:]
    print(f"[fc5] train={len(train)} val={len(val)} held={len(held)}")
    return train, val, held


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, pairs: list, opt: torch.optim.Optimizer,
                device: torch.device, noise: float = NOISE_AUG):
    model.train()
    random.shuffle(pairs)
    total = 0.0
    for sample in pairs:
        eeg, mel = sample_arrays(sample)
        speech = sample_speech_target(sample)
        units = sample_unit_target(sample)
        if mel.shape[0] != MEL_FRAMES:
            mel = ensure_mel_frames(mel)
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)
        y = torch.from_numpy(mel).unsqueeze(0).to(device)
        y_speech = torch.from_numpy(speech).unsqueeze(0).to(device) if speech is not None else None
        y_units = torch.from_numpy(units).unsqueeze(0).long().to(device) if units is not None else None
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        with autocast_ctx(device):
            pred, aux = model(x, return_aux=True)
            loss = mel_loss(pred, y)
            if VOICE_W > 0 and "voice_logits" in aux:
                v_tgt = voice_activity_target(y)
                loss = loss + VOICE_W * F.binary_cross_entropy_with_logits(aux["voice_logits"], v_tgt)
            if "speech" in aux:
                loss = loss + speech_aux_loss(aux["speech"], y_speech)
            if "units" in aux:
                loss = loss + unit_aux_loss(aux["units"], y_units)
                loss = loss + unit_temporal_loss(aux["units"], y_units)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / max(1, len(pairs))


def train_epoch_gan(
    model: nn.Module,
    disc: nn.Module,
    pairs: list,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    adv_w: float,
    fm_w: float = GAN_FM_W,
    noise: float = NOISE_AUG,
) -> Tuple[float, float, float, float]:
    model.train()
    disc.train()
    random.shuffle(pairs)
    total_g, total_d, total_adv, total_fm = 0.0, 0.0, 0.0, 0.0

    for sample in pairs:
        eeg, mel = sample_arrays(sample)
        speech = sample_speech_target(sample)
        units = sample_unit_target(sample)
        if mel.shape[0] != MEL_FRAMES:
            mel = ensure_mel_frames(mel)
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)
        y = torch.from_numpy(mel).unsqueeze(0).to(device)
        y_speech = torch.from_numpy(speech).unsqueeze(0).to(device) if speech is not None else None
        y_units = torch.from_numpy(units).unsqueeze(0).long().to(device) if units is not None else None
        if noise > 0:
            x = x + torch.randn_like(x) * noise

        with torch.no_grad():
            with autocast_ctx(device):
                pred_det = model(x)
        y_disc = y.float()
        pred_det_disc = pred_det.detach().float()
        real_logits, _ = disc(y_disc)
        fake_logits_d, _ = disc(pred_det_disc)
        d_loss = gan_d_loss(real_logits, fake_logits_d)
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt_d.step()

        with autocast_ctx(device):
            pred, aux = model(x, return_aux=True)
            rec = mel_loss(pred, y)
            if VOICE_W > 0 and "voice_logits" in aux:
                v_tgt = voice_activity_target(y)
                rec = rec + VOICE_W * F.binary_cross_entropy_with_logits(aux["voice_logits"], v_tgt)
            if "speech" in aux:
                rec = rec + speech_aux_loss(aux["speech"], y_speech)
            if "units" in aux:
                rec = rec + unit_aux_loss(aux["units"], y_units)
                rec = rec + unit_temporal_loss(aux["units"], y_units)
        pred_disc = pred.float()
        fake_logits_g, fake_feats = disc(pred_disc)
        with torch.no_grad():
            _, real_feats = disc(y_disc)
        adv = gan_g_adv_loss(fake_logits_g)
        fm = feature_matching_loss(fake_feats, real_feats)
        g_loss = rec + adv_w * adv + fm_w * fm
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_g.step()

        total_g += g_loss.item()
        total_d += d_loss.item()
        total_adv += adv.item()
        total_fm += fm.item()

    denom = max(1, len(pairs))
    return total_g / denom, total_d / denom, total_adv / denom, total_fm / denom


@torch.no_grad()
def val_loss(model: nn.Module, pairs: list, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for sample in pairs:
        eeg, mel = sample_arrays(sample)
        speech = sample_speech_target(sample)
        units = sample_unit_target(sample)
        if mel.shape[0] != MEL_FRAMES:
            mel = ensure_mel_frames(mel)
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)
        y = torch.from_numpy(mel).unsqueeze(0).to(device)
        y_speech = torch.from_numpy(speech).unsqueeze(0).to(device) if speech is not None else None
        y_units = torch.from_numpy(units).unsqueeze(0).long().to(device) if units is not None else None
        with autocast_ctx(device):
            pred, aux = model(x, return_aux=True)
            loss = mel_loss(pred, y)
            if VOICE_W > 0 and "voice_logits" in aux:
                v_tgt = voice_activity_target(y)
                loss = loss + VOICE_W * F.binary_cross_entropy_with_logits(aux["voice_logits"], v_tgt)
            if "speech" in aux:
                loss = loss + speech_aux_loss(aux["speech"], y_speech)
            if "units" in aux:
                loss = loss + unit_aux_loss(aux["units"], y_units)
                loss = loss + unit_temporal_loss(aux["units"], y_units)
        total += loss.item()
    return total / max(1, len(pairs))


# ── Evaluation (WER) ──────────────────────────────────────────────────────────

def decode_sentence(model: nn.Module, pair: dict, device: torch.device,
                    whisper_model, whisper_proc, vocoder=None, encodec_model=None) -> str:
    """EEG → speech units → Encodec decode → Whisper → text."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(pair["eeg"]).unsqueeze(0).to(device)
        with autocast_ctx(device):
            pred_mel, aux = model(x, return_aux=True)
            pred_units = logits_to_unit_codes(aux["units"][0])
            pred_mel = pred_mel[0].float().cpu().numpy()
    if encodec_model is not None:
        audio = units_to_waveform(pred_units, encodec_model=encodec_model, device=device, enhance=True)
    else:
        audio = mel_to_waveform(pred_mel, vocoder=vocoder, device=device, griffin_iters=EVAL_GRIFFIN_ITERS, enhance=True)
    # Whisper inference via transformers
    inputs = whisper_proc(audio, sampling_rate=AUDIO_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = whisper_model.generate(**inputs)
    text = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text


@torch.no_grad()
def transcribe_audio(audio: np.ndarray, device: torch.device, whisper_model, whisper_proc) -> str:
    inputs = whisper_proc(audio, sampling_rate=AUDIO_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = whisper_model.generate(**inputs)
    return whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def save_example_wavs(
    model: nn.Module,
    pairs: list,
    device: torch.device,
    out_dir: Path,
    tag: str,
    limit: int = 3,
    vocoder=None,
    encodec_model=None,
):
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    items = []
    for idx, pair in enumerate(pairs[:limit]):
        if isinstance(pair, dict):
            eeg = pair["eeg"]
            mel = pair["mel"]
        else:
            eeg, mel = pair
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast_ctx(device):
                pred_mel, aux = model(x, return_aux=True)
                pred_mel = pred_mel[0].float().cpu().numpy()
                pred_units = logits_to_unit_codes(aux["units"][0])
        if encodec_model is not None:
            pred_audio = units_to_waveform(pred_units, encodec_model=encodec_model, device=device, enhance=True)
        else:
            pred_audio = mel_to_waveform(pred_mel, vocoder=vocoder, device=device, griffin_iters=SAVE_WAV_GRIFFIN_ITERS, enhance=True)
        use_true_ref = isinstance(pair, dict) and "audio" in pair
        ref_audio = pair["audio"] if use_true_ref else mel_to_waveform(mel, vocoder=vocoder, device=device, griffin_iters=SAVE_WAV_GRIFFIN_ITERS, enhance=False)
        pred_path = out_dir / f"{tag}_{idx:02d}_pred.wav"
        ref_path = out_dir / f"{tag}_{idx:02d}_ref.wav"
        sf.write(pred_path, pred_audio, AUDIO_SR)
        sf.write(ref_path, ref_audio, AUDIO_SR)
        print(f"[save] {pred_path}")
        print(f"[save] {ref_path}")
        items.append(
            {
                "idx": idx,
                "tag": tag,
                "dataset": pair.get("dataset", "tuple") if isinstance(pair, dict) else "tuple",
                "ref_source": "true_audio" if use_true_ref else "mel_reconstruction",
                "pred_seconds": float(len(pred_audio) / AUDIO_SR),
                "ref_seconds": float(len(ref_audio) / AUDIO_SR),
                "pred_wav": str(pred_path),
                "ref_wav": str(ref_path),
            }
        )
    if items:
        write_jsonl(out_dir / f"{tag}_meta.jsonl", items)


def compute_wer(hyps: List[str], refs: List[str]) -> float:
    import jiwer
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    wers = []
    for h, r in zip(hyps, refs):
        try:
            w = jiwer.wer(r, h,
                          truth_transform=transform,
                          hypothesis_transform=transform)
        except Exception:
            w = 1.0
        wers.append(min(w, 1.0))
    return float(np.mean(wers))


def strict_eval_report(
    model: nn.Module,
    held_pairs: List[dict],
    device: torch.device,
    whisper_model,
    whisper_proc,
    vocoder=None,
    encodec_model=None,
) -> dict:
    """Compute leak-resistant evaluation metrics and controls on held set."""
    import jiwer

    preds = []
    refs_audio_asr = []
    refs_script = []
    shuffled_targets = []

    for pair in held_pairs:
        preds.append(decode_sentence(model, pair, device, whisper_model, whisper_proc, vocoder=vocoder, encodec_model=encodec_model))
        refs_audio_asr.append(transcribe_audio(pair["audio"], device, whisper_model, whisper_proc))
        refs_script.append(pair.get("text", ""))

    if len(held_pairs) > 1:
        for i, pair in enumerate(held_pairs):
            neg_pair = held_pairs[(i + 1) % len(held_pairs)]
            neg_sample = dict(pair)
            neg_sample["eeg"] = neg_pair["eeg"]
            shuffled_targets.append(
                decode_sentence(model, neg_sample, device, whisper_model, whisper_proc, vocoder=vocoder, encodec_model=encodec_model)
            )
    else:
        shuffled_targets = [""]

    report = {
        "held_window_asr_wer": compute_wer(preds, refs_audio_asr),
        "held_vs_script_wer": compute_wer(preds, refs_script),
        "ref_audio_vs_script_wer": compute_wer(refs_audio_asr, refs_script),
        "negative_control_shuffled_eeg_vs_ref_audio_wer": compute_wer(shuffled_targets, refs_audio_asr),
        "n_held": len(held_pairs),
    }

    per_item = []
    for i, (pred, ref_a, ref_s, neg) in enumerate(
        zip(preds, refs_audio_asr, refs_script, shuffled_targets)
    ):
        per_item.append(
            {
                "idx": i,
                "pred": pred,
                "ref_audio_asr": ref_a,
                "ref_script": ref_s,
                "neg_pred": neg,
                "pred_vs_ref_audio_wer": float(min(jiwer.wer(ref_a.lower(), pred.lower()), 1.0)),
                "pred_vs_script_wer": float(min(jiwer.wer(ref_s.lower(), pred.lower()), 1.0)),
                "neg_vs_ref_audio_wer": float(min(jiwer.wer(ref_a.lower(), neg.lower()), 1.0)),
            }
        )

    report["items"] = per_item
    return report


@torch.no_grad()
def eval_alice_ckpt_only(
    model: nn.Module,
    alice_pairs: List[dict],
    device: torch.device,
    whisper_model,
    whisper_proc,
    out_dir: Path,
    limit: int = ALICE_EVAL_LIMIT,
    vocoder=None,
    encodec_model=None,
) -> dict:
    import jiwer
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for idx, pair in enumerate(alice_pairs[:limit]):
        eeg = pair["eeg"]
        ref_mel = pair["mel"]
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)
        pred_mel, aux = model(x, return_aux=True)
        pred_mel = pred_mel[0].cpu().numpy()
        pred_units = logits_to_unit_codes(aux["units"][0])
        clar_mel = clarify_mel_for_listening(pred_mel)

        raw_audio = (
            units_to_waveform(pred_units, encodec_model=encodec_model, device=device, enhance=False)
            if encodec_model is not None else
            mel_to_waveform(pred_mel, vocoder=vocoder, device=device, griffin_iters=ALICE_RAW_GRIFFIN_ITERS, enhance=False)
        )
        clar_audio = mel_to_waveform(clar_mel, vocoder=vocoder, device=device, griffin_iters=ALICE_CLARIFIED_GRIFFIN_ITERS, enhance=True)
        ref_audio = pair["audio"] if "audio" in pair else mel_to_waveform(ref_mel, vocoder=vocoder, device=device, griffin_iters=SAVE_WAV_GRIFFIN_ITERS, enhance=False)

        raw_txt = transcribe_audio(raw_audio, device, whisper_model, whisper_proc)
        clar_txt = transcribe_audio(clar_audio, device, whisper_model, whisper_proc)
        ref_txt = transcribe_audio(ref_audio, device, whisper_model, whisper_proc)

        stem = f"alice_eval_{idx:02d}_{pair['subject_id']}_seg{pair['segment_idx']:02d}"
        raw_path = out_dir / f"{stem}_raw.wav"
        clar_path = out_dir / f"{stem}_clarified.wav"
        ref_path = out_dir / f"{stem}_ref.wav"
        sf.write(raw_path, raw_audio, AUDIO_SR)
        sf.write(clar_path, clar_audio, AUDIO_SR)
        sf.write(ref_path, ref_audio, AUDIO_SR)

        items.append({
            "idx": idx,
            "subject_id": pair["subject_id"],
            "segment_idx": pair["segment_idx"],
            "ref_audio_asr": ref_txt,
            "raw_pred_asr": raw_txt,
            "clarified_pred_asr": clar_txt,
            "raw_vs_ref_wer": float(min(jiwer.wer(ref_txt.lower(), raw_txt.lower()), 1.0)),
            "clarified_vs_ref_wer": float(min(jiwer.wer(ref_txt.lower(), clar_txt.lower()), 1.0)),
            "raw_wav": str(raw_path),
            "clarified_wav": str(clar_path),
            "ref_wav": str(ref_path),
        })
    return {
        "n_eval": len(items),
        "raw_asr_wer": compute_wer([x["raw_pred_asr"] for x in items], [x["ref_audio_asr"] for x in items]),
        "clarified_asr_wer": compute_wer([x["clarified_pred_asr"] for x in items], [x["ref_audio_asr"] for x in items]),
        "items": items,
    }


# ── Main Modal function ────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
    volumes={"/alice": alice_vol, "/ckpts": ckpt_vol},
    memory=32768,
)
def train_and_eval():
    import torch
    from transformers import SpeechT5HifiGan, WhisperForConditionalGeneration, WhisperProcessor

    t0_total = time.time()
    device = torch.device("cuda")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    print(
        "[config] "
        f"hidden={HIDDEN_DIM} layers={N_LAYERS} pretrain_ep={PRETRAIN_EPOCHS} finetune_ep={FINETUNE_EPOCHS} "
        f"pretrain_lr={PRETRAIN_LR} finetune_lr={FINETUNE_LR} adapter_lr={ADAPTER_LR} "
        f"unit_w={UNIT_AUX_W} unit_temp_w={UNIT_TEMPORAL_W} unit_change_w={UNIT_CHANGE_W} "
        f"speech_w={SPEECH_AUX_W} unit_hold={UNIT_MIN_HOLD_FRAMES} unit_smooth_passes={UNIT_DECODE_SMOOTH_PASSES}"
    )
    vocoder = None
    if USE_HIFIGAN_VOCODER:
        try:
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
            print("[audio] SpeechT5 HiFi-GAN vocoder loaded")
        except Exception as e:
            print(f"[audio] HiFi-GAN load failed, falling back to Griffin-Lim: {e}")
            vocoder = None
    speech_feature_fn = make_speech_feature_fn(device)
    print("[audio] Wav2Vec2 speech feature target loaded")
    unit_feature_fn = make_encodec_unit_fn(device)
    print("[audio] Encodec unit target loaded")
    encodec_decoder = load_encodec_decoder(device)
    print("[audio] Encodec neural decoder loaded")

    # ── Alice pretraining ──
    alice_train_split: List[dict] = []
    alice_val_split: List[dict] = []
    alice_test_split: List[dict] = []
    alice_pretrained = False
    n_ch_alice = 61  # retained only for compatibility with older checkpoints
    if USE_ALICE_PRETRAIN:
        alice_dir = Path("/alice_v2")
        vhdr_files = []
        print(f"\n[phase0] Alice mount exists: {alice_dir.exists()}")
        if alice_dir.exists():
            try:
                top = sorted(p.name for p in alice_dir.iterdir())[:25]
                print(f"[phase0] alice_v2 top-level sample: {top}")
            except Exception as e:
                print(f"[phase0] could not list alice_v2 contents: {e}")

        # Only accept complete BrainVision triplets.
        vhdr_files = []
        if alice_dir.exists():
            for vhdr in sorted(alice_dir.glob("S*.vhdr")):
                eeg = vhdr.with_suffix(".eeg")
                vmrk = vhdr.with_suffix(".vmrk")
                if eeg.exists() and vmrk.exists():
                    vhdr_files.append(vhdr)
        print(f"[phase0] Alice EEG ready from /alice_v2: {len(vhdr_files)} complete subjects")
        if not vhdr_files:
            raise RuntimeError("Alice mount is visible but contains no complete S*.vhdr/.eeg/.vmrk triplets")
        if vhdr_files:
            subject_ids = [vhdr.stem for vhdr in vhdr_files]
            train_subjects, val_subjects, test_subjects = split_alice_subjects(subject_ids)
            print(
                f"[phase0] Alice subject split: "
                f"train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)}"
            )
            print("\n[phase1] Building Alice (EEG, mel) windows...")
            audio_zip = Path("/alice_audio/audio.zip")
            alice_all_samples: List[dict] = []
            for vhdr in vhdr_files:
                subj_pairs = load_alice_subject(
                    vhdr, audio_zip, max_windows=500 // max(1, len(vhdr_files)) + 50,
                    speech_feature_fn=speech_feature_fn,
                    unit_feature_fn=unit_feature_fn,
                )
                subject_id = vhdr.stem
                split = (
                    "train" if subject_id in train_subjects else
                    "val" if subject_id in val_subjects else
                    "test"
                )
                for sample in subj_pairs:
                    sample["split"] = split
                alice_all_samples.extend(subj_pairs)
                print(f"  {vhdr.name}: +{len(subj_pairs)} windows → total={len(alice_all_samples)}")

            if alice_all_samples:
                n_ch_alice = alice_all_samples[0]["eeg"].shape[0]
            for sample in alice_all_samples:
                eeg = sample["eeg"]
                if eeg.shape[0] != n_ch_alice:
                    eeg = eeg[:n_ch_alice] if eeg.shape[0] >= n_ch_alice else np.pad(
                        eeg, ((0, n_ch_alice - eeg.shape[0]), (0, 0)))
                    sample["eeg"] = eeg.astype(np.float32)
                if sample["split"] == "train":
                    alice_train_split.append(sample)
                elif sample["split"] == "val":
                    alice_val_split.append(sample)
                else:
                    alice_test_split.append(sample)
            if alice_all_samples:
                print(
                    f"[phase1] Alice windows: "
                    f"train={len(alice_train_split)} val={len(alice_val_split)} "
                    f"test={len(alice_test_split)} shape={alice_all_samples[0]['eeg'].shape}"
                )
                write_jsonl(
                    Path("/ckpts/manifests/alice_train.jsonl"),
                    [sample_manifest(s) for s in alice_train_split],
                )
                write_jsonl(
                    Path("/ckpts/manifests/alice_val.jsonl"),
                    [sample_manifest(s) for s in alice_val_split],
                )
                write_jsonl(
                    Path("/ckpts/manifests/alice_test.jsonl"),
                    [sample_manifest(s) for s in alice_test_split],
                )
    else:
        print("[phase0] Alice pretraining disabled; training FC5-only")

    # ── Load FC5 ZUNA data (needed whether or not Alice available) ──
    print("\n[fc5] Loading FC5 ZUNA data...")
    fc5_train, fc5_val, fc5_held = load_fc5_data(
        zuna_dir=Path("/fc5/zuna"),
        audio_path=Path("/fc5/audio.wav"),
        session_path=Path("/fc5/session.json"),
        speech_feature_fn=speech_feature_fn,
        unit_feature_fn=unit_feature_fn,
    )
    split_audit = audit_fc5_split_integrity(
        fc5_train, fc5_val, fc5_held, strict=STRICT_SPLIT_AUDIT
    )
    print(f"[fc5] split_audit={json.dumps(split_audit)}")

    # ── Build model ──
    write_jsonl(Path("/ckpts/manifests/fc5_train.jsonl"), [sample_manifest(s) for s in fc5_train])
    write_jsonl(Path("/ckpts/manifests/fc5_val.jsonl"), [sample_manifest(s) for s in fc5_val])
    write_jsonl(Path("/ckpts/manifests/fc5_held.jsonl"), [sample_manifest(s) for s in fc5_held])

    if alice_train_split:
        model = EEGToMelNet(n_ch_alice, HIDDEN_DIM, N_LAYERS, USE_ATTENTION).to(device)
        disc = MelPatchDiscriminator().to(device) if (USE_MEL_GAN and USE_GAN_PRETRAIN) else None
        if device.type == "cuda" and USE_TORCH_COMPILE:
            model = torch.compile(model, mode="max-autotune")
            if disc is not None:
                disc = torch.compile(disc, mode="max-autotune")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[model] EEGToMelNet (Alice pretrain): {n_params/1e6:.2f}M params, "
              f"in_ch={n_ch_alice}, hidden={HIDDEN_DIM}")
        resumed_alice = False
        if RESUME_FROM_ALICE_CKPT:
            resumed_alice = load_checkpoint_flexible(
                model, Path(ALICE_CKPT_PATH), device, label="alice_pretrain"
            )
        if resumed_alice:
            alice_pretrained = True
            best_val = val_loss(model, alice_val_split, device) if alice_val_split else float("nan")
            print(f"[phase1] Resumed Alice checkpoint. Val loss={best_val:.4f}")
        elif not RUN_PRETRAIN_IF_RESUME_MISSING and RESUME_FROM_ALICE_CKPT:
            raise RuntimeError(
                f"Requested resume from Alice checkpoint, but none found at {ALICE_CKPT_PATH}"
            )
        else:
            print(f"\n[phase1] Alice pretrain — {PRETRAIN_EPOCHS} epochs, "
                  f"{len(alice_train_split)} train, {len(alice_val_split)} val, {len(alice_test_split)} test")
            opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
            opt_d = torch.optim.AdamW(disc.parameters(), lr=GAN_D_LR, weight_decay=1e-4) if disc is not None else None
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
            best_val, best_state, best_ep = 1e9, None, 0
            stale_pretrain = 0
            for ep in range(1, PRETRAIN_EPOCHS + 1):
                if disc is not None and ep > GAN_WARMUP_EPOCHS_PRETRAIN:
                    adv_w_ep = GAN_G_ADV_W_PRETRAIN * min(1.0, (ep - GAN_WARMUP_EPOCHS_PRETRAIN) / max(1, GAN_RAMP_EPOCHS))
                    tr, dtr, adv, fm = train_epoch_gan(
                        model,
                        disc,
                        alice_train_split,
                        opt,
                        opt_d,
                        device,
                        adv_w=adv_w_ep,
                    )
                    gan_log = f" d={dtr:.4f} adv={adv:.4f} fm={fm:.4f} adv_w={adv_w_ep:.4f}"
                else:
                    tr = train_epoch(model, alice_train_split, opt, device)
                    gan_log = ""
                vl = val_loss(model, alice_val_split, device)
                sched.step()
                if vl < best_val:
                    best_val = vl
                    best_ep = ep
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    stale_pretrain = 0
                    torch.save(
                        {
                            "state": best_state,
                            "n_ch": n_ch_alice,
                            "hidden": HIDDEN_DIM,
                            "n_layers": N_LAYERS,
                            "best_val": float(best_val),
                            "best_epoch": int(best_ep),
                        },
                        Path("/ckpts/v2400_alice_pretrain.pt"),
                    )
                    ckpt_vol.commit()
                else:
                    stale_pretrain += 1
                print(
                    f"  ep={ep:3d}  train={tr:.4f}  val={vl:.4f}  "
                    f"best={best_val:.4f}@{best_ep}  stale={stale_pretrain}{gan_log}"
                )
                if stale_pretrain >= PRETRAIN_EARLY_STOP_PATIENCE:
                    print(
                        f"  Alice early stop triggered at ep={ep} "
                        f"(patience={PRETRAIN_EARLY_STOP_PATIENCE})"
                    )
                    break
            model.load_state_dict(best_state)
            torch.save(
                {
                    "state": best_state,
                    "n_ch": n_ch_alice,
                    "hidden": HIDDEN_DIM,
                    "n_layers": N_LAYERS,
                    "best_val": float(best_val),
                    "best_epoch": int(best_ep),
                },
                Path("/ckpts/v2400_alice_pretrain.pt"),
            )
            ckpt_vol.commit()
            print(f"[phase1] Alice pretrain done. Best val loss={best_val:.4f} @ ep={best_ep}")
            alice_pretrained = True

        try:
            save_example_wavs(
                model,
                alice_test_split or alice_val_split,
                device,
                Path("/ckpts/v2400_examples/alice_test"),
                tag="alice_test",
                limit=3,
                vocoder=vocoder,
                encodec_model=encodec_decoder,
            )
        except Exception as e:
            print(f"[warn] Alice example WAV export failed: {e}")

        # Swap to FC5 input channels for finetune.
        model.ch_proj = FC5InputAdapter(HIDDEN_DIM).to(device)
        disc = MelPatchDiscriminator().to(device) if (USE_MEL_GAN and USE_GAN_FINETUNE) else None
    else:
        model = EEGToMelNet(32, HIDDEN_DIM, N_LAYERS, USE_ATTENTION).to(device)
        disc = MelPatchDiscriminator().to(device) if (USE_MEL_GAN and USE_GAN_FINETUNE) else None
        if device.type == "cuda" and USE_TORCH_COMPILE:
            model = torch.compile(model, mode="max-autotune")
            if disc is not None:
                disc = torch.compile(disc, mode="max-autotune")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[model] EEGToMelNet (FC5-only): {n_params/1e6:.2f}M params, "
              f"in_ch=32, hidden={HIDDEN_DIM}")

    if EVAL_ALICE_CKPT_ONLY:
        print("\n[alice_eval] Loading Whisper for Alice checkpoint-only evaluation...")
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small.en"
        ).to(device).eval()
        if not alice_pretrained:
            raise RuntimeError("Alice checkpoint-only eval requested, but no Alice checkpoint/model is loaded")
        alice_eval = eval_alice_ckpt_only(
            model,
            alice_test_split or alice_val_split,
            device,
            whisper_model,
            whisper_proc,
            Path("/ckpts/v2400_examples/alice_ckpt_eval"),
            limit=ALICE_EVAL_LIMIT,
            vocoder=vocoder,
            encodec_model=encodec_decoder,
        )
        for item in alice_eval["items"]:
            print(f"  idx={item['idx']:02d} subj={item['subject_id']} seg={item['segment_idx']:02d}")
            print(f"    ref : {item['ref_audio_asr'][:80]}")
            print(f"    raw : {item['raw_pred_asr'][:80]}  WER={item['raw_vs_ref_wer']:.2f}")
            print(f"    clar: {item['clarified_pred_asr'][:80]}  WER={item['clarified_vs_ref_wer']:.2f}")
        write_jsonl(Path("/ckpts/manifests/alice_ckpt_eval_items.jsonl"), alice_eval["items"])
        Path("/ckpts/manifests/alice_ckpt_eval_report.json").write_text(
            json.dumps({k: v for k, v in alice_eval.items() if k != "items"}, indent=2)
        )
        ckpt_vol.commit()
        print(f"\n---")
        print(f"alice_raw_asr_wer: {alice_eval['raw_asr_wer']:.6f}")
        print(f"alice_clarified_asr_wer: {alice_eval['clarified_asr_wer']:.6f}")
        print(f"alice_eval_n: {alice_eval['n_eval']}")
        return float(alice_eval["clarified_asr_wer"])

    # ── Phase 2: FC5 finetune ──
    if alice_pretrained:
        print(f"\n[phase2] FC5 finetune — Alice-pretrained backbone")
    else:
        print(f"\n[phase2] FC5 finetune — FC5-only (no Alice pretrain)")
    print(
        f"  Unit-focused finetune: unit_w={UNIT_AUX_W:.2f} temporal_w={UNIT_TEMPORAL_W:.2f} "
        f"change_w={UNIT_CHANGE_W:.2f} speech_w={SPEECH_AUX_W:.2f}"
    )
    if FREEZE_ENCODER and alice_pretrained:
        for name, p in model.named_parameters():
            train_now = (
                name.startswith("ch_proj")
                or name.startswith("latent_adapter")
                or name.startswith("voice_head")
                or name.startswith("speech_head")
                or name.startswith("unit_head")
                or name.startswith("bigru")
                or name.startswith("pre_mel")
                or name.startswith("mel_head")
            )
            p.requires_grad_(train_now)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"  Low-destruction transfer: train ch_proj + latent_adapter + bigru + mel_head "
            f"({trainable/1e3:.1f}K params)"
        )
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        print("  Full fine-tune (all layers trainable)")

    finetune_epochs = QUICK_FINETUNE_EPOCHS if QUICK_FINETUNE else FINETUNE_EPOCHS
    print(f"  Finetune epochs: {finetune_epochs} (quick={QUICK_FINETUNE})")
    ft_val = fc5_val
    full_unfrozen = not (FREEZE_ENCODER and alice_pretrained)
    opt2 = make_finetune_optimizer(model, FINETUNE_LR, full_unfrozen=full_unfrozen)
    opt2_d = torch.optim.AdamW(disc.parameters(), lr=GAN_D_LR, weight_decay=1e-4) if disc is not None else None
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=finetune_epochs)
    best_val2, best_state2, best_epoch2 = 1e9, None, 0
    stale = 0
    for ep in range(1, finetune_epochs + 1):
        if (
            FREEZE_ENCODER
            and alice_pretrained
            and ep == FREEZE_WARMUP_EPOCHS + 1
        ):
            print("  Progressive FC5 unfreeze: opening decoder + lower-rate backbone")
            for name, p in model.named_parameters():
                if (
                    name.startswith("up1")
                    or name.startswith("up2")
                    or name.startswith("skip_proj")
                    or name.startswith("skip_norm")
                    or name.startswith("temporal_refine")
                    or name.startswith("enc_down")
                    or name.startswith("pool_down")
                ):
                    p.requires_grad_(True)
            opt2 = make_finetune_optimizer(model, FINETUNE_LR, full_unfrozen=True)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=max(1, finetune_epochs - ep + 1)
            )

        ft_train_epoch = build_fc5_augmented_train_epoch(fc5_train)
        if disc is not None and ep > GAN_WARMUP_EPOCHS_FINETUNE:
            adv_w_ep = GAN_G_ADV_W_FINETUNE * min(1.0, (ep - GAN_WARMUP_EPOCHS_FINETUNE) / max(1, GAN_RAMP_EPOCHS))
            tr, dtr, adv, fm = train_epoch_gan(
                model,
                disc,
                ft_train_epoch,
                opt2,
                opt2_d,
                device,
                adv_w=adv_w_ep,
            )
            gan_log = f" d={dtr:.4f} adv={adv:.4f} fm={fm:.4f} adv_w={adv_w_ep:.4f}"
        else:
            tr = train_epoch(model, ft_train_epoch, opt2, device)
            gan_log = ""
        vl = val_loss(model, ft_val, device)
        sched2.step()
        if vl < best_val2:
            best_val2 = vl
            best_epoch2 = ep
            best_state2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            torch.save(
                {
                    "state": best_state2,
                    "hidden": HIDDEN_DIM,
                    "n_layers": N_LAYERS,
                    "best_val": float(best_val2),
                    "best_epoch": int(best_epoch2),
                },
                Path("/ckpts/v2400_fc5_finetune.pt"),
            )
            ckpt_vol.commit()
        else:
            stale += 1

        if ep % 5 == 0 or ep == finetune_epochs:
            print(
                f"  ep={ep:3d}  train={tr:.4f}  val={vl:.4f}  "
                f"best={best_val2:.4f}@{best_epoch2}  stale={stale}{gan_log}"
            )

        if stale >= EARLY_STOP_PATIENCE:
            print(f"  Early stop triggered at ep={ep} (patience={EARLY_STOP_PATIENCE})")
            break

    model.load_state_dict(best_state2)
    fc5_ckpt_path = Path("/ckpts/v2400_fc5_finetune.pt")
    torch.save({"state": best_state2, "hidden": HIDDEN_DIM, "n_layers": N_LAYERS}, fc5_ckpt_path)
    save_example_wavs(
        model,
        fc5_held,
        device,
        Path("/ckpts/v2400_examples/fc5_held"),
        tag="fc5_held",
        limit=3,
        vocoder=vocoder,
        encodec_model=encodec_decoder,
    )
    ckpt_vol.commit()

    # ── Phase 3: Evaluation on held set ──
    print("\n[phase3] Loading Whisper for ASR evaluation...")
    whisper_proc  = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small.en"
    ).to(device).eval()

    print("\n[eval] Held set decoding + strict controls:")
    eval_report = strict_eval_report(
        model,
        fc5_held,
        device,
        whisper_model,
        whisper_proc,
        vocoder=vocoder,
        encodec_model=encodec_decoder,
    )
    for item in eval_report["items"]:
        print(f"  idx={item['idx']:02d}")
        print(f"    ref_audio_asr: {item['ref_audio_asr'][:80]}")
        print(f"    pred         : {item['pred'][:80]}  WER={item['pred_vs_ref_audio_wer']:.2f}")
        print(f"    source_script: {item['ref_script'][:80]}  WER={item['pred_vs_script_wer']:.2f}")
        print(f"    neg_ctrl_pred: {item['neg_pred'][:80]}  WER={item['neg_vs_ref_audio_wer']:.2f}")

    write_jsonl(
        Path("/ckpts/manifests/fc5_eval_items.jsonl"),
        eval_report["items"],
    )
    Path("/ckpts/manifests/fc5_split_audit.json").write_text(
        json.dumps(split_audit, indent=2)
    )
    Path("/ckpts/manifests/fc5_eval_report.json").write_text(
        json.dumps({k: v for k, v in eval_report.items() if k != "items"}, indent=2)
    )

    val_wer = float(eval_report["held_window_asr_wer"])
    t_total = time.time() - t0_total

    print(f"\n---")
    print(f"held_window_asr_wer: {val_wer:.6f}")
    print(f"n_held:  {len(fc5_held)}")
    print(f"n_train: {len(fc5_train)}")
    print(f"total_seconds: {t_total:.1f}")
    print(f"hidden:  {HIDDEN_DIM}")
    print(f"n_layers: {N_LAYERS}")
    print(f"pretrain_epochs: {PRETRAIN_EPOCHS}")
    print(f"finetune_epochs: {FINETUNE_EPOCHS}")
    print(f"freeze_warmup_epochs: {FREEZE_WARMUP_EPOCHS}")
    print(f"freeze_encoder: {FREEZE_ENCODER}")
    print(f"subject_limit: {SUBJECT_LIMIT}")
    print(f"alice_pretrained: {alice_pretrained}")
    print(f"use_alice_pretrain: {USE_ALICE_PRETRAIN}")
    print(f"held_vs_script_wer: {eval_report['held_vs_script_wer']:.6f}")
    print(f"ref_audio_vs_script_wer: {eval_report['ref_audio_vs_script_wer']:.6f}")
    print(
        "negative_control_shuffled_eeg_vs_ref_audio_wer: "
        f"{eval_report['negative_control_shuffled_eeg_vs_ref_audio_wer']:.6f}"
    )

    return val_wer


@app.local_entrypoint()
def main():
    result = train_and_eval.remote()
    print(f"\n[local] Final held_window_asr_wer = {result:.4f}")
