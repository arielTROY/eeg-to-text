# EEG-to-Text: Open-Vocabulary Brain Decoding

EEG → open-text decoding using the ZuCo dataset and a VideoMAE-based architecture.
The core insight: **EEG is a video** — topographic brain maps over time are processed by
a Vision Transformer (VideoMAE) trained to decode semantic content.

## Architecture

```
EEG (64ch × 512 timepoints)
    │
    ▼
MultiScaleRasterizer          # 64ch → 64×64 inverse-distance-weighted topographic maps
    │
    ▼
ChannelAdapter (6 bands → RGB) # delta/theta/alpha/beta/gamma/broadband → 3 channels
    │
    ▼
VideoMAE-base (86M)           # MCG-NJU/videomae-base, 12 layers, 768-dim
    │
    ▼  eeg_to_t5 (1.6M)
    ▼
Frozen Flan-T5-base           # EEG as cross-attention K/V at all 12 decoder layers
    │
    ▼
Open-vocabulary text output
```

**Training objectives:**
- `loss_lm` (×1.0): Language model cross-entropy — primary text generation objective
- `loss_align` (×0.5): MoCo InfoNCE — align EEG embeddings with SBERT sentence embeddings
- `loss_rel` (×0.5): Relational distillation on eeg_to_t5 output — prevent mode collapse
- `loss_sent` (×0.2): Auxiliary sentiment prediction
- `loss_len` (×0.1): Auxiliary sequence length prediction

## Training Pipeline

**Stage 1** (V1000, 15 epochs): Train VideoMAE to be discriminative across EEG signals.
No LM loss — only alignment + diversity objectives. Achieved 51.7% diversity.

**Stage 2** (V1200+): Freeze VideoMAE (preserving Stage 1 features). Train fresh `eeg_to_t5`
to map discriminative EEG features → T5 cross-attention conditioning.

## Key Versions

| Version | Architecture | Best Metric |
|---------|-------------|-------------|
| V17 | VideoMAE → Qwen2.5 | Oracle WER 0.938 |
| V700 | Flan-T5 cross-attn | ep5: div=30.2%, CR=11.7% |
| V1000 | Two-stage + div_loss on v_mean | Stage 1: 51.7% diversity |
| V1100 | Frozen VideoMAE + fresh eeg_to_t5 | ep3: WER=0.974 (collapsed) |
| V1200 | + rel_loss on enc_mean | ep7: div=100%, CR=10.8% ← current |

## Dataset

[ZuCo](https://osf.io/q3zws/) (v1 + v2) — ~1,363 EEG-sentence pairs from naturalistic reading.
Wikipedia articles + movie reviews, 400ms word-level EEG recorded at 500Hz, 105 channels.

## Hardware Interface

`tgam_zuna_ble.py` — BLE interface for TMAG1/Zuna research-grade 64-channel EEG.
Used for personal calibration after ZuCo pre-training.

**BCI Roadmap:**
1. ZuCo pre-training (current)
2. Fine-tune on ThinkEEG + Inner Speech (imagined speech)
3. 30-min personal calibration via `tgam_zuna_ble.py`
4. Test on 3-5 word sentences

## Running on Modal

Each `eeg_vlm_v*_modal.py` runs a training experiment on Modal H100:

```bash
python3 -m modal run eeg_vlm_v1200_modal.py
```

Requires a Modal account with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` set.

## Why EEG-as-Video?

Each person's brain creates a unique topographic "movie" of their thoughts.
VideoMAE learns to decode the semantic plot regardless of individual neural cinematography style.
This is the key insight that makes cross-subject transfer feasible.

## Results

Current best (V1200, ep7):
- Diversity: **100%** (no mode collapse)
- Content Recall: **10.8%**
- WER: **1.315** (improving each epoch)

The model correctly captures domain (movie reviews vs. biography) and semantic themes,
even when exact word matches are rare.

## License

MIT
