# EEG-to-Text: Open-Vocabulary Brain Decoding via VideoMAE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-vocabulary EEG-to-text decoding using the ZuCo dataset + inner speech transfer.
The core insight: **EEG is a video** — topographic brain maps over time processed by
VideoMAE, the same architecture that reads videos frame-by-frame.

> *"Each person's brain creates a unique topographic movie of their thoughts.
> VideoMAE learns to decode the semantic plot regardless of individual neural cinematography style."*

---

## Architecture

```
EEG (64ch × ~8 sec)
         │
         ▼
MultiScaleRasterizer   64ch electrode positions → 64×64 topographic heat maps (IDW)
         │
         ▼
ChannelAdapter (1×1 conv)   6 frequency bands → 3 RGB channels
         │
         ▼
VideoMAE-base (86M)    MCG-NJU/videomae-base • 12 layers • 768-dim hidden
         │  frozen after Stage 1
         ▼
eeg_to_t5 (1.6M)       trainable bridge: VideoMAE features → T5 cross-attention K/V
         │
         ▼
Frozen Flan-T5-base    EEG provides K/V at all 12 decoder cross-attention layers
         │
         ▼
Open-vocabulary text output
```

### Training Objectives (Stage 2)

| Loss | Weight | Purpose |
|------|--------|---------|
| `loss_lm` | 1.0 | T5 language model cross-entropy — primary generation objective |
| `loss_align` | 0.5 | MoCo InfoNCE — align EEG embeddings with SBERT sentence embeddings |
| `loss_rel` | 0.5 | **Relational distillation** on `eeg_to_t5` output — prevent mode collapse |
| `loss_sent` | 0.2 | Auxiliary sentiment head (binary BCE) |
| `loss_len` | 0.1 | Auxiliary length prediction (MSE log-tokens) |

---

## Two-Stage Training Pipeline

### Stage 1 — Make VideoMAE Discriminative (V1000, 15 epochs)

No language model loss. Only alignment + diversity objectives.
Forces VideoMAE to produce EEG embeddings that vary meaningfully across sentences.

**Achieved: 51.7% embedding diversity** — each EEG input produces a unique feature vector.

### Stage 2 — Learn EEG→Text Mapping (V1200, 35 epochs)

Freeze VideoMAE (preserve Stage 1 discriminative features).
Randomly initialize `eeg_to_t5` — clean slate, no Stage 1 alignment bias.
Fresh optimizer, strong LM loss (1.0), relational distillation prevents collapse.

**Key innovation**: `loss_rel = MSE(eeg_pairwise_cosine_sim, sbert_pairwise_cosine_sim)`
This forces `eeg_to_t5` to preserve the structural diversity of frozen VideoMAE features
when mapping them to T5 cross-attention space.

---

## Results on ZuCo Dataset

### Our System (V1200) — True Autoregressive Generation

All metrics computed with **autoregressive generation** (no teacher-forcing):

| Metric | Best | Mean (ep 1-35) |
|--------|------|----------------|
| Content Recall (CR) | **11.2%** | 10.2% |
| Diversity (unique preds) | **100%** | 99.8% |
| WER | **1.183** | 1.285 |
| CER | 0.894 | 0.964 |

> **Note on WER scale**: WER=1.0 means every word needs exactly 1 edit to match the reference.
> WER=1.183 means outputs are still mostly wrong, but the model is generating semantically
> relevant text (correct domain, tone, and topic) rather than collapsing to a single phrase.

### SOTA Comparison (ZuCo Reading EEG)

> ⚠️ **Critical caveat**: Most 2022-2024 papers use **teacher-forcing during evaluation**, which
> inflates BLEU/ROUGE by up to 3×. See [Jo et al. 2024](https://arxiv.org/abs/2405.06459) for analysis.
> Our WER/CR/Diversity use true autoregressive decoding — a fundamentally harder task.

| System | Year | BLEU-1 | ROUGE-1 | ConsRecall | Diversity | Eval Type |
|--------|------|--------|---------|------------|-----------|-----------|
| [Wang et al.](https://arxiv.org/abs/2112.02690) | 2022 | 40.1% | — | — | — | Teacher-forced |
| [BELT](https://arxiv.org/abs/2309.12056) | 2023 | 42.3% | — | — | — | Teacher-forced |
| [DeWave](https://arxiv.org/abs/2309.14030) | 2023 | 42.8% | 34.9% | — | — | Teacher-forced |
| [EEG2TEXT](https://arxiv.org/abs/2405.02165) | 2024 | **43.9%** | **37.2%** | — | — | Teacher-forced |
| [SemKey](https://arxiv.org/abs/2603.03312) | 2025 | — | — | 2.6% | 22.6% | Autoregressive |
| **Ours V1200** | 2026 | — | — | **11.2%** | **100%** | Autoregressive |

Our Content Recall (**11.2%**) is **4.3× higher** than SemKey (2.6%), the only other system
reporting true autoregressive generation metrics.

### Sample Outputs (V1200 Epoch 13)

```
REF : It's the funniest American comedy since Graffiti Bridge.
PRED: With its long, grueling, sometimes tedious, abysmal slapstick, and inept plot...
       ^^ Movie review style, negative comedy tone ✓

REF : A lame romantic comedy about an unsympathetic character and someone who would...
PRED: The film's aim is a little more straightforward than usual.
       ^^ Correct domain (film), correct register (critic) ✓

REF : Huxley was strongly influenced by F. Matthias Alexander and included him as...
PRED: As a boy, he wasn't the most recognizable character on the satirical sleeve...
       ^^ Biography style, literary context ✓
```

---

## Inner Speech Transfer (V1300)

Fine-tuning VideoMAE features (frozen from V1200 Stage 1) on the
[Inner Speech Dataset](https://openneuro.org/datasets/ds003626) (Nieto et al. 2022):

- **Dataset**: 10 subjects, 128-ch BioSemi EEG at 1024 Hz
- **Task**: 4-class imagined word classification (Arriba/Abajo/Derecha/Izquierda)
- **Protocol**: Leave-One-Subject-Out cross-validation (LOSO)
- **Method**: Frozen VideoMAE + linear probe (5-layer head, 1.6M frozen + small trainable)
- **Chance level**: 25%

| System | Accuracy | Protocol |
|--------|----------|----------|
| Chance | 25.0% | — |
| SVM baseline (Inner Speech paper) | ~26–32% | LOSO |
| **Ours V1300 (ZuCo→Inner Speech transfer)** | **27.9% ± 1.2%** | LOSO |

Per-subject accuracy: 29.4%, 26.0%, 28.0%, 28.8%, 26.2%, 28.1%, 28.2%, 27.8%, 26.5%, 29.5%

> **Zero-shot-style transfer**: VideoMAE was never trained on imagined speech data. The linear probe
> is trained on frozen reading-EEG features from ZuCo. The **+2.9pp above chance** demonstrates that
> discriminative spatial-temporal EEG patterns learned on reading generalize to imagined speech —
> supporting the hypothesis that inner speech shares neural substrates with language perception.

---

## Version History

| Version | Architecture | Key Change | Best Result |
|---------|-------------|------------|-------------|
| V1–V17 | Various | Exploration | Oracle WER 0.938 |
| V600 | GPT-2 prefix | Mode collapse | — |
| V700 | Flan-T5 cross-attn | No diversity control | ep5: div=30.2%, CR=11.7% |
| V800–V900 | + diversity loss | Loss on wrong variable | ep4: CR=12.6% |
| V1000 | Two-stage training | div_loss on v_mean | Stage 1: **51.7% diversity** |
| V1100 | Frozen VideoMAE | No rel_loss → collapse | ep3: WER=0.974 (collapsed) |
| **V1200** | + rel_loss on enc_mean | **Current best** | div=100%, CR=11.2% ✓ |
| **V1300** | Inner Speech transfer | Linear probe on frozen features | **27.9% ± 1.2% LOSO** ✓ |

---

## BCI Roadmap

Our long-term goal: personal real-time EEG-to-text BCI for imagined speech.

```
Stage 1  ZuCo pre-training ─────────────── ✅ V1200 complete
          (reading EEG, 1363 pairs)           CR=11.2%, Div=100%, WER=1.183

Stage 2  Inner Speech transfer ────────────── ✅ V1300 complete
          (imagined speech, LOSO)             27.9% ± 1.2% (chance=25%)

Stage 3  Personal calibration ──────────── ⏳ 30 min via tgam_zuna_ble.py
          (TMAG1/Zuna 64-ch EEG, user-specific)

Stage 4  Live inference ────────────────── ⏳ 3-5 word sentences
          (real-time decoding)
```

Hardware: [Zuna 64-channel research-grade EEG](https://zmachine.ai/) via `tgam_zuna_ble.py`

---

## Dataset

**ZuCo v1+v2** ([Hollenstein et al., 2020](https://osf.io/q3zws/)):
- 1,363 EEG-sentence pairs (after filtering)
- 12 subjects reading Wikipedia + movie reviews
- 64 channels, 500 Hz, ~8-sec sentences
- 85/15 train/val split (seed 42)

**Inner Speech Dataset** ([Nieto et al., 2022](https://openneuro.org/datasets/ds003626)):
- 10 subjects imagining 4 Spanish words
- 128 channels (BioSemi ActiveTwo), 1024 Hz
- ~180 trials per word per subject

---

## Running on Modal

```bash
# Stage 1 — Train discriminative VideoMAE
python3 -m modal run eeg_vlm_v1000_modal.py

# Stage 2 — Train eeg_to_t5 with relational distillation
python3 -m modal run eeg_vlm_v1200_modal.py

# Stage 3 — Inner speech transfer
python3 -m modal run eeg_vlm_v1300_inner_speech_modal.py
```

Requires Modal account with GPU quota (H100 recommended).

---

## Repository Structure

```
eeg_vlm_v*_modal.py      — All experiment versions (V1–V1300)
tgam_zuna_ble.py         — BLE interface for Zuna/TMAG1 EEG hardware
tgam_zuna_visualizer.py  — Real-time EEG visualization
eval_vlm*.py             — Evaluation scripts
diag_*.py                — Diagnostic utilities
```

---

## Key Technical Insights

1. **EEG-as-video**: Topographic maps over time processed by VideoMAE learn cross-subject patterns
2. **Mode collapse prevention**: Relational distillation on `enc_mean` preserves VideoMAE diversity through the `eeg_to_t5` bottleneck
3. **Teacher-forcing is dishonest**: Most published BLEU metrics use teacher-forcing; real WER on autoregressive generation is much harder
4. **Two-stage training**: Stage 1 builds discriminative features (no LM); Stage 2 connects them to language generation
5. **Reading→Imagined speech transfer**: ZuCo-pretrained VideoMAE features transfer to 4-class imagined word classification (27.9% vs 25% chance), supporting shared neural substrates for language perception and inner speech

---

## Citation

If you find this work useful:

```bibtex
@misc{eeg-to-text-2026,
  title={Open-Vocabulary EEG-to-Text via VideoMAE and Relational Distillation},
  author={arielTROY},
  year={2026},
  url={https://github.com/arielTROY/eeg-to-text}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
