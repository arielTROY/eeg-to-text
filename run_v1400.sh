#!/usr/bin/env bash
# run_v1400.sh — Full V1400 bootstrap + training on GPUHub
# Usage: bash run_v1400.sh
# Runs setup, Stage 1 training (25 epochs), then Stage 2 (40 epochs).
# All output tee'd to /tmp/v1400_train.log — paste that log back to Claude for monitoring.

set -e
LOG=/tmp/v1400_train.log
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo " V1400 EEG-to-Text Bootstrap — $(date)"
echo "======================================================"
echo ""

# ── GPU check ─────────────────────────────────────────────
echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: no nvidia-smi"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null || echo "torch not yet installed"
echo ""

# ── Repo ──────────────────────────────────────────────────
echo "[REPO]"
if [ ! -d "/root/eeg-to-text" ]; then
    git clone https://github.com/arielTROY/eeg-to-text.git /root/eeg-to-text
fi
cd /root/eeg-to-text
git fetch origin claude/continue-project-s71yR
git checkout claude/continue-project-s71yR
git pull origin claude/continue-project-s71yR
echo "Branch: $(git branch --show-current)  Commit: $(git rev-parse --short HEAD)"
echo ""

# ── Python deps ───────────────────────────────────────────
echo "[DEPS]"
pip install --quiet --upgrade pip
pip install --quiet "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=0.30" \
    "sentence-transformers>=2.6" "peft>=0.13" \
    mne scipy h5py einops jiwer editdistance vaderSentiment \
    osfclient numpy pandas "rouge-score>=0.1" nltk
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
echo "Deps installed."
echo ""

# ── Data ──────────────────────────────────────────────────
echo "[DATA]"
mkdir -p /root/eeg-to-text/data
cd /root/eeg-to-text

V1_COUNT=$(find ./data/ZuCo_v1 -name '*.mat' 2>/dev/null | wc -l)
V2_COUNT=$(find ./data/ZuCo_v2 -name '*.mat' 2>/dev/null | wc -l)

if [ "$V1_COUNT" -lt 5 ]; then
    echo "Downloading ZuCo v1..."
    osf -p q3zws clone ./data/ZuCo_v1 || echo "ZuCo v1 download failed — check osf"
else
    echo "ZuCo v1: $V1_COUNT .mat files already present"
fi

if [ "$V2_COUNT" -lt 5 ]; then
    echo "Downloading ZuCo v2..."
    osf -p 2urht clone ./data/ZuCo_v2 || echo "ZuCo v2 download failed — check osf"
else
    echo "ZuCo v2: $V2_COUNT .mat files already present"
fi
echo ""

# ── Models ────────────────────────────────────────────────
echo "[MODELS]"
python3 - <<'EOF'
print("Downloading VideoMAE-base...")
from transformers import VideoMAEModel
VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
print("OK VideoMAE")

print("Downloading Flan-T5-base...")
from transformers import T5ForConditionalGeneration, T5Tokenizer
T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
T5Tokenizer.from_pretrained("google/flan-t5-base")
print("OK T5")

print("Downloading SBERT...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("OK SBERT")
EOF
echo ""

# ── Dry run ───────────────────────────────────────────────
echo "[DRY RUN — Stage 1, 5 steps]"
cd /root/eeg-to-text
python3 eeg_vlm_v1400.py --stage 1 --dry-run
echo ""

echo "[DRY RUN — Stage 2, 5 steps]"
python3 eeg_vlm_v1400.py --stage 2 --dry-run
echo ""

# ── Stage 1 training ─────────────────────────────────────
echo "======================================================"
echo " STAGE 1: VideoMAE Discriminative Pretraining"
echo " Target: diversity >= 65%  (V1000 baseline: 51.7%)"
echo " Epochs: 25"
echo "======================================================"
python3 eeg_vlm_v1400.py --stage 1 --epochs 25
echo ""

# ── Stage 2 training ─────────────────────────────────────
echo "======================================================"
echo " STAGE 2: Q-Former + T5 Text Generation"
echo " Target: CR >= 25%  (V1200 baseline: 11.2%)"
echo " Epochs: 40"
echo "======================================================"
python3 eeg_vlm_v1400.py --stage 2 --epochs 40
echo ""

# ── Evaluation ────────────────────────────────────────────
echo "======================================================"
echo " EVALUATION"
echo "======================================================"
python3 eval_v1400.py --ckpt ./ckpts/v1400/s2_best.pt
echo ""

echo "======================================================"
echo " ALL DONE — $(date)"
echo " Full log: $LOG"
echo "======================================================"
