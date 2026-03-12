#!/usr/bin/env bash
# setup_server.sh — V1400 GPUHub server setup
# Usage: bash setup_server.sh
# Installs all dependencies, downloads ZuCo datasets and model weights.
set -e

echo "=== V1400 EEG-to-Text Server Setup ==="
echo ""

# ── GPU check ──────────────────────────────────────────────────────────────────
echo "[1/5] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  WARNING: nvidia-smi not found. Continuing (CPU mode)."
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo ""

# ── Python packages ────────────────────────────────────────────────────────────
echo "[2/5] Installing Python packages..."
pip install --upgrade pip --quiet

# Core ML stack
pip install "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=0.30" --quiet

# EEG processing
pip install mne scipy h5py --quiet

# Training utilities
pip install "sentence-transformers>=2.6" einops jiwer editdistance vaderSentiment --quiet

# Data utilities
pip install osfclient numpy pandas --quiet

# Evaluation
pip install "rouge-score>=0.1" nltk --quiet

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "  Packages installed."
echo ""

# ── ZuCo dataset ───────────────────────────────────────────────────────────────
echo "[3/5] Downloading ZuCo datasets..."
mkdir -p ./data

if [ -d "./data/ZuCo_v1" ] && [ "$(find ./data/ZuCo_v1 -name '*.mat' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  ZuCo v1 already present ($(find ./data/ZuCo_v1 -name '*.mat' | wc -l) .mat files)."
else
    echo "  Downloading ZuCo v1 (OSF project q3zws)..."
    osf -p q3zws clone ./data/ZuCo_v1 || echo "  WARNING: ZuCo v1 download failed. Re-run or download manually."
fi

if [ -d "./data/ZuCo_v2" ] && [ "$(find ./data/ZuCo_v2 -name '*.mat' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  ZuCo v2 already present ($(find ./data/ZuCo_v2 -name '*.mat' | wc -l) .mat files)."
else
    echo "  Downloading ZuCo v2 (OSF project 2urht)..."
    osf -p 2urht clone ./data/ZuCo_v2 || echo "  WARNING: ZuCo v2 download failed. Re-run or download manually."
fi
echo ""

# ── Model weights ──────────────────────────────────────────────────────────────
echo "[4/5] Downloading/caching model weights..."

python - <<'EOF'
import sys

print("  VideoMAE-base...")
try:
    from transformers import VideoMAEModel, VideoMAEConfig
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    print("  VideoMAE-base: OK")
except Exception as e:
    print(f"  WARNING: VideoMAE download failed: {e}")

print("  Flan-T5-base...")
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    T5Tokenizer.from_pretrained("google/flan-t5-base")
    print("  Flan-T5-base: OK")
except Exception as e:
    print(f"  WARNING: T5 download failed: {e}")

print("  SBERT all-mpnet-base-v2...")
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("  SBERT: OK")
except Exception as e:
    print(f"  WARNING: SBERT download failed: {e}")
EOF
echo ""

# ── Checkpoint directory ───────────────────────────────────────────────────────
echo "[5/5] Creating checkpoint directories..."
mkdir -p ./ckpts/v1400
echo "  ./ckpts/v1400/ ready"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  Stage 1 (25 epochs):  python eeg_vlm_v1400.py --stage 1 --epochs 25"
echo "  Stage 1 dry run:      python eeg_vlm_v1400.py --stage 1 --dry-run"
echo "  Stage 2 (40 epochs):  python eeg_vlm_v1400.py --stage 2 --epochs 40"
echo "  Evaluate:             python eval_v1400.py --ckpt ./ckpts/v1400/s2_best.pt"
echo ""
echo "ZuCo .mat files found:"
find ./data -name "*.mat" 2>/dev/null | wc -l | xargs echo "  Total:"
