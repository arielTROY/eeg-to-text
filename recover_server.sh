#!/usr/bin/env bash
# recover_server.sh — Run this on the GPUHub server to fix disk-full and resume setup
# Usage: bash recover_server.sh 2>&1 | tee /root/v1400/recover.log
set -euo pipefail

echo "=== [1/7] Disk layout ==="
df -h
echo ""
echo "=== Largest items under /root ==="
du -sh /root/* 2>/dev/null | sort -rh | head -20

echo ""
echo "=== [2/7] Clearing /tmp ==="
rm -rf /tmp/* 2>/dev/null || true
df -h /tmp

echo ""
echo "=== [3/7] Setting TMPDIR ==="
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

echo ""
echo "=== [4/7] Verify torch imports ==="
python3 -c "import torch; print('torch OK:', torch.__version__)"

echo ""
echo "=== [5/7] Check ZuCo v1 completeness ==="
ZUCO_DIR=/root/v1400/data/ZuCo_v1
if [ -d "$ZUCO_DIR" ]; then
    COUNT=$(find "$ZUCO_DIR" -type f | wc -l)
    echo "ZuCo v1 files present: $COUNT"
    du -sh "$ZUCO_DIR"
else
    echo "ZuCo v1 directory missing — will need to re-download"
fi

echo ""
echo "=== [6/7] Download HuggingFace models (TMPDIR=$TMPDIR) ==="
cd /root/v1400
python3 - <<'PYEOF'
import os
os.environ['TMPDIR'] = '/root/tmp'
os.makedirs('/root/tmp', exist_ok=True)
import tempfile
print("tempfile.gettempdir() =", tempfile.gettempdir())

print("\nDownloading VideoMAE-base...")
from transformers import VideoMAEModel
VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')
print("VideoMAE OK")

print("\nDownloading flan-t5-base...")
from transformers import T5ForConditionalGeneration, T5Tokenizer
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
T5Tokenizer.from_pretrained('google/flan-t5-base')
print("T5 OK")

print("\nDownloading SBERT (all-mpnet-base-v2)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("SBERT OK")

print("\n=== All models downloaded ===")
PYEOF

echo ""
echo "=== [7/7] Dry-run Stage 1 (5 steps) ==="
cd /root/v1400
TMPDIR=/root/tmp python3 eeg_vlm_v1400.py --stage 1 --dry-run 2>&1

echo ""
echo "=== Recovery complete. If dry-run passed, launch training: ==="
echo "  cd /root/v1400"
echo "  tmux new -s s1 'TMPDIR=/root/tmp python3 eeg_vlm_v1400.py --stage 1 --epochs 25 2>&1 | tee logs/s1.log'"
