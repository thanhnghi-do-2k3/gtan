#!/bin/bash
# ============================================================
#  Setup môi trường cho GTAN Fraud Detection
#  Chạy: bash setup.sh
# ============================================================

set -e
echo "=========================================="
echo "  GTAN Setup Script"
echo "=========================================="

# Auto-detect python binary (python3 hoặc python)
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "ERROR: Không tìm thấy python hoặc python3 trong PATH"
    exit 1
fi
PIP="$PY -m pip"
echo "  Dùng: $PY  ($($PY --version 2>&1))"

# Detect PyTorch và CUDA
CUDA_VER=$($PY -c "import torch; print('cu' + torch.version.cuda.replace('.','') if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
TORCH_VER=$($PY -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
echo ""
echo "Detected: PyTorch=${TORCH_VER:-'chưa cài'}  CUDA=$CUDA_VER"

# ── Bước 1: PyTorch ──────────────────────────────────────────
echo ""
echo "[1/4] Kiểm tra PyTorch..."
if $PY -c "import torch" 2>/dev/null; then
    echo "  OK: PyTorch $TORCH_VER (bỏ qua)"
else
    echo "  Cài PyTorch CPU..."
    $PIP install torch --index-url https://download.pytorch.org/whl/cpu
    # Refresh version sau khi cài
    TORCH_VER=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_VER="cpu"
    echo "  OK: PyTorch $TORCH_VER"
fi

# ── Bước 2: PyTorch Geometric ────────────────────────────────
echo ""
echo "[2/4] Cài PyTorch Geometric..."
$PIP install torch_geometric -q
echo "  OK"

# ── Bước 3: PyG extensions ───────────────────────────────────
echo ""
echo "[3/4] Cài PyG extensions (torch-scatter, torch-sparse)..."
PYG_WHEEL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"
echo "  Từ: $PYG_WHEEL"
$PIP install torch-scatter torch-sparse -f "$PYG_WHEEL" -q
echo "  OK"

# ── Bước 4: Các thư viện còn lại ─────────────────────────────
echo ""
echo "[4/4] Cài scikit-learn, pandas, numpy, scipy, tqdm..."
$PIP install scikit-learn pandas numpy scipy tqdm matplotlib requests -q
echo "  OK"

echo ""
echo "=========================================="
echo "  ✓ Setup xong!"
echo ""
echo "  Chạy training (test nhanh, CPU):"
echo "    $PY train.py --dataset yelp --device cpu --epochs 5 --n_fold 2"
echo ""
echo "  Full training (cần GPU):"
echo "    $PY train.py --dataset yelp --device cuda:0 --epochs 100 --n_fold 5"
echo "=========================================="
