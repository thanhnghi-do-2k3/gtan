# GTAN – Semi-supervised Credit Card Fraud Detection

Implementation PyTorch Geometric của paper:
> **Semi-supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation**
> Sheng Xiang et al. — AAAI 2023 | [arXiv:2412.18287](https://arxiv.org/abs/2412.18287)

---

## Cấu trúc

```
gtan_fraud/
├── gtan_model.py   # Model: PosEncoding, TransEmbedding, GTANConv, GraphAttnModel
├── data_loader.py  # Load YelpChi / Amazon / S-FFSD (tự động download)
├── trainer.py      # Training loop, early stopping, K-fold, evaluation
├── train.py        # Entry point
├── setup.sh        # Cài dependencies
├── requirements.txt
└── data/           # Dataset tự động tải vào đây
```

---

## Cài đặt

```bash
bash setup.sh
```

Hoặc cài thủ công:

```bash
# Bước 1: PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8

# Bước 2: PyTorch Geometric
pip install torch_geometric

# Bước 3: PyG extensions (bắt buộc)
# Thay {TORCH_VER} bằng version thực, ví dụ 2.1.0
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-{TORCH_VER}+cpu.html

# Bước 4: các thư viện khác
pip install scikit-learn pandas numpy scipy tqdm matplotlib
```

---

## Chạy training

### Quick test (CPU, dataset nhỏ)
```bash
python train.py --dataset yelp --device cpu --epochs 5 --n_fold 2
```

### Full training như paper (cần GPU)
```bash
# YelpChi
python train.py --dataset yelp --device cuda:0 --epochs 100 --n_fold 5

# Amazon
python train.py --dataset amazon --device cuda:0 --epochs 100 --n_fold 5

# S-FFSD (semi-supervised, dataset lớn nhất)
python train.py --dataset sffsd --device cuda:0 --epochs 100 --n_fold 5
```

### Tùy chỉnh hyperparams
```bash
python train.py \
  --dataset yelp \
  --epochs 100 \
  --batch_size 256 \
  --lr 0.003 \
  --n_layers 2 \
  --n_fold 5 \
  --hid_dim 256 \
  --device cuda:0

# GTAN-A ablation (không có gated residual):
python train.py --dataset yelp --no_gated
```

---

## Hyperparams (theo paper)

| Param | Value |
|---|---|
| batch_size | 128 |
| hidden_dim | 256 (64×4 heads) |
| learning_rate | 0.003 |
| weight_decay | 1e-4 |
| n_layers | 2 |
| n_heads | 4 |
| dropout | [0.2, 0.1] |
| optimizer | Adam |
| LR scheduler | MultiStepLR [4000, 12000] ×0.3 |
| max_epochs | 100 |
| early_stopping | 10 |
| k_fold | 5 |

---

## Kết quả kỳ vọng (theo paper)

| Dataset | AUC | F1-macro | AP |
|---|---|---|---|
| YelpChi | **0.9241** | 0.7988 | 0.7513 |
| Amazon | **0.9630** | 0.9213 | 0.8838 |
| S-FFSD | **0.7616** | 0.6764 | 0.5767 |

---

## Kiến trúc model

```
Input (transaction records)
  ↓
[Attribute Embedding]         ← categorical features: Target, Location, Type
  one-hot → Embedding → MLP → add-pooling
  ↓
[Risk Embedding Fusion]       ← label as categorical (0/1/2=unlabeled)
  Masked cho center nodes (tránh label leakage)
  ↓
[GTANConv × n_layers]         ← Gated Temporal Attention Network
  Scaled dot-product attention (multi-head)
  Attribute-driven Gated Residual
  ↓
[MLP Classifier]
  2-layer MLP → sigmoid → fraud probability
```

### Key contribution: Masked Label Propagation
- Label của transaction được embed như categorical feature
- Center nodes (nodes đang predict) bị **mask label = 2** (unlabeled)
- Model học từ risk embedding của **neighbor nodes** → tránh label leakage
- Khi inference: dùng toàn bộ observed labels → risk propagation full graph

---

## Datasets

| Dataset | Nodes | Edges | Fraud% | Unlabeled |
|---|---|---|---|---|
| YelpChi | 45,954 | 7.7M | 14.5% | 0 |
| Amazon | 11,948 | 8.8M | 6.9% | 0 |
| S-FFSD | 1,820,840 | 31.6M | 19.0% | 90.3% |

Dataset tự động download từ:
- YelpChi/Amazon: [CARE-GNN repo](https://github.com/YingtongDou/CARE-GNN)
- S-FFSD: [antifraud repo](https://github.com/AI4Risk/antifraud)
