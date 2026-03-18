"""
GTAN Train Script (PyTorch Geometric version)
=============================================
Paper: Semi-supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation
       https://arxiv.org/abs/2412.18287

Usage:
    # Dataset nhẹ nhất để test nhanh (YelpChi):
    python train.py --dataset yelp --device cpu

    # Amazon:
    python train.py --dataset amazon

    # S-FFSD (semi-supervised, giống paper nhất — dataset lớn hơn):
    python train.py --dataset sffsd

    # Full training theo paper (cần GPU):
    python train.py --dataset yelp --epochs 100 --n_fold 5 --device cuda:0

    # Chạy nhanh để test pipeline (CPU, 3 epochs):
    python train.py --dataset yelp --device cpu --epochs 3 --n_fold 2 --batch_size 512
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data
from trainer import gtan_train


# ─── Default hyperparams ──────────────────────────────────────────────────────

DEFAULTS = {
    # === Theo paper ===
    "batch_size"    : 128,
    "hid_dim"       : 256,    # hidden = 256/4=64 × 4heads → 256 dim
    "lr"            : 0.003,
    "wd"            : 1e-4,
    "n_layers"      : 2,
    "dropout"       : [0.2, 0.1],
    "early_stopping": 10,
    "n_fold"        : 5,
    "seed"          : 2023,
    "max_epochs"    : 100,
    "gated"         : True,
    "test_size"     : 0.4,
    "checkpoint_dir": "checkpoints",
    # === Mặc định dùng CPU nếu không có CUDA ===
    "device"        : "cuda:0",
}

DATASET_OVERRIDES = {
    "yelp"  : {"dataset": "yelp"},
    "amazon": {"dataset": "amazon"},
    "sffsd" : {"dataset": "sffsd", "test_size": 0.4},
}


def get_args(dataset: str) -> dict:
    args = dict(DEFAULTS)
    args.update(DATASET_OVERRIDES[dataset])
    return args


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_cli():
    p = argparse.ArgumentParser(description="Train GTAN Fraud Detection")
    p.add_argument("--dataset",    default="yelp",   choices=["yelp","amazon","sffsd"])
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--n_layers",   type=int,   default=None)
    p.add_argument("--n_fold",     type=int,   default=None)
    p.add_argument("--hid_dim",    type=int,   default=None)
    p.add_argument("--device",     type=str,   default=None,
                   help="cpu | cuda:0 | cuda:1 ...")
    p.add_argument("--test_size",  type=float, default=None)
    p.add_argument("--no_gated",   action="store_true",
                   help="Tắt gated residual (GTAN-A ablation variant)")
    return p.parse_args()


def main():
    cli  = parse_cli()
    args = get_args(cli.dataset)

    # Apply CLI overrides
    if cli.epochs     is not None: args["max_epochs"]   = cli.epochs
    if cli.batch_size is not None: args["batch_size"]   = cli.batch_size
    if cli.lr         is not None: args["lr"]           = cli.lr
    if cli.n_layers   is not None: args["n_layers"]     = cli.n_layers
    if cli.n_fold     is not None: args["n_fold"]       = cli.n_fold
    if cli.hid_dim    is not None: args["hid_dim"]      = cli.hid_dim
    if cli.device     is not None: args["device"]       = cli.device
    if cli.test_size  is not None: args["test_size"]    = cli.test_size
    if cli.no_gated:               args["gated"]        = False

    print(f"\n{'─'*50}")
    print(f"  Dataset      : {args['dataset']}")
    print(f"  Epochs       : {args['max_epochs']}")
    print(f"  Batch size   : {args['batch_size']}")
    print(f"  LR           : {args['lr']}")
    print(f"  GTAN layers  : {args['n_layers']}")
    print(f"  K-folds      : {args['n_fold']}")
    print(f"  Hidden dim   : {args['hid_dim']}")
    print(f"  Gated        : {args['gated']}")
    print(f"  Device       : {args['device']}")
    print(f"{'─'*50}\n")

    # Load data (tự động download nếu chưa có)
    print(f"Loading {args['dataset']} dataset...")
    feat_df, labels, train_idx, test_idx, edge_index, cat_features = load_data(
        args["dataset"], cli.data_dir, args.get("test_size")
    )
    print(f"  Nodes        : {len(feat_df):,}")
    print(f"  Edges        : {edge_index.shape[1]:,}")
    print(f"  Features     : {feat_df.shape[1]}")
    print(f"  Train nodes  : {len(train_idx):,}")
    print(f"  Test nodes   : {len(test_idx):,}")
    print(f"  Fraud ratio  : {(labels==1).sum()/len(labels):.2%}")
    if (labels==2).sum() > 0:
        print(f"  Unlabeled    : {(labels==2).sum():,}")
    print(f"  Cat features : {cat_features}\n")

    # Train
    results = gtan_train(
        feat_df, edge_index, train_idx, test_idx,
        labels, args, cat_features
    )

    print(f"\n{'─'*50}")
    print("  DONE!")
    print(f"  AUC      = {results['auc']:.4f}")
    print(f"  F1-macro = {results['f1_macro']:.4f}")
    print(f"  AP       = {results['ap']:.4f}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    # Hỗ trợ cả: python train.py / python3 train.py
    main()
