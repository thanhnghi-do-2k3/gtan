"""
Data loading & preprocessing cho GTAN (PyG version)
Hỗ trợ 3 dataset:
  - yelp   : YelpChi (fraud review detection)
  - amazon : Amazon  (fraud user detection)
  - sffsd  : S-FFSD  (credit card semi-supervised)
"""

import os
import io
import pickle
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


# ─── Download helpers ─────────────────────────────────────────────────────────

def _download(url: str, dest: str):
    """Download file nếu chưa có."""
    if os.path.exists(dest):
        return
    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  ✓ Saved: {dest}")
    except Exception as e:
        raise RuntimeError(
            f"Download thất bại: {url}\nLỗi: {e}\n"
            f"Tải thủ công về đặt vào: {dest}"
        )


def _download_zip_and_extract(url: str, fname: str, dest_dir: str):
    """Download file .zip rồi extract file fname ra dest_dir."""
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        return
    zip_path = dest + ".zip"
    print(f"  Downloading {os.path.basename(zip_path)} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        os.remove(zip_path)
        print(f"  ✓ Extracted: {dest}")
    except Exception as e:
        raise RuntimeError(
            f"Download/extract thất bại: {url}\nLỗi: {e}\n"
            f"Tải thủ công về đặt {fname} vào: {dest_dir}"
        )


# ─── Tự generate adjlist từ .mat (không cần download pickle) ─────────────────

def _sparse_to_homo_adjlist(sp_matrix) -> dict:
    """Convert sparse adjacency matrix → homo adjlist dict."""
    adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    rows, cols = adj.nonzero()
    for r, c in zip(rows, cols):
        adj_lists[int(r)].add(int(c))
        adj_lists[int(c)].add(int(r))
    return dict(adj_lists)


def _get_or_build_adjlist(mat_data, key: str, pickle_path: str) -> dict:
    """Load pickle nếu có, không thì build từ mat và cache lại."""
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    print(f"  Building adjlist từ '{key}' ...")
    homo = _sparse_to_homo_adjlist(mat_data[key])
    with open(pickle_path, "wb") as f:
        pickle.dump(homo, f)
    print(f"  ✓ Cached: {pickle_path}")
    return homo


# ─── Edge index helpers ───────────────────────────────────────────────────────

def _adj_to_edge_index(homo: dict) -> torch.Tensor:
    src, tgt = [], []
    for i, neighbors in homo.items():
        for j in neighbors:
            src.append(i)
            tgt.append(j)
    return torch.tensor([src, tgt], dtype=torch.long)


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    sl = torch.arange(num_nodes, dtype=torch.long)
    sl = sl.unsqueeze(0).expand(2, -1)
    return torch.cat([edge_index, sl], dim=1)


def _sffsd_edge_index(data: pd.DataFrame, edge_per_trans: int = 3) -> torch.Tensor:
    pair_cols = ["Source", "Target", "Location", "Type"]
    alls, allt = [], []
    for col in pair_cols:
        for _, c_df in data.groupby(col):
            c_df = c_df.sort_values("Time")
            idxs = list(c_df.index)
            n = len(idxs)
            for i in range(n):
                for j in range(1, edge_per_trans + 1):
                    if i + j < n:
                        alls.append(idxs[i])
                        allt.append(idxs[i + j])
    return torch.tensor([alls, allt], dtype=torch.long)


# ─── Dataset URLs ─────────────────────────────────────────────────────────────

# CARE-GNN repo lưu dưới dạng .zip
CARE_BASE = "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/"
ANTIFRAUD_BASE = "https://raw.githubusercontent.com/AI4Risk/antifraud/main/data/"


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_yelp(data_dir: str, test_size: float = 0.4):
    mat_path    = os.path.join(data_dir, "YelpChi.mat")
    pickle_path = os.path.join(data_dir, "yelp_homo_adjlists.pickle")

    # Download YelpChi.zip → extract YelpChi.mat
    _download_zip_and_extract(CARE_BASE + "YelpChi.zip", "YelpChi.mat", data_dir)

    mat    = loadmat(mat_path)
    labels = pd.Series(mat["label"].flatten().astype(int))
    feat_df = pd.DataFrame(mat["features"].todense().A)

    # Build hoặc load homo adjlist
    homo = _get_or_build_adjlist(mat, "homo", pickle_path)

    edge_index = _adj_to_edge_index(homo)
    edge_index = _add_self_loops(edge_index, len(labels))

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, stratify=labels, test_size=test_size, random_state=2, shuffle=True
    )
    return feat_df, labels, train_idx, test_idx, edge_index, []


def load_amazon(data_dir: str, test_size: float = 0.4):
    mat_path    = os.path.join(data_dir, "Amazon.mat")
    pickle_path = os.path.join(data_dir, "amz_homo_adjlists.pickle")

    _download_zip_and_extract(CARE_BASE + "Amazon.zip", "Amazon.mat", data_dir)

    mat    = loadmat(mat_path)
    labels = pd.Series(mat["label"].flatten().astype(int))
    feat_df = pd.DataFrame(mat["features"].todense().A)

    homo = _get_or_build_adjlist(mat, "homo", pickle_path)

    edge_index = _adj_to_edge_index(homo)
    edge_index = _add_self_loops(edge_index, len(labels))

    # Paper: skip first 3305 nodes
    idx = list(range(3305, len(labels)))
    train_idx, test_idx = train_test_split(
        idx, stratify=labels[3305:], test_size=test_size, random_state=2, shuffle=True
    )
    return feat_df, labels, train_idx, test_idx, edge_index, []


def load_sffsd(data_dir: str, test_size: float = 0.4):
    csv_path = os.path.join(data_dir, "S-FFSDneofull.csv")
    _download(ANTIFRAUD_BASE + "S-FFSDneofull.csv", csv_path)

    df   = pd.read_csv(csv_path)
    df   = df.loc[:, ~df.columns.str.contains("Unnamed")]
    data = df[df["Labels"] <= 2].reset_index(drop=True)

    for col in ["Source", "Target", "Location", "Type"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    cat_features = ["Target", "Location", "Type"]
    labels   = data["Labels"]
    feat_df  = data.drop("Labels", axis=1)

    print("  Building S-FFSD graph edges (vài phút)...")
    edge_index = _sffsd_edge_index(data)
    edge_index = _add_self_loops(edge_index, len(data))

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, stratify=labels, test_size=test_size / 2,
        random_state=2, shuffle=True
    )
    return feat_df, labels, train_idx, test_idx, edge_index, cat_features


def load_data(dataset: str, data_dir: str, test_size: float = None):
    os.makedirs(data_dir, exist_ok=True)
    if dataset == "yelp":
        return load_yelp(data_dir, test_size or 0.4)
    elif dataset == "amazon":
        return load_amazon(data_dir, test_size or 0.4)
    elif dataset == "sffsd":
        return load_sffsd(data_dir, test_size or 0.4)
    else:
        raise ValueError(f"Dataset không hợp lệ: {dataset}. Chọn: yelp | amazon | sffsd")


# ─── Mini-batch: masked label leakage prevention ─────────────────────────────

def prepare_batch(feat_df, cat_feat_tensors, labels_tensor,
                  batch_idx, device):
    x_batch = torch.tensor(
        feat_df.iloc[batch_idx].values, dtype=torch.float32
    ).to(device)
    y_batch = labels_tensor[batch_idx].to(device)

    # Center nodes bị mask = 2 (padding_idx) để tránh label leakage
    lpa_labels = labels_tensor[batch_idx].clone().to(device)
    lpa_labels[:] = 2

    cat_batch = {
        col: t[batch_idx].to(device)
        for col, t in cat_feat_tensors.items()
    } if cat_feat_tensors else None

    return x_batch, cat_batch, y_batch, lpa_labels
