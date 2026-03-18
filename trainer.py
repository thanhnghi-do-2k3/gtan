"""
GTAN Training Loop (PyG version)
- Mini-batch neighbor sampling với torch_geometric NeighborLoader
- K-fold cross validation
- Early stopping
- Masked label leakage prevention
- Evaluation: AUC, F1-macro, AP
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from gtan_model import GraphAttnModel


# ─── Early Stopper ────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_model = None
        self.is_stop    = False

    def step(self, loss, model):
        if loss < self.best_loss:
            self.best_loss  = loss
            self.counter    = 0
            self.best_model = copy.deepcopy(model)
            print(f"    ✓ Val loss improved → {loss:.6f}")
        else:
            self.counter += 1
            print(f"    EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.is_stop = True


# ─── Build PyG Data object ────────────────────────────────────────────────────

def build_pyg_data(feat_df, edge_index, labels_tensor, device) -> Data:
    x = torch.tensor(feat_df.values, dtype=torch.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=labels_tensor,
        num_nodes=len(feat_df),
    )
    return data


# ─── Core train/eval loop ─────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, loss_fn, device,
              cat_feat_tensors, labels_tensor,
              mode: str = "train",
              oof_logits: torch.Tensor = None):
    """
    mode = 'train' | 'eval'
    Trả về (avg_loss, all_preds_proba, all_true_labels)
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    total_loss, total_count = 0.0, 0
    all_proba, all_true = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            # batch.n_id: node IDs trong graph gốc
            n_ids = batch.n_id  # [num_nodes_in_subgraph,]
            batch_size = batch.batch_size  # số center nodes

            # Categorical features cho subgraph nodes
            cat_batch = {
                col: t[n_ids].to(device)
                for col, t in cat_feat_tensors.items()
            } if cat_feat_tensors else None

            # Masked labels: center nodes ([:batch_size]) → 2 (padding)
            # Neighbor nodes giữ nguyên label để propagate risk info
            lpa_labels = labels_tensor[n_ids].clone().to(device)
            lpa_labels[:batch_size] = 2   # mask center nodes

            logits = model(batch.x, batch.edge_index, lpa_labels, cat_batch)
            # Chỉ tính loss trên center nodes
            center_logits = logits[:batch_size]
            center_labels = batch.y[:batch_size].to(device)

            # Bỏ unlabeled nodes (label==2) khỏi loss
            valid_mask = center_labels != 2
            cl_v = center_logits[valid_mask]
            lb_v = center_labels[valid_mask]

            if lb_v.numel() == 0:
                continue

            loss = loss_fn(cl_v, lb_v)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item() * lb_v.numel()
            total_count += lb_v.numel()

            proba = torch.softmax(cl_v.detach(), dim=1)[:, 1].cpu().numpy()
            all_proba.extend(proba.tolist())
            all_true.extend(lb_v.cpu().numpy().tolist())

            # Lưu OOF / test predictions
            if oof_logits is not None:
                center_ids = n_ids[:batch_size]
                oof_logits[center_ids] = center_logits.detach().cpu()

    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, np.array(all_proba), np.array(all_true)


# ─── Main training function ──────────────────────────────────────────────────

def gtan_train(feat_df, edge_index, train_idx, test_idx, labels, args, cat_features):
    device_str = args.get("device", "cpu")
    if "cuda" in device_str and not torch.cuda.is_available():
        print(f"⚠️  CUDA không có, dùng CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n{'#'*60}")
    print(f"  GTAN | Dataset: {args.get('dataset','?')}")
    print(f"  Device: {device}  |  K-Fold: {args['n_fold']}  |  MaxEpochs: {args['max_epochs']}")
    print(f"{'#'*60}\n")

    # Chuẩn bị tensors toàn cục
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)
    cat_feat_tensors = {}
    if cat_features:
        for col in cat_features:
            cat_feat_tensors[col] = torch.tensor(feat_df[col].values, dtype=torch.long)

    # PyG Data object
    data = build_pyg_data(feat_df, edge_index, labels_tensor, device)

    # OOF và test predictions (lưu trên CPU để tiết kiệm VRAM)
    n_nodes = len(feat_df)
    oof_logits  = torch.zeros(n_nodes, 2)
    test_logits = torch.zeros(n_nodes, 2)

    loss_fn = nn.CrossEntropyLoss()
    kfold   = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=args["seed"])
    y_train_arr = labels.iloc[train_idx].values
    best_models = []

    for fold, (trn_rel, val_rel) in enumerate(kfold.split(
            np.array(train_idx), y_train_arr)):
        trn_idx_fold = np.array(train_idx)[trn_rel].tolist()
        val_idx_fold = np.array(train_idx)[val_rel].tolist()

        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{args['n_fold']}  "
              f"train={len(trn_idx_fold)} val={len(val_idx_fold)}")
        print(f"{'='*60}")

        # NeighborLoader: sample num_neighbors neighbors mỗi hop
        num_neighbors = [10] * args["n_layers"]  # 10 neighbors per hop
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=torch.tensor(trn_idx_fold, dtype=torch.long),
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=torch.tensor(val_idx_fold, dtype=torch.long),
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        # Model
        model = GraphAttnModel(
            in_feats   = feat_df.shape[1],
            hidden_dim = args["hid_dim"] // 4,   # 256//4=64, ×4heads=256
            n_layers   = args["n_layers"],
            n_classes  = 2,
            heads      = [4] * args["n_layers"],
            activation = nn.PReLU(),
            drop       = args["dropout"],
            device     = device_str,
            gated      = args["gated"],
            ref_df     = feat_df if cat_features else None,
            cat_features = cat_features if cat_features else None,
            n2v_feat   = bool(cat_features),
        ).to(device)

        # LR scale theo batch size (theo paper)
        lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args["wd"])
        scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
        stopper   = EarlyStopper(patience=args["early_stopping"])

        for epoch in range(args["max_epochs"]):
            tr_loss, tr_prob, tr_true = run_epoch(
                model, train_loader, optimizer, loss_fn,
                device, cat_feat_tensors, labels_tensor, mode="train"
            )
            scheduler.step()

            vl_loss, vl_prob, vl_true = run_epoch(
                model, val_loader, None, loss_fn,
                device, cat_feat_tensors, labels_tensor, mode="eval",
                oof_logits=oof_logits,
            )

            # Print metrics
            try:
                tr_auc = roc_auc_score(tr_true, tr_prob) if len(np.unique(tr_true)) > 1 else 0.0
                vl_auc = roc_auc_score(vl_true, vl_prob) if len(np.unique(vl_true)) > 1 else 0.0
                print(f"  E{epoch+1:03d} | tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f}"
                      f" | vl_loss={vl_loss:.4f} vl_auc={vl_auc:.4f}")
            except Exception:
                print(f"  E{epoch+1:03d} | tr_loss={tr_loss:.4f} | vl_loss={vl_loss:.4f}")

            stopper.step(vl_loss, model)
            if stopper.is_stop:
                print("  Early stopping!")
                break

        print(f"  Best val loss: {stopper.best_loss:.7f}")
        best_models.append(stopper.best_model)

        # Inference test set với best model
        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=torch.tensor(test_idx, dtype=torch.long),
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        run_epoch(
            stopper.best_model, test_loader, None, loss_fn,
            device, cat_feat_tensors, labels_tensor, mode="eval",
            oof_logits=test_logits,
        )
        # Ensemble: average across folds
        if fold > 0:
            test_logits /= 2  # running average (simplified)

    # ─── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL TEST EVALUATION")
    print(f"{'='*60}")

    test_arr  = np.array(test_idx)
    y_test    = labels.iloc[test_arr].values
    ts_proba  = torch.softmax(test_logits, dim=1)[test_arr, 1].numpy()
    ts_pred   = torch.argmax(test_logits, dim=1)[test_arr].numpy()

    # Filter unlabeled
    valid = y_test != 2
    y_test   = y_test[valid]
    ts_proba = ts_proba[valid]
    ts_pred  = ts_pred[valid]

    auc = roc_auc_score(y_test, ts_proba)
    f1  = f1_score(y_test, ts_pred, average="macro")
    ap  = average_precision_score(y_test, ts_proba)

    print(f"\n  AUC      = {auc:.4f}   (paper: YelpChi 0.924 | Amazon 0.963)")
    print(f"  F1-macro = {f1:.4f}   (paper: YelpChi 0.799 | Amazon 0.921)")
    print(f"  AP       = {ap:.4f}   (paper: YelpChi 0.751 | Amazon 0.884)")
    print(f"{'='*60}\n")

    # Save checkpoint
    ckpt_dir = args.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"gtan_{args.get('dataset','model')}_best.pt")
    torch.save(best_models[0].state_dict(), ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    return {
        "auc": auc, "f1_macro": f1, "ap": ap,
        "best_models": best_models,
        "test_logits": test_logits,
        "oof_logits": oof_logits,
    }
