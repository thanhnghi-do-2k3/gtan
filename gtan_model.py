"""
GTAN Model Architecture (PyTorch Geometric version)
Semi-supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation
Paper: https://arxiv.org/abs/2412.18287

Dùng PyTorch Geometric thay DGL (tương thích Python 3.13+)
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PosEncoding(nn.Module):
    """Positional Encoding cho temporal features"""

    def __init__(self, dim: int, device, base: int = 10000, bias: float = 0.0):
        super().__init__()
        p, sft = [], []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            sft.append(np.pi / 2.0 + bias if i % 2 else bias)
        self.device = device
        self.sft  = torch.tensor(sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p,   dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pos = pos.view(-1, 1).to(self.device)
            return torch.sin(pos / self.base + self.sft)


class TransEmbedding(nn.Module):
    """
    Attribute Embedding & Feature Learning Layer.
    Xử lý categorical attributes: one-hot → Embedding → MLP per table → add-pooling
    """

    def __init__(self, df, device, dropout: float = 0.2,
                 in_feats: int = 82, cat_features: list = None):
        super().__init__()
        if cat_features is None:
            cat_features = []
        self.cat_features = [c for c in cat_features if c not in {"Labels", "Time"}]
        self.cat_table = nn.ModuleDict({
            col: nn.Embedding(int(df[col].max()) + 1, in_feats).to(device)
            for col in self.cat_features
        })
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for _ in self.cat_features]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, cat_dict: dict) -> torch.Tensor:
        output = 0
        for i, col in enumerate(self.cat_features):
            emb = self.cat_table[col](cat_dict[col])
            emb = self.dropout(emb)
            emb = self.forward_mlp[i](emb)
            output = output + emb
        return output


class GTANConv(MessagePassing):
    """
    Gated Temporal Attention Network (GTAN) layer — PyG version.
    Implements scaled dot-product multi-head attention + gated residual.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int,
        skip_feat: bool = True,
        gated: bool = True,
        layer_norm: bool = True,
        activation=None,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add")
        self.in_feats  = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.scale     = (out_feats ** 0.5)

        self.lin_q = nn.Linear(in_feats, out_feats * num_heads)
        self.lin_k = nn.Linear(in_feats, out_feats * num_heads)
        self.lin_v = nn.Linear(in_feats, out_feats * num_heads)

        self.skip = nn.Linear(in_feats, out_feats * num_heads) if skip_feat else None
        self.gate = nn.Linear(3 * out_feats * num_heads, 1) if gated else None
        self.norm = nn.LayerNorm(out_feats * num_heads) if layer_norm else None
        self.act  = activation if activation is not None else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        q = self.lin_q(x)  # [N, H*D]
        k = self.lin_k(x)
        v = self.lin_v(x)
        out = self.propagate(edge_index, q=q, k=k, v=v, x_orig=x)

        if self.skip is not None:
            skip = self.skip(x)
            if self.gate is not None:
                g = torch.sigmoid(self.gate(torch.cat([skip, out, skip - out], dim=-1)))
                out = g * skip + (1 - g) * out
            else:
                out = skip + out

        if self.norm is not None:
            out = self.norm(out)
        return self.act(out)

    def message(self, q_i, k_j, v_j, index, size_i):
        # Scaled dot-product attention
        # q_i: [E, H*D], k_j: [E, H*D]
        E, HD = q_i.shape
        q_i = q_i.view(E, self.num_heads, self.out_feats)
        k_j = k_j.view(E, self.num_heads, self.out_feats)
        v_j = v_j.view(E, self.num_heads, self.out_feats)

        attn = (q_i * k_j).sum(dim=-1, keepdim=True) / self.scale  # [E, H, 1]
        attn = softmax(attn, index, num_nodes=size_i)               # edge-wise softmax
        out  = (attn * v_j).view(E, HD)                             # [E, H*D]
        return out

    def update(self, aggr_out):
        return self.drop(aggr_out)


class GraphAttnModel(nn.Module):
    """
    Full GTAN model:
      1) Attribute Embedding (TransEmbedding) — categorical features
      2) Risk Embedding — label as categorical (masked cho center nodes)
      3) Stack GTANConv layers
      4) MLP classifier
    """

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        n_layers: int,
        n_classes: int,
        heads: list,
        activation=None,
        skip_feat: bool = True,
        gated: bool = True,
        layer_norm: bool = True,
        post_proc: bool = True,
        n2v_feat: bool = True,
        drop: list = None,
        ref_df=None,
        cat_features: list = None,
        device: str = "cpu",
    ):
        super().__init__()
        if drop is None:
            drop = [0.2, 0.1]
        if cat_features is None:
            cat_features = []
        if activation is None:
            activation = nn.PReLU()

        self.n_layers   = n_layers
        self.n_classes  = n_classes
        self.input_drop = nn.Dropout(drop[0])
        self.out_drop   = nn.Dropout(drop[1])

        # Categorical embedding (n2v)
        self.n2v_mlp = None
        if n2v_feat and ref_df is not None and cat_features:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device,
                in_feats=in_feats, cat_features=cat_features
            )

        # Risk (label) embedding: 0=legit, 1=fraud, 2=unlabeled(padding)
        self.label_emb  = nn.Embedding(n_classes + 1, in_feats, padding_idx=n_classes)
        self.proj_num   = nn.Linear(in_feats, hidden_dim * heads[0])
        self.proj_label = nn.Linear(in_feats, hidden_dim * heads[0])
        self.merge_mlp  = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * heads[0]),
            nn.PReLU(),
            nn.Dropout(drop[1]),
            nn.Linear(hidden_dim * heads[0], in_feats),
        )

        # GTAN layers
        self.convs = nn.ModuleList()
        self.convs.append(GTANConv(
            in_feats, hidden_dim, heads[0],
            skip_feat=skip_feat, gated=gated, layer_norm=layer_norm,
            activation=activation, dropout=drop[1],
        ))
        for l in range(1, n_layers):
            self.convs.append(GTANConv(
                hidden_dim * heads[l - 1], hidden_dim, heads[l],
                skip_feat=skip_feat, gated=gated, layer_norm=layer_norm,
                activation=activation, dropout=drop[1],
            ))

        # Final classifier
        final_dim = hidden_dim * heads[-1]
        if post_proc:
            self.clf = nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.BatchNorm1d(final_dim),
                nn.PReLU(),
                nn.Dropout(drop[1]),
                nn.Linear(final_dim, n_classes),
            )
        else:
            self.clf = nn.Linear(final_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        cat_feat: dict = None,
    ) -> torch.Tensor:
        """
        x          : numerical features [N, feat_dim]
        edge_index : [2, E]
        labels     : masked labels [N,] (2 = unlabeled/masked → zero embedding via padding_idx)
        cat_feat   : dict of categorical tensors
        """
        h = x
        if cat_feat is not None and self.n2v_mlp is not None:
            h = h + self.n2v_mlp(cat_feat)

        # Risk embedding fusion
        lbl_emb = self.input_drop(self.label_emb(labels))
        fused   = self.proj_num(h) + self.proj_label(lbl_emb)
        fused   = self.merge_mlp(fused)
        h = h + fused  # residual

        for conv in self.convs:
            h = self.out_drop(conv(h, edge_index))

        return self.clf(h)
