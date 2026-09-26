"""PatchTST classifier adapter for the Cell HDF5 cache.

The tsai model is a forecasting backbone. We keep its channel/temporal output,
pool over the predicted sequence, and attach a classifier for bacteria labels.
"""
import sys
from pathlib import Path

import torch
from torch import nn


def import_patchtst(tsai_root=None):
    if tsai_root:
        root = str(Path(tsai_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    from tsai.models.PatchTST import PatchTST
    return PatchTST


class PatchTSTClassifier(nn.Module):
    def __init__(self, n_classes, n_points, tsai_root=None, patch_len=32,
                 stride=16, n_layers=3, n_heads=8, d_model=128, d_ff=256,
                 dropout=0.1, revin=True, decomposition=False):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        PatchTST = import_patchtst(tsai_root)
        self.n_points = n_points
        self.backbone = PatchTST(
            c_in=3, c_out=3, seq_len=n_points, pred_dim=1,
            n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff,
            dropout=dropout, attn_dropout=dropout / 2, patch_len=patch_len,
            stride=stride, revin=revin, decomposition=decomposition,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3), nn.Linear(3, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, self.n_points):
            raise ValueError(f"需要 [N, 3, {self.n_points}] 输入，收到 {tuple(x.shape)}")
        z = self.backbone(x)
        # pred_dim=1 turns PatchTST's channel-wise forecasting head into a
        # learned channel representation, avoiding its large seq_len-wide head.
        z = z.squeeze(-1)
        return self.head(z)
