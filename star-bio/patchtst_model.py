"""PatchTST classifier adapter for the Cell HDF5 cache."""
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
                 dropout=0.1, revin=True, decomposition=False,
                 pooling="mean", pool_segments=4):
        super().__init__()
        if n_heads < 1 or d_model < 1 or d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if pooling not in ("mean", "segments", "attention"):
            raise ValueError("pooling 必须为 mean、segments 或 attention")
        if not 1 <= patch_len <= n_points or stride < 1 or pool_segments < 1:
            raise ValueError("patch_len、stride 和 pool_segments 无效")
        PatchTST = import_patchtst(tsai_root)
        self.n_points = n_points
        # Reuse tsai's patching and Transformer encoder. The forecasting head
        # is omitted because pred_dim=1 would compress all patches to scalars.
        forecast_model = PatchTST(
            c_in=3, c_out=3, seq_len=n_points, pred_dim=1,
            n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff,
            dropout=dropout, attn_dropout=dropout / 2, patch_len=patch_len,
            stride=stride, revin=revin, decomposition=decomposition,
        )
        if decomposition:
            raise ValueError("classification adapter currently requires decomposition=False")
        core = forecast_model.model
        self.revin_layer = core.revin_layer
        self.use_revin = revin
        if not revin or not self.revin_layer.affine:
            self.revin_layer.requires_grad_(False)
        self.padding_patch_layer = core.padding_patch_layer
        self.unfold = core.unfold
        self.encoder = core.backbone
        self.patch_len = patch_len
        self.patch_num = core.patch_num
        self.pooling = pooling
        self.pool_segments = pool_segments
        if pooling == "segments" and pool_segments > self.patch_num:
            raise ValueError("pool_segments 不能超过 patch 数量")
        if pooling == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, max(1, d_model // 2)), nn.Tanh(),
                nn.Linear(max(1, d_model // 2), 1),
            )
        feature_dim = 3 * d_model * (pool_segments if pooling == "segments" else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes)
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, self.n_points):
            raise ValueError(f"需要 [N, 3, {self.n_points}] 输入，收到 {tuple(x.shape)}")
        z = self.encode(x)
        return self.head(z)

    def encode_patches(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, self.n_points):
            raise ValueError(f"需要 [N, 3, {self.n_points}] 输入，收到 {tuple(x.shape)}")
        z = self.revin_layer(x, torch.tensor(True, dtype=torch.bool, device=x.device)) if self.use_revin else x
        z = self.padding_patch_layer(z)
        b, c, s = z.size()
        z = z.reshape(-1, 1, 1, s)
        z = self.unfold(z)
        z = z.permute(0, 2, 1).reshape(b, c, -1, self.patch_len).permute(0, 1, 3, 2)
        z = self.encoder(z)  # [N, channels, d_model, patches]
        return z

    def encode(self, x):
        z = self.encode_patches(x)
        b = z.shape[0]
        if self.pooling == "segments":
            # Preserve the order of four non-overlapping spectral regions.
            z = torch.stack([part.mean(-1) for part in z.tensor_split(self.pool_segments, dim=-1)], dim=-1)
        elif self.pooling == "attention":
            weights = self.attention_pool(z.transpose(-1, -2)).squeeze(-1).softmax(-1)
            z = (z * weights.unsqueeze(-2)).sum(-1)
        else:
            z = z.mean(dim=-1)
        z = z.reshape(b, -1)
        return z
