"""Models for controlled cross-batch experiments; checkpoints store full spec."""
import torch
from torch import nn
from patchtst_model import PatchTSTClassifier


class SearchModel(nn.Module):
    def __init__(self, spec, n_points, n_classes):
        super().__init__()
        self.spec = spec
        kind = spec.get('model', 'patch')
        if kind == 'patch':
            dim = spec.get('dim',64)
            self.net = PatchTSTClassifier(n_classes,n_points,
                spec['tsai_root'],patch_len=spec.get('patch',32),stride=spec.get('stride',16),
                n_layers=spec.get('layers',2),n_heads=4,d_model=dim,d_ff=dim*2,
                dropout=spec.get('dropout',.2),revin=spec.get('revin',True),
                pooling='segments' if spec.get('segments',1)>1 else 'mean',
                pool_segments=spec.get('segments',1))
            if spec.get('norm','layer') == 'layer':
                # Upstream _PatchTST_backbone does not forward its norm argument.
                for block in self.net.encoder.layers:
                    block.norm_attn = nn.LayerNorm(dim)
                    block.norm_ffn = nn.LayerNorm(dim)
        elif kind == 'mlp':
            self.net = nn.Sequential(nn.Flatten(),nn.Linear(3*n_points,256),nn.LayerNorm(256),
                nn.GELU(),nn.Dropout(.3),nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,n_classes))
        elif kind == 'cnn':
            self.net = nn.Sequential(nn.Conv1d(3,32,15,padding=7,stride=2),nn.GroupNorm(8,32),nn.GELU(),
                nn.Conv1d(32,64,9,padding=4,stride=2),nn.GroupNorm(8,64),nn.GELU(),
                nn.Conv1d(64,128,7,padding=3,stride=2),nn.GroupNorm(8,128),nn.GELU(),
                nn.AdaptiveAvgPool1d(16),nn.Flatten(),nn.Dropout(.3),nn.Linear(128*16,256),
                nn.GELU(),nn.Dropout(.2),nn.Linear(256,n_classes))
        else: raise ValueError(kind)

    def forward(self,x):
        if self.spec.get('input_snv',False):
            x=(x-x.mean(-1,keepdim=True))/(x.std(-1,keepdim=True,unbiased=False)+1e-6)
        return self.net(x)
