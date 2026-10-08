"""Self-supervised masked spectral pretraining for external Raman .npy data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from patchtst_model import PatchTSTClassifier


class ExternalSpectra(Dataset):
    def __init__(self, path, points=1105):
        x = np.load(path, mmap_mode="r").astype(np.float32)
        if x.ndim != 2 or x.shape[1] < points:
            raise ValueError(f"外部光谱需要 [N, points] 且 points >= {points}")
        self.x = x[:, :points]

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.x[i], copy=True))
        x = (x - x.mean()) / (x.std(unbiased=False) + 1e-6)
        # Three views: spectrum, first derivative, and SNV-like normalized view.
        d = torch.diff(x, prepend=x[:1])
        return torch.stack([x, d, x])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--external", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tsai-root", default="/home/wjx/CodeData/code/tsai-main")
    p.add_argument("--points", type=int, default=1105)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--mask-fraction", type=float, default=.4)
    p.add_argument("--patch-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=256)
    args = p.parse_args()
    if not 0 < args.mask_fraction < 1: p.error("mask-fraction 必须在 (0,1) 内")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ExternalSpectra(args.external, args.points), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    # Use a classifier adapter without relying on labels; reconstruction targets
    # are the unmasked channel sequences and logits are ignored.
    model = PatchTSTClassifier(3, args.points, args.tsai_root, patch_len=args.patch_len,
                               stride=args.stride, n_layers=args.layers, n_heads=args.heads,
                               d_model=args.d_model, d_ff=args.d_ff).to(device)
    decoder = nn.Sequential(nn.Linear(3 * args.d_model, args.d_ff), nn.GELU(),
                            nn.Linear(args.d_ff, 3 * args.patch_len)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); decoder.train(); total = 0.; count = 0
        for x in tqdm(loader, desc=f"pretrain {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            patches = model.patch_num
            patch_mask = torch.rand(x.shape[0], patches, device=device) < args.mask_fraction
            if not patch_mask.any():
                patch_mask[:, torch.randint(patches, (1,), device=device)] = True
            point_mask = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=device)
            # Match the backbone's left replication padding and patch stride.
            for patch_index in range(patches):
                start = patch_index * args.stride
                left_padded_start = start - args.stride
                a, b = max(0, left_padded_start), min(x.shape[-1], left_padded_start + args.patch_len)
                if b > a:
                    point_mask[:, a:b] |= patch_mask[:, patch_index, None]
            masked = x.masked_fill(point_mask[:, None, :], 0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                z = model.encode_patches(masked).permute(0, 3, 1, 2).reshape(x.shape[0], patches, -1)
                reconstruction = decoder(z).reshape(x.shape[0], patches, 3, args.patch_len)
                padded = nn.functional.pad(x, (args.stride, 0), mode="replicate")
                target = padded.unfold(-1, args.patch_len, args.stride).permute(0, 2, 1, 3)
                mask = patch_mask[:, :, None, None].expand_as(target)
                loss = nn.functional.smooth_l1_loss(reconstruction[mask], target[mask])
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), 1.0); scaler.step(opt); scaler.update()
            total += float(loss) * len(x); count += len(x)
        history.append({"epoch": epoch, "loss": total / max(1, count)})
        print(json.dumps(history[-1]), flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args), "history": history}, out)
    out.with_suffix(".json").write_text(json.dumps({"config": vars(args), "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
