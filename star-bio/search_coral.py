"""Unlabeled partition CORAL/standardization probe for batch-shift diagnosis."""
import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
import torch

from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv


def rbf_head(train, labels, query, centers, gamma, alpha):
    n_cls = len(np.unique(labels))
    tr = torch.as_tensor(train, device="cuda", dtype=torch.float32)
    q = torch.as_tensor(query, device="cuda", dtype=torch.float32)
    c = torch.as_tensor(centers, device="cuda", dtype=torch.float32)
    phi = torch.exp(-gamma * torch.cdist(tr, c).square() / (torch.cdist(c[:256], c[:256]).square().median().clamp_min(1e-6)))
    qp = torch.exp(-gamma * torch.cdist(q, c).square() / (torch.cdist(c[:256], c[:256]).square().median().clamp_min(1e-6)))
    phi = torch.cat([phi, torch.ones(len(phi), 1, device="cuda")], 1)
    qp = torch.cat([qp, torch.ones(len(qp), 1, device="cuda")], 1)
    y = torch.nn.functional.one_hot(torch.as_tensor(labels, device="cuda"), n_cls).float()
    cov, rhs = phi.T @ phi, phi.T @ y
    coef = torch.linalg.solve(cov + alpha * cov.diag().mean() * torch.eye(cov.shape[0], device="cuda"), rhs)
    return (qp @ coef).argmax(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["partition", "global"], default="partition")
    a = p.parse_args()
    meta = load_meta("/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5", True,
                     "/home/wjx/CodeData/data/Star-Com/7class-4patch")
    with h5py.File('/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5', "r") as h:
        x = h["x"][:, 0, :].astype("float32")
    labels = meta["labels"]; rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fold in meta["batch_values"]:
        idx, val_batch = make_splits(meta, fold)
        train = x[idx["train"]]; val = x[idx["val"]]; test = x[idx["test"]]
        mean, std = train.mean(0), train.std(0) + 1e-5
        train_z = (train - mean) / std
        if a.mode == "partition":
            val_z = (val - val.mean(0)) / (val.std(0) + 1e-5)
            test_z = (test - test.mean(0)) / (test.std(0) + 1e-5)
        else:
            val_z = (val - mean) / std; test_z = (test - mean) / std
        rng = np.random.default_rng(3407 + fold)
        centers = train_z[rng.choice(len(train_z), min(4096, len(train_z)), replace=False)]
        best = (-1, None, None); train_y = meta["y"][idx["train"]]
        for gamma in (.03, .1, .3, 1.):
            for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
                pred = rbf_head(train_z, train_y, val_z, centers, gamma, alpha)
                m = classification_metrics(meta["y"][idx["val"]], pred, labels)
                if m["macro_f1"] > best[0]: best = (m["macro_f1"], gamma, alpha)
        pred = rbf_head(train_z, train_y, test_z, centers, best[1], best[2])
        m = classification_metrics(meta["y"][idx["test"]], pred, labels)
        m.update(test_batch=fold, val_batch=val_batch, best_val_f1=best[0], mode=a.mode,
                 gamma=best[1], alpha=best[2])
        rows.append(m); print(f"{a.mode} fold={fold} f1={m['macro_f1']:.6f}", flush=True)
    summary={"mode":a.mode,"folds":rows,"AVG":{k:float(np.mean([r[k] for r in rows])) for k in ("acc","bacc","macro_f1")}}
    (out/f"summary_{a.mode}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARY",summary["AVG"],flush=True)


if __name__ == "__main__": main()
