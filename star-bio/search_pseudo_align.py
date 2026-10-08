"""Unsupervised class-conditional batch alignment for strict LOBO.

A classifier trained on the observed batches produces pseudo-labels for the
unlabeled validation/test batch. Per-pseudo-class moments are then aligned to
the corresponding training-class moments. The held-out labels are used only by
the existing validation score for selecting lambda and kernel settings.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from metadata import load_meta
from metrics import classification_metrics
from search_domain import rbf_scores, row_snv, fast_macro_f1
from splits import make_splits


def class_align(x, pseudo, source_mean, source_std, mode, lam):
    out = x.copy()
    global_mean = x.mean(0)
    for c in range(source_mean.shape[0]):
        mask = pseudo == c
        if mask.sum() < 20:
            continue
        tm = x[mask].mean(0)
        if mode == "meanstd":
            ts = x[mask].std(0) + 1e-5
            out[mask] = (x[mask] - tm) / ts * source_std[c] + source_mean[c]
        else:
            out[mask] = x[mask] - tm + source_mean[c]
        out[mask] = x[mask] + lam * (out[mask] - x[mask])
    # Preserve the train-only global coordinate system used by the RBF head.
    return out


def align_with_iterations(x, initial_scores, source_mean, source_std, mode, lam, iters):
    scores = initial_scores
    aligned = x
    for _ in range(iters):
        pseudo = scores.argmax(1)
        aligned = class_align(x, pseudo, source_mean, source_std, mode, lam)
        scores = None
        yield aligned, pseudo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["mean", "meanstd"], required=True)
    p.add_argument("--centers", type=int, default=4096)
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--feature-path", default="/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy")
    a = p.parse_args()
    meta = load_meta(a.cache, True, "/home/wjx/CodeData/data/Star-Com/7class-4patch")
    with h5py.File(a.cache, "r") as h:
        batch = h["batch"][:].astype(np.int64)
    x = np.asarray(np.load(a.feature_path, mmap_mode="r")[:, 0, :], dtype=np.float32)
    y, labels = meta["y"], meta["labels"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in meta["batch_values"]:
        started = time.perf_counter()
        idx, val_batch = make_splits(meta, fold)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        train = row_snv(x[tr]); val = row_snv(x[va]); test = row_snv(x[te])
        center = train.mean(0); train -= center; val -= center; test -= center
        n_cls = len(labels)
        src_mean = np.stack([train[y[tr] == c].mean(0) for c in range(n_cls)])
        src_std = np.stack([train[y[tr] == c].std(0) + 1e-5 for c in range(n_cls)])
        rng = np.random.default_rng(3407 + fold)
        keys = y[tr] * len(meta["batch_values"]) + batch[tr]
        counts = np.bincount(keys)
        weights = 1.0 / np.maximum(counts[keys], 1)
        centers = train[rng.choice(len(train), min(a.centers, len(train)), replace=False,
                                  p=weights / weights.sum())]
        best = None
        for gamma in (.1, .3, 1.):
            for alpha in (1e-6, 1e-4):
                val0 = rbf_scores(train, y[tr], val, centers, gamma, alpha, weights)
                # Select the adaptation strength on validation labels only.
                for lam in (0., .5, 1.):
                    for iters in (1,):
                        aligned = val
                        scores = val0
                        for _ in range(iters):
                            pseudo = scores.argmax(1)
                            aligned = class_align(val, pseudo, src_mean, src_std, a.mode, lam)
                            scores = rbf_scores(train, y[tr], aligned, centers, gamma, alpha, weights)
                        score = fast_macro_f1(y[va], scores.argmax(1), n_cls)
                        if best is None or score > best[0]:
                            best = (score, gamma, alpha, lam, iters)
        gamma, alpha, lam, iters = best[1:]
        train_scores = rbf_scores(train, y[tr], test, centers, gamma, alpha, weights)
        aligned = test
        scores = train_scores
        for _ in range(iters):
            pseudo = scores.argmax(1)
            aligned = class_align(test, pseudo, src_mean, src_std, a.mode, lam)
            scores = rbf_scores(train, y[tr], aligned, centers, gamma, alpha, weights)
        metrics = classification_metrics(y[te].tolist(), scores.argmax(1).tolist(), labels)
        metrics.update(test_batch=fold, val_batch=val_batch, best_val_f1=best[0],
                       gamma=gamma, alpha=alpha, lambda_=lam, iterations=iters,
                       mode=a.mode, elapsed_seconds=time.perf_counter() - started)
        fd = out / f"fold{fold}"; fd.mkdir(exist_ok=True)
        (fd / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        rows.append(metrics); print(json.dumps({k: v for k, v in metrics.items() if k not in ("cm", "report")}, ensure_ascii=False), flush=True)
    avg = {k: float(np.mean([m[k] for m in rows])) for k in ("acc", "bacc", "macro_f1")}
    (out / "summary.json").write_text(json.dumps({"mode": a.mode, "folds": rows, "AVG": avg}, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps(avg), flush=True)


if __name__ == "__main__": main()
