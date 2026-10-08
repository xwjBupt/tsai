"""Compare source-to-target versus target-to-source unsupervised alignment."""
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


def align_source_to_target(train, query, mode):
    sm, ss = train.mean(0), train.std(0) + 1e-5
    qm, qs = query.mean(0), query.std(0) + 1e-5
    if mode == "target_center":
        return train - sm + qm, query
    return (train - sm) / ss * qs + qm, query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["target_center", "target_affine"], required=True)
    p.add_argument("--centers", type=int, default=2048)
    p.add_argument("--feature-path", default="/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy")
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--data-root", default="/home/wjx/CodeData/data/Star-Com/7class-4patch")
    a = p.parse_args()
    meta = load_meta(a.cache, True, a.data_root)
    with h5py.File(a.cache, "r") as h:
        x = np.asarray(np.load(a.feature_path, mmap_mode="r")[:, 0, :], dtype=np.float32)
    y, labels, batch = meta["y"], meta["labels"], meta["batch"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for fold in meta["batch_values"]:
        t0 = time.perf_counter(); idx, val_batch = make_splits(meta, fold)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        train0 = row_snv(x[tr]); val0 = row_snv(x[va]); test0 = row_snv(x[te])
        # Feature center is fitted on the train samples only before alignment.
        global_mean = train0.mean(0)
        train0 -= global_mean; val0 -= global_mean; test0 -= global_mean
        keys = y[tr] * len(meta["batch_values"]) + batch[tr]
        counts = np.bincount(keys); weights = 1.0 / np.maximum(counts[keys], 1)
        rng = np.random.default_rng(3407 + fold)
        best = None
        for gamma in (.03, .1, .3, 1., 3.):
            for alpha in (1e-6, 1e-5, 1e-4):
                train, val = align_source_to_target(train0, val0, a.mode)
                centers = train[rng.choice(len(train), min(a.centers, len(train)), replace=False,
                                          p=weights / weights.sum())]
                scores = rbf_scores(train, y[tr], val, centers, gamma, alpha, weights)
                score = fast_macro_f1(y[va], scores.argmax(1), len(labels))
                if best is None or score > best[0]: best = (score, gamma, alpha)
        train, test = align_source_to_target(train0, test0, a.mode)
        # Re-sample deterministic centers for the chosen model.
        rng = np.random.default_rng(3407 + fold)
        centers = train[rng.choice(len(train), min(a.centers, len(train)), replace=False,
                                  p=weights / weights.sum())]
        scores = rbf_scores(train, y[tr], test, centers, best[1], best[2], weights)
        m = classification_metrics(y[te].tolist(), scores.argmax(1).tolist(), labels)
        m.update(test_batch=fold, val_batch=val_batch, best_val_f1=best[0], gamma=best[1], alpha=best[2], mode=a.mode,
                 elapsed_seconds=time.perf_counter() - t0)
        fd = out / f"fold{fold}"; fd.mkdir(exist_ok=True); (fd / "test_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2)); rows.append(m)
        print(json.dumps({k:v for k,v in m.items() if k not in ("cm","report")}, ensure_ascii=False), flush=True)
    avg = {k: float(np.mean([m[k] for m in rows])) for k in ("acc", "bacc", "macro_f1")}
    (out / "summary.json").write_text(json.dumps({"mode":a.mode,"folds":rows,"AVG":avg}, ensure_ascii=False, indent=2)); print("SUMMARY",json.dumps(avg),flush=True)


if __name__ == "__main__": main()
