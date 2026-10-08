"""Four-fold leave-one-batch-out RBF evaluation without a validation set.

Each fold fits on all three non-test batches and evaluates once on the held-out
batch. Kernel hyperparameters are fixed before the run via command-line flags.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from metadata import load_meta
from metrics import classification_metrics
from search_domain import rbf_scores, row_snv
from splits import make_lobo_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--feature-path", default="/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy")
    p.add_argument("--channels", default="0")
    p.add_argument("--centers", type=int, default=4096)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--data-root", default="/home/wjx/CodeData/data/Star-Com/7class-4patch")
    args = p.parse_args()
    if args.centers < 1 or args.gamma <= 0 or args.alpha <= 0:
        p.error("centers、gamma、alpha 必须为正数")

    meta = load_meta(args.cache, True, args.data_root)
    y, labels = meta["y"], meta["labels"]
    channels = [int(value) for value in args.channels.split(",") if value.strip()]
    if not channels or any(value < 0 or value >= 3 for value in channels):
        p.error("channels 必须是 0、1、2 的逗号分隔列表")
    with h5py.File(args.cache, "r") as h:
        batch = h["batch"][:]
    arr = np.load(args.feature_path, mmap_mode="r")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x = np.asarray(arr[:, channels, :], dtype="float32").reshape(len(arr), -1).copy()
    x = row_snv(x)
    rows = []

    for fold in meta["batch_values"]:
        started = time.perf_counter()
        indices = make_lobo_splits(meta, fold)
        train, test = indices["train"], indices["test"]
        tx, ex = x[train].copy(), x[test].copy()
        mean = tx.mean(0)
        tx -= mean
        ex -= mean
        keys = y[train] * len(meta["batch_values"]) + batch[train]
        counts = np.bincount(keys)
        weights = 1 / np.maximum(counts[keys], 1)
        rng = np.random.default_rng(args.seed + fold)
        centers = tx[rng.choice(
            len(tx), min(args.centers, len(tx)), replace=False,
            p=weights / weights.sum(),
        )]
        scores = rbf_scores(tx, y[train], ex, centers, args.gamma, args.alpha, weights)
        metrics = classification_metrics(y[test].tolist(), scores.argmax(1).tolist(), labels)
        metrics.update(
            test_batch=fold,
            train_count=len(train),
            gamma=args.gamma,
            alpha=args.alpha,
            centers=len(centers),
            channels=channels,
            validation=False,
            elapsed_seconds=time.perf_counter() - started,
        )
        folder = out / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        (folder / "test_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(folder / "splits.npz", **indices)
        rows.append(metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("cm", "report")}, ensure_ascii=False), flush=True)

    average = {key: float(np.mean([item[key] for item in rows])) for key in ("acc", "bacc", "macro_f1")}
    summary = {
        "folds": rows,
        "AVG": average,
        "channels": channels,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "centers": args.centers,
        "validation": False,
        "split_protocol": "three_batches_train_one_batch_test",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", average, flush=True)


if __name__ == "__main__":
    main()
