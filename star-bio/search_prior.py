"""RBF score calibration with an unlabeled target-class-prior constraint."""
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


def balanced_bias(scores, prior, iters=80, temperature=1.0):
    """Find additive class biases whose softmax marginal matches prior."""
    z = scores / max(temperature, 1e-5)
    bias = np.zeros(z.shape[1], dtype=np.float64)
    target = np.asarray(prior, dtype=np.float64); target /= target.sum()
    for _ in range(iters):
        p = np.exp(z + bias - (z + bias).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        diff = p.mean(0) - target
        bias -= 0.8 * diff / (p.var(0) + 1e-3)
        bias -= bias.mean()
        bias = np.clip(bias, -8, 8)
    return bias.astype(np.float32)


def prior_sets(y, batch, tr, n_cls, n_batches):
    global_prior = np.bincount(y[tr], minlength=n_cls) + 1
    per_batch = np.stack([np.bincount(y[tr][batch[tr] == b], minlength=n_cls) + 1 for b in np.unique(batch[tr])])
    return {
        "uniform": np.ones(n_cls),
        "global": global_prior,
        "batch_mean": per_batch.mean(0),
        "batch_median": np.median(per_batch, axis=0),
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--centers", type=int, default=4096)
    p.add_argument("--feature-path", default="/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy")
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--data-root", default="/home/wjx/CodeData/data/Star-Com/7class-4patch")
    a = p.parse_args(); meta = load_meta(a.cache, True, a.data_root)
    with h5py.File(a.cache, "r") as h: batch = h["batch"][:].astype(np.int64)
    x = np.asarray(np.load(a.feature_path, mmap_mode="r")[:, 0, :], dtype=np.float32)
    y, labels = meta["y"], meta["labels"]; n_cls = len(labels); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for fold in meta["batch_values"]:
        t0 = time.perf_counter(); idx, val_batch = make_splits(meta, fold); tr, va, te = idx["train"], idx["val"], idx["test"]
        train = row_snv(x[tr]); val = row_snv(x[va]); test = row_snv(x[te]); mu = train.mean(0); train -= mu; val -= mu; test -= mu
        keys = y[tr] * len(meta["batch_values"]) + batch[tr]; cnt = np.bincount(keys); weights = 1.0 / np.maximum(cnt[keys], 1)
        rng = np.random.default_rng(3407 + fold); centers = train[rng.choice(len(train), min(a.centers, len(train)), replace=False, p=weights / weights.sum())]
        priors = prior_sets(y, batch, tr, n_cls, len(meta["batch_values"])); best = None
        for gamma in (.1, .3, 1.):
            for alpha in (1e-6, 1e-4):
                sv = rbf_scores(train, y[tr], val, centers, gamma, alpha, weights)
                for pname, prior in priors.items():
                    for temp in (.5, 1., 2.):
                        bias = balanced_bias(sv, prior, temperature=temp); score = fast_macro_f1(y[va], (sv + bias).argmax(1), n_cls)
                        if best is None or score > best[0]: best = (score, gamma, alpha, pname, temp, bias)
        st = rbf_scores(train, y[tr], test, centers, best[1], best[2], weights)
        bias = balanced_bias(st, priors[best[3]], temperature=best[4]); pred = (st + bias).argmax(1)
        m = classification_metrics(y[te].tolist(), pred.tolist(), labels); m.update(test_batch=fold, val_batch=val_batch, best_val_f1=best[0], gamma=best[1], alpha=best[2], prior=best[3], temperature=best[4], bias=bias.tolist(), elapsed_seconds=time.perf_counter()-t0)
        fd=out/f"fold{fold}";fd.mkdir(exist_ok=True);(fd/"test_metrics.json").write_text(json.dumps(m,ensure_ascii=False,indent=2));rows.append(m);print(json.dumps({k:v for k,v in m.items() if k not in ('cm','report')},ensure_ascii=False),flush=True)
    avg={k:float(np.mean([m[k] for m in rows])) for k in ('acc','bacc','macro_f1')};(out/'summary.json').write_text(json.dumps({'folds':rows,'AVG':avg},ensure_ascii=False,indent=2));print('SUMMARY',json.dumps(avg),flush=True)


if __name__=='__main__':main()
