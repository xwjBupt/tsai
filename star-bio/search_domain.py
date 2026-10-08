"""Strict LOBO search for unlabeled batch alignment and score calibration.

The held-out batch is used only for unsupervised marginal statistics. Labels from
the held-out batch are never used to select a transform, kernel, or score bias.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from metadata import load_meta
from metrics import classification_metrics
from splits import make_splits


def row_snv(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def row_rank(x):
    """Map each spectrum to a stable within-spectrum rank in [-1, 1]."""
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[np.arange(len(x))[:, None], order] = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float32)
    return ranks


def quantile_map(x, source, reference, q):
    """Map each wavelength's source marginal to the reference marginal."""
    sx = np.quantile(source, q, axis=0).astype(np.float32)
    rx = np.quantile(reference, q, axis=0).astype(np.float32)
    # np.interp does not vectorize over columns, but this remains bounded by
    # 1105 columns and avoids allocating a third copy of the full data matrix.
    out = np.empty_like(x, dtype=np.float32)
    for j in range(x.shape[1]):
        # Repeated quantiles can occur in nearly flat spectral regions.
        keep = np.r_[True, np.diff(sx[:, j]) > 1e-7]
        out[:, j] = np.interp(x[:, j], sx[keep, j], rx[keep, j]).astype(np.float32)
    return out


def fit_transformer(train_x, train_batch, mode):
    reference = train_x
    state = {"mode": mode, "ref_mean": reference.mean(0),
             "ref_std": reference.std(0) + 1e-5}
    if mode in ("batch_center", "batch_affine"):
        state["stats"] = {}
        for b in np.unique(train_batch):
            z = train_x[train_batch == b]
            state["stats"][int(b)] = (z.mean(0), z.std(0) + 1e-5)
    if mode == "batch_quantile":
        state["q"] = np.linspace(.01, .99, 33, dtype=np.float32)
        state["source_q"] = {int(b): np.quantile(train_x[train_batch == b], state["q"], axis=0).astype(np.float32)
                              for b in np.unique(train_batch)}
        state["reference_q"] = np.quantile(reference, state["q"], axis=0).astype(np.float32)
    return state


def apply_transform(x, batch, state):
    mode = state["mode"]
    if mode in ("base", "row_rank"):
        return x
    if mode == "batch_center":
        out = np.empty_like(x)
        for b in np.unique(batch):
            z = state["stats"].get(int(b))
            if z is None:
                # This branch is used only when applying a genuinely unseen
                # domain; its own unlabeled moments are intentionally allowed.
                mu, _ = x[batch == b].mean(0), x[batch == b].std(0)
            else:
                mu, _ = z
            out[batch == b] = x[batch == b] - mu + state["ref_mean"]
        return out
    if mode == "batch_affine":
        out = np.empty_like(x)
        for b in np.unique(batch):
            mask = batch == b
            z = state["stats"].get(int(b))
            if z is None:
                mu, sd = x[mask].mean(0), x[mask].std(0) + 1e-5
            else:
                mu, sd = z
            out[mask] = (x[mask] - mu) / sd * state["ref_std"] + state["ref_mean"]
        return out
    if mode == "batch_quantile":
        out = np.empty_like(x)
        for b in np.unique(batch):
            mask = batch == b
            qsrc = state["source_q"].get(int(b))
            if qsrc is None:
                q = state["q"]
                qsrc = np.quantile(x[mask], q, axis=0).astype(np.float32)
            for j in range(x.shape[1]):
                keep = np.r_[True, np.diff(qsrc[:, j]) > 1e-7]
                out[mask, j] = np.interp(x[mask, j], qsrc[keep, j], state["reference_q"][keep, j])
        return out
    raise ValueError(f"unknown mode: {mode}")


def rbf_scores(train, labels, query, centers, gamma, alpha, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    tx = torch.as_tensor(train, dtype=dtype, device=device)
    qx = torch.as_tensor(query, dtype=dtype, device=device)
    cc = torch.as_tensor(centers, dtype=dtype, device=device)
    # Median squared distance stabilizes gamma across preprocessing variants.
    denom = torch.cdist(cc[: min(256, len(cc))], cc[: min(256, len(cc))]).square().median().clamp_min(1e-6)
    tr_d = torch.cdist(tx, cc).square() / denom
    q_d = torch.cdist(qx, cc).square() / denom
    phi = torch.exp(-gamma * tr_d)
    qphi = torch.exp(-gamma * q_d)
    phi = torch.cat([phi, torch.ones(len(phi), 1, device=device)], 1)
    qphi = torch.cat([qphi, torch.ones(len(qphi), 1, device=device)], 1)
    n_cls = int(np.max(labels)) + 1
    yy = torch.nn.functional.one_hot(torch.as_tensor(labels, device=device), n_cls).to(dtype)
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    ww = torch.as_tensor(weights, dtype=dtype, device=device)
    ww = ww / ww.sum()
    cov = (phi.T * ww) @ phi
    rhs = (phi.T * ww) @ yy
    ridge = torch.as_tensor(alpha, dtype=dtype, device=device)
    coef = torch.linalg.solve(cov + ridge * torch.eye(cov.shape[0], dtype=dtype, device=device), rhs)
    return (qphi @ coef).detach().cpu().numpy()


def fast_macro_f1(y, pred, n_cls):
    cm = np.bincount(y.astype(np.int64) * n_cls + pred.astype(np.int64), minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(cm).astype(np.float64)
    f1 = 2 * tp / np.maximum(2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1e-12)
    return float(f1.mean())


def tune_bias(scores, y, labels, steps=2):
    """Coordinate-search class score offsets for validation Macro-F1."""
    bias = np.zeros(scores.shape[1], dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    best = fast_macro_f1(y, (scores + bias).argmax(1), len(labels))
    for step in (0.50, 0.20, 0.08)[:steps + 1]:
        changed = True
        while changed:
            changed = False
            for c in range(scores.shape[1]):
                old = bias[c]
                local_best = best
                local_value = old
                for delta in np.arange(-step, step + 1e-6, step / 2):
                    bias[c] = old + float(delta)
                    score = fast_macro_f1(y, (scores + bias).argmax(1), len(labels))
                    if score > local_best + 1e-8:
                        local_best, local_value = score, bias[c]
                bias[c] = local_value
                if local_best > best + 1e-8:
                    best, changed = local_best, True
    return bias, best


def run_fold(x, batch, y, labels, fold, mode, centers_n, gammas, calibrate):
    idx, val_batch = make_splits(META, fold)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    train_batch = batch[tr]
    state = fit_transformer(x[tr], train_batch, mode)
    # For validation and test, batch IDs are known but labels are never used.
    tx = apply_transform(x[tr], train_batch, state)
    vx = apply_transform(x[va], batch[va], state)
    ex = apply_transform(x[te], batch[te], state)
    # Match the strongest existing RBF baseline: fit a train-only global
    # feature center after any domain transform and reuse it everywhere.
    post_mean = tx.mean(0)
    tx = tx - post_mean
    vx = vx - post_mean
    ex = ex - post_mean
    rng = np.random.default_rng(3407 + fold)
    keys = y[tr] * len(np.unique(batch)) + train_batch
    counts = np.bincount(keys)
    sample_w = 1.0 / np.maximum(counts[keys], 1)
    centers = tx[rng.choice(len(tx), min(centers_n, len(tx)), replace=False, p=sample_w / sample_w.sum())]
    best = None
    for gamma in gammas:
        for alpha in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1):
            scores_v = rbf_scores(tx, y[tr], vx, centers, gamma, alpha, sample_w)
            bias = np.zeros(len(labels), dtype=np.float32)
            val_score = classification_metrics(y[va].tolist(), scores_v.argmax(1).tolist(), labels)["macro_f1"]
            if calibrate:
                bias, val_score = tune_bias(scores_v, y[va], labels)
            if best is None or val_score > best[0]:
                best = (val_score, gamma, alpha, bias)
    scores_t = rbf_scores(tx, y[tr], ex, centers, best[1], best[2], sample_w)
    pred = (scores_t + best[3]).argmax(1)
    metrics = classification_metrics(y[te].tolist(), pred.tolist(), labels)
    metrics.update(test_batch=fold, val_batch=val_batch, best_val_f1=best[0], gamma=best[1],
                   alpha=best[2], bias=best[3].tolist(), mode=mode, calibrated=calibrate)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["base", "batch_center", "batch_affine", "batch_quantile", "row_rank"], required=True)
    p.add_argument("--centers", type=int, default=2048)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--data-root", default="/home/wjx/CodeData/data/Star-Com/7class-4patch")
    p.add_argument("--feature-path", default="/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy")
    p.add_argument("--start", type=int, default=0, help="特征起始点，按目标波数网格索引")
    p.add_argument("--stop", type=int, default=1105, help="特征终止点（不含）")
    p.add_argument("--step", type=int, default=1, help="特征抽样步长")
    a = p.parse_args()
    global META
    META = load_meta(a.cache, True, a.data_root)
    with h5py.File(a.cache, "r") as h:
        batch = h["batch"][:].astype("int64")
    feature = np.load(a.feature_path, mmap_mode="r")
    if feature.ndim != 3 or feature.shape[0] != len(META["y"]):
        raise ValueError("feature-path 必须是 [N, channels, points] 且与 cache 行数一致")
    # raw_snv channel 0 is the strongest existing RBF representation. It is
    # already per-spectrum SNV normalized; other modes operate on this same
    # representation for a fair comparison.
    x = np.asarray(feature[:, 0, :], dtype="float32")
    if not (0 <= a.start < a.stop <= x.shape[1] and a.step >= 1):
        p.error("start/stop/step 超出特征范围")
    x = x[:, a.start:a.stop:a.step]
    if a.mode == "row_rank":
        x = row_rank(x)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in META["batch_values"]:
        started = time.perf_counter()
        m = run_fold(x, batch, META["y"], META["labels"], fold, a.mode, a.centers, (.01, .03, .1, .3, 1., 3.), a.calibrate)
        m["elapsed_seconds"] = time.perf_counter() - started
        fd = out / f"fold{fold}"; fd.mkdir(exist_ok=True)
        (fd / "test_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
        rows.append(m); print(json.dumps({k: v for k, v in m.items() if k not in ("cm", "report")}, ensure_ascii=False), flush=True)
    avg = {k: float(np.mean([m[k] for m in rows])) for k in ("acc", "bacc", "macro_f1")}
    summary = {"mode": a.mode, "centers": a.centers, "calibrated": a.calibrate, "folds": rows, "AVG": avg}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps(avg), flush=True)


if __name__ == "__main__":
    main()
