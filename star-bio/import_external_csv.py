"""Convert a public Raman CSV to the target wavelength grid for pretraining.

Labels and replicate metadata are intentionally ignored. The output is a
single [N, points] array so it can be consumed by ``pretrain_masked.py``.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--cache", default="/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with h5py.File(a.cache, "r") as h:
        grid = h["wave"][:].astype(np.float64)
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        raise ValueError("CSV 至少需要一行波数和两条光谱")
    # Zenodo's CSV has a textual V1... header followed by the wavelength row.
    try:
        wave = np.asarray(rows[0][1:], dtype=np.float64)
        data_rows = rows[1:]
    except ValueError:
        wave = np.asarray(rows[1][1:], dtype=np.float64)
        data_rows = rows[2:]
    raw = np.asarray([r[1:] for r in data_rows], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(wave) or not np.isfinite(raw).all():
        raise ValueError("CSV 光谱必须为有限数值矩阵")
    order = np.argsort(wave)
    wave, raw = wave[order], raw[:, order]
    if grid[0] < wave[0] or grid[-1] > wave[-1]:
        raise ValueError(f"外部波数范围 {wave[0]}-{wave[-1]} 未覆盖目标 {grid[0]}-{grid[-1]}")
    x = interp1d(wave, raw, axis=1, bounds_error=True)(grid).astype(np.float32)
    smooth = savgol_filter(x, 9, 2, axis=1, mode="interp").astype(np.float32)
    smooth = median_filter(smooth, size=(1, 5), mode="nearest")
    baseline = savgol_filter(smooth, 101, 3, axis=1, mode="interp")
    corr = smooth - baseline
    ch = (grid >= 2800) & (grid <= 3000)
    scale = np.percentile(np.maximum(corr[:, ch], 0), 99, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    norm = (corr / scale).astype(np.float32)
    norm = (norm - norm.mean(1, keepdims=True)) / (norm.std(1, keepdims=True) + 1e-6)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, norm.astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps({
        "source": str(Path(a.csv).resolve()), "shape": list(norm.shape),
        "grid": [float(grid[0]), float(grid[-1])], "labels_discarded": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "shape": list(norm.shape)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
