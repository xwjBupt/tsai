"""Build a versioned, self-describing spectral cache from class-batch folders."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from tqdm import tqdm

SCHEMA_VERSION = 2
PREPROCESS = {
    "version": 1, "crop_start": 220, "crop_stop": 1260,
    "median_window": 5, "spike_threshold": 8.0,
    "baseline_window": 101, "baseline_order": 3,
    "ch_start": 2800.0, "ch_end": 3000.0,
    "derivative_window": 9, "derivative_order": 2,
}


def discover_files(data_root):
    entries = []
    for path in Path(data_root).glob("*/Cell_data.csv"):
        name = path.parent.name
        if "-" not in name:
            raise ValueError(f"目录名必须是 '<类别>-<数字批次>': {name}")
        label, batch = name.rsplit("-", 1)
        if not label or not batch.isdigit():
            raise ValueError(f"目录名必须是 '<类别>-<数字批次>': {name}")
        entries.append((path, label, int(batch), name))
    if not entries:
        raise FileNotFoundError(f"未找到 {data_root}/*/Cell_data.csv")
    entries.sort(key=lambda x: (x[1], x[2], x[3]))
    pairs = [(e[1], e[2]) for e in entries]
    if len(pairs) != len(set(pairs)):
        raise ValueError("存在重复的类别/批次，如 class-1 与 class-01")
    return entries, sorted({e[1] for e in entries}), sorted({e[2] for e in entries})


def source_manifest(data_root, entries=None):
    entries = entries if entries is not None else discover_files(data_root)[0]
    return [{"path": str(e[0].relative_to(data_root)), "size": e[0].stat().st_size,
             "mtime_ns": e[0].stat().st_mtime_ns} for e in entries]


def validate_wave(wave, path="光谱"):
    if wave.ndim != 1 or len(wave) < PREPROCESS["crop_stop"]:
        raise ValueError(f"{path}: 波数必须至少有 {PREPROCESS['crop_stop']} 列")
    if not np.isfinite(wave).all() or len(np.unique(wave)) != len(wave):
        raise ValueError(f"{path}: 波数包含非有限值或重复值")
    d = np.diff(wave)
    if not ((d > 0).all() or (d < 0).all()):
        raise ValueError(f"{path}: 波数必须严格单调")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            wave = np.asarray(next(reader), dtype=np.float64)
            rows = [np.asarray(row, dtype=np.float32) for row in reader if row]
        except (StopIteration, ValueError) as exc:
            raise ValueError(f"{path}: CSV 为空或含非数字数据") from exc
    validate_wave(wave, path)
    if not rows or any(row.shape != wave.shape for row in rows):
        raise ValueError(f"{path}: 无细胞数据或强度列数与波数列数不一致")
    raw = np.stack(rows)
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: 强度含 NaN/Inf")
    return wave, raw


def process(wave, raw, target_wave, settings=None):
    settings = PREPROCESS if settings is None else settings
    if settings != PREPROCESS:
        raise ValueError("预处理版本/参数与当前实现不兼容")
    wave, raw, grid = np.asarray(wave), np.asarray(raw), np.asarray(target_wave)
    validate_wave(wave)
    if raw.ndim != 2 or raw.shape[1] != len(wave) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("强度必须是非空且有限的 [细胞数, 波数点数] 数组")
    if grid.ndim != 1 or len(grid) < settings["baseline_window"] or not np.isfinite(grid).all() or not (np.diff(grid) > 0).all():
        raise ValueError("目标波数网格必须递增且至少有 101 个点")
    start, stop = settings["crop_start"], settings["crop_stop"]
    wave, raw = wave[start:stop], raw[:, start:stop]
    order = np.argsort(wave)
    wave, raw = wave[order], raw[:, order]
    if grid[0] < wave[0] - 1e-4 or grid[-1] > wave[-1] + 1e-4:
        raise ValueError("CSV 有效波数范围未覆盖训练网格，不能外推；请检查仪器/列索引")
    x = interp1d(wave, raw, axis=1, bounds_error=True)(np.clip(grid, wave[0], wave[-1])).astype(np.float32)
    med = median_filter(x, size=(1, settings["median_window"]), mode="nearest")
    bad = np.abs(x - med) > settings["spike_threshold"] * (np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-3)
    x[bad] = med[bad]
    base = savgol_filter(x, settings["baseline_window"], settings["baseline_order"], axis=1, mode="interp")
    corr = np.maximum(x - base, 0)
    ch = (grid >= settings["ch_start"]) & (grid <= settings["ch_end"])
    if ch.sum() < 3:
        raise ValueError("目标网格没有覆盖足够的 CH 区 (2800–3000 cm⁻¹)")
    scale = np.percentile(corr[:, ch], 99, axis=1, keepdims=True)
    if np.any(scale <= 1e-6):
        raise ValueError("存在没有可用 CH 峰的光谱；请先检查/清理数据")
    norm = corr / (scale + 1e-6)
    deriv = savgol_filter(norm, settings["derivative_window"], settings["derivative_order"], deriv=1, axis=1)
    snv = (norm - norm.mean(1, keepdims=True)) / (norm.std(1, keepdims=True) + 1e-6)
    result = np.stack([norm, deriv, snv], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("预处理产生非有限值")
    return result


def build_cache(data_root, cache, step=1.856, grid_start=None, grid_end=None, force=False):
    entries, labels, batches = discover_files(data_root)
    manifest = source_manifest(data_root, entries)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step 必须为正数")
    waves = []
    for path, *_ in entries:
        with path.open(newline="", encoding="utf-8-sig") as f:
            wave = np.asarray(next(csv.reader(f)), dtype=np.float64)
        validate_wave(wave, path)
        waves.append(wave[PREPROCESS["crop_start"]:PREPROCESS["crop_stop"]])
    lo = max(w.min() for w in waves) if grid_start is None else grid_start
    hi = min(w.max() for w in waves) if grid_end is None else grid_end
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("各文件有效波数区间无交集或指定范围无效")
    grid = lo + np.arange(int(np.floor((hi - lo) / step)) + 1) * step
    if len(grid) < PREPROCESS["baseline_window"]:
        raise ValueError("目标网格至少需要 101 个点")
    cache = Path(cache)
    if cache.exists() and not force:
        raise FileExistsError(f"缓存已存在: {cache}，重建请加 --force")
    cache.parent.mkdir(parents=True, exist_ok=True)
    label_ids = {name: i for i, name in enumerate(labels)}
    batch_ids = {value: i for i, value in enumerate(batches)}
    fd, temp = tempfile.mkstemp(prefix=cache.name + ".", suffix=".tmp", dir=cache.parent)
    os.close(fd)
    counts = []
    try:
        with h5py.File(temp, "w") as h:
            h.attrs["schema_version"] = SCHEMA_VERSION
            h.attrs["labels"] = json.dumps(labels, ensure_ascii=False)
            h.attrs["batch_values"] = json.dumps(batches)
            h.attrs["preprocess"] = json.dumps(PREPROCESS)
            h.attrs["source_manifest"] = json.dumps(manifest, ensure_ascii=False)
            h.attrs["data_root"] = str(Path(data_root).resolve())
            h.attrs["n_points"] = len(grid)
            h.create_dataset("wave", data=grid)
            h.create_dataset("x", shape=(0, 3, len(grid)), maxshape=(None, 3, len(grid)), dtype="f4", chunks=(64, 3, len(grid)), compression="lzf")
            for key in ("y", "batch", "row"):
                h.create_dataset(key, shape=(0,), maxshape=(None,), dtype="i8")
            h.create_dataset("file", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
            offset = 0
            for path, label, batch, name in tqdm(entries, desc="preparing"):
                wave, raw = read_csv(path)
                x = process(wave, raw, grid)
                end = offset + len(x)
                for key in ("x", "y", "batch", "row", "file"):
                    h[key].resize(end, axis=0)
                h["x"][offset:end] = x
                h["y"][offset:end] = label_ids[label]
                h["batch"][offset:end] = batch_ids[batch]
                h["row"][offset:end] = np.arange(1, len(x) + 1)
                h["file"][offset:end] = [name] * len(x)
                counts.append({"class": label, "batch": batch, "cells": len(x)})
                offset = end
            h.attrs["counts"] = json.dumps(counts, ensure_ascii=False)
        if source_manifest(data_root) != manifest:
            raise RuntimeError("构建期间原始文件发生变化，请重新运行")
        os.replace(temp, cache)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    summary = {"cache": str(cache), "cells": offset, "labels": labels, "batches": batches,
               "n_points": len(grid), "wave_range": [float(grid[0]), float(grid[-1])], "counts": counts}
    cache.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    from config import Config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Config().data_root)
    parser.add_argument("--cache", help="默认 <data-root>/cache_v2.h5")
    parser.add_argument("--grid-start", type=float)
    parser.add_argument("--grid-end", type=float)
    parser.add_argument("--step", type=float, default=1.856)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_cache(args.data_root, args.cache or str(Path(args.data_root) / "cache_v2.h5"), args.step, args.grid_start, args.grid_end, args.force)


if __name__ == "__main__":
    main()
