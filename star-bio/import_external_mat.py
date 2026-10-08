"""Convert a public MATLAB Raman dataset into an unlabeled .npy corpus.

The importer intentionally ignores class labels: external data is used for
self-supervised pretraining, so no external class mapping can leak into LOBO.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def numeric_arrays(value):
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.ndim >= 1:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from numeric_arrays(item)


def to_cells(array):
    x = np.asarray(array, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        return None
    # MATLAB datasets are commonly [cells, points] or [points, cells].
    return x if x.shape[0] <= x.shape[1] else x.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-points", type=int, default=200)
    args = p.parse_args()
    raw = loadmat(args.mat)
    groups = []
    for name, value in raw.items():
        if name.startswith("_"):
            continue
        for array in numeric_arrays(value):
            cells = to_cells(array)
            if cells is not None and cells.shape[1] >= args.min_points:
                cells = cells[np.isfinite(cells).all(1)]
                if len(cells):
                    groups.append((name, cells))
    if not groups:
        raise ValueError("没有找到二维有限数值光谱矩阵")
    lengths = [x.shape[1] for _, x in groups]
    point_count = min(lengths)
    x = np.concatenate([cells[:, :point_count] for _, cells in groups]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("外部数据仍包含 NaN/Inf")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x)
    out.with_suffix(".json").write_text(json.dumps({
        "source": str(Path(args.mat).resolve()), "groups": {k: int(len(v)) for k, v in groups},
        "shape": list(x.shape), "labels_discarded": True,
        "point_count": point_count,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "shape": list(x.shape), "groups": [k for k, _ in groups]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
