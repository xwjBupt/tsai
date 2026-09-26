"""Batch-held-out splits, with explicit coverage checks for future datasets."""
import numpy as np


def make_splits(meta, test_batch, val_batch=None):
    values = meta["batch_values"]
    if len(values) < 3:
        raise ValueError("独立训练/验证/测试至少需要 3 个批次")
    if test_batch not in values:
        raise ValueError(f"测试批次 {test_batch} 不存在，可选值: {values}")
    test_id = values.index(test_batch)
    # Choose next batch cyclically, using only batch IDs, never measured performance.
    val_batch = values[(test_id + 1) % len(values)] if val_batch is None else val_batch
    if val_batch not in values or val_batch == test_batch:
        raise ValueError("验证批次必须存在且与测试批次不同")
    val_id = values.index(val_batch)
    batches, y = meta["batch"], meta["y"]
    indices = {
        "train": np.flatnonzero((batches != test_id) & (batches != val_id)),
        "val": np.flatnonzero(batches == val_id),
        "test": np.flatnonzero(batches == test_id),
    }
    for name, idx in indices.items():
        missing = sorted(set(range(len(meta["labels"]))) - set(y[idx]))
        if missing:
            labels = [meta["labels"][i] for i in missing]
            raise ValueError(f"{name} 集缺少类别 {labels}；请补充相应批次数据或指定其他 --val-batch")
    return indices, val_batch


def limit_per_class(indices, y, limit, seed):
    """A deterministic small subset for smoke tests, preserving every class."""
    if limit is None:
        return indices
    if limit < 1:
        raise ValueError("每类样本上限必须为正数")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(indices[y[indices] == c], min(limit, np.sum(y[indices] == c)), replace=False)
        for c in np.unique(y[indices])
    ]))
