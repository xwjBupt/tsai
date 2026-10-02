"""Shared cache/checkpoint contracts; label IDs are local to each cache."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_data import PREPROCESS, SCHEMA_VERSION, source_manifest


def load_meta(cache, check_sources=False, data_root=None):
    with h5py.File(cache, "r") as h:
        if h.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("旧版或未知缓存，请运行 prepare_data.py 重建为 cache_v2.h5")
        result = {key: json.loads(h.attrs[key]) for key in
                  ("labels", "batch_values", "preprocess", "source_manifest", "counts")}
        result.update(y=h["y"][:], batch=h["batch"][:], wave=h["wave"][:],
                      n_points=h["x"].shape[-1], data_root=h.attrs["data_root"])
        if h["x"].shape[:2] != (len(result["y"]), 3) or len(h["batch"]) != len(result["y"]):
            raise ValueError("缓存形状不一致")
    if result["preprocess"] != PREPROCESS:
        raise ValueError("缓存预处理版本不兼容，请重建")
    if set(np.unique(result["y"])) != set(range(len(result["labels"]))) or set(np.unique(result["batch"])) != set(range(len(result["batch_values"]))):
        raise ValueError("缓存类别或批次映射无效")
    identity = {key: result[key] for key in ("labels", "batch_values", "preprocess", "source_manifest")}
    identity["wave"] = result["wave"].tolist()
    result["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if check_sources:
        root = Path(data_root or result["data_root"])
        if source_manifest(root) != result["source_manifest"]:
            raise ValueError("原始数据已新增/修改，缓存已过期；请运行 prepare_data.py --force 重建")
    return result


def model_from_config(config):
    from patchtst_model import PatchTSTClassifier
    return PatchTSTClassifier(config["n_classes"], config["n_points"], config.get("tsai_root"),
                              patch_len=config.get("patch_len", 32), stride=config.get("stride", 16),
                              n_layers=config.get("layers", 3), n_heads=config.get("heads", 8),
                              d_model=config.get("d_model", 128), d_ff=config.get("d_ff", 256),
                              dropout=config.get("dropout", .1),
                              revin=config.get("revin", True),
                              pooling=config.get("pooling", "mean"),
                              pool_segments=config.get("pool_segments", 4))


def load_checkpoint(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    required = ("wave", "preprocess", "labels", "batch_values", "config", "model", "test_batch",
                "val_batch", "test_indices", "limit_per_class", "cache_fingerprint")
    if ck.get("schema_version") != SCHEMA_VERSION or not all(key in ck for key in required):
        raise ValueError("旧版 checkpoint 缺少自描述信息，请用新版训练脚本重新训练")
    config = ck["config"]
    if (config["n_points"] != len(ck["wave"]) or config["labels"] != ck["labels"]
            or config["batch_values"] != ck["batch_values"]
            or config["n_classes"] != len(ck["labels"]) or config["n_batches"] != len(ck["batch_values"])):
        raise ValueError("checkpoint 的模型配置与类别/波数元数据不一致")
    if ck["preprocess"] != PREPROCESS:
        raise ValueError("checkpoint 使用不兼容的预处理版本")
    return ck


def check_checkpoint_cache(ck, meta):
    if ck["cache_fingerprint"] != meta["fingerprint"]:
        raise ValueError("缓存与 checkpoint 不匹配；请使用训练时的缓存，新增类别后必须重新训练")
