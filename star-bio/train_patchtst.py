"""Train tsai PatchTST on the Cell cache using exactly one GPU per process."""
import argparse
import fcntl
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import csv
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from loguru import logger

from config import Config
from dataset import SpectraDataset
from metadata import load_meta
from metrics import classification_metrics
from prepare_data import SCHEMA_VERSION
from splits import make_splits, limit_per_class
from patchtst_model import PatchTSTClassifier


def is_dist(): return False
def rank(): return 0
def world(): return 1
def main_process(): return True


def setup_logger():
    logger.remove()
    if main_process():
        logger.add(sys.stderr, level="INFO", colorize=True,
                   format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    else:
        logger.add(lambda _: None, level="CRITICAL")
    return logger

def setup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("当前配置为单卡训练，请直接运行 python train_patchtst.py，不要使用 torchrun")
    requested = os.environ.get("PATCHTST_DEVICE", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_gpu(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if args.device == "cuda":
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")
        return torch.device("cpu")
    gpu_id = args.gpu_id if args.gpu_id is not None else max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id 必须在 0..{torch.cuda.device_count()-1} 范围内")
    torch.cuda.set_device(gpu_id)
    return torch.device("cuda", gpu_id)

def cleanup_ddp():
    return None
def seed_all(seed):
    random.seed(seed + rank()); np.random.seed(seed + rank()); torch.manual_seed(seed + rank())

def save_json(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit_for_experiment(stamp):
    """Commit the star-bio code and return the short commit identifier."""
    repo = Path(__file__).resolve().parents[1]
    try:
        subprocess.run(["git", "add", "star-bio"], cwd=repo, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        message = f"star-bio experiment {stamp}"
        subprocess.run(["git", "commit", "--allow-empty", "-m", message], cwd=repo,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True).stdout.strip()
        return commit, message
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"正式实验 git commit 失败: {detail}") from exc


def prepare_experiment(args):
    stamp = datetime.now().strftime("%y-%m-%d@%H-%M-%S")
    if args.debug:
        experiment_id = f"{stamp}+debug"
        commit, commit_message = None, None
        root_name = "debug"
    elif args.commit_id:
        experiment_id = f"{stamp}+commit-{args.commit_id}"
        commit, commit_message = args.commit_id, f"shared commit {args.commit_id}"
        root_name = "outputs"
    else:
        commit, commit_message = git_commit_for_experiment(stamp)
        experiment_id = f"{stamp}+commit-{commit}"
        root_name = "outputs"
    base = Path(args.output_root) if args.output_root else Path(__file__).resolve().parent / root_name
    experiment_root = base / experiment_id
    if experiment_root.exists():
        raise FileExistsError(f"实验目录已存在: {experiment_root}，请稍后重试或指定其他 --output-root")
    experiment_root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": stamp,
        "debug": bool(args.debug),
        "commit": commit,
        "commit_message": commit_message,
        "command": sys.argv,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(experiment_root / "experiment.json", metadata)
    return experiment_root, metadata


def append_results_csv(results, labels, experiment_meta, output_path=None):
    """Append a completed four-fold test summary to the fixed results file."""
    if experiment_meta["debug"] or len(results) != 4:
        return None
    result_path = Path(output_path) if output_path else Path(__file__).resolve().parent / "results.csv"
    fold_keys = [f"fold{key}" for key in sorted(results)]
    metric_values = {
        "ACC": lambda item: item["acc"],
        "Balanced_Accuracy": lambda item: item["bacc"],
        "Macro_F1": lambda item: item["macro_f1"],
        "Macro_Precision": lambda item: item["report"]["macro avg"]["precision"],
        "Macro_Recall": lambda item: item["report"]["macro avg"]["recall"],
        "Weighted_F1": lambda item: item["report"]["weighted avg"]["f1-score"],
        "Weighted_Precision": lambda item: item["report"]["weighted avg"]["precision"],
        "Weighted_Recall": lambda item: item["report"]["weighted avg"]["recall"],
        "Test_Loss": lambda item: item.get("test_loss"),
    }
    for label in labels:
        metric_values[f"{label}_Precision"] = lambda item, name=label: item["report"][name]["precision"]
        metric_values[f"{label}_Recall"] = lambda item, name=label: item["report"][name]["recall"]
        metric_values[f"{label}_F1"] = lambda item, name=label: item["report"][name]["f1-score"]
        metric_values[f"{label}_Support"] = lambda item, name=label: item["report"][name]["support"]
    rows = []
    for metric, getter in metric_values.items():
        raw_values = [getter(results[int(key[4:])]) for key in fold_keys]
        values = [float(value) if value is not None else None for value in raw_values]
        complete = all(value is not None for value in values)
        row = {
            "experiment_id": experiment_meta["experiment_id"],
            "timestamp": experiment_meta["timestamp"],
            "commit": experiment_meta["commit"] or "",
            "metric": metric,
        }
        row.update({key: value if value is not None else "" for key, value in zip(fold_keys, values)})
        row["AVG"] = float(np.mean(values)) if complete else ""
        row["STD"] = float(np.std(values)) if complete else ""
        rows.append(row)
    fieldnames = ["experiment_id", "timestamp", "commit", "metric", *fold_keys, "AVG", "STD"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(result_path.resolve()).encode("utf-8")
    lock_name = __import__("hashlib").sha256(lock_key).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"star-bio-results-{lock_name}.lock"
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        previous_rows = []
        previous_fields = []
        if result_path.exists() and result_path.stat().st_size > 0:
            with result_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                previous_fields = reader.fieldnames or []
                previous_rows = [row for row in reader if row.get("experiment_id") != experiment_meta["experiment_id"]]
        merged_fields = list(dict.fromkeys([*previous_fields, *fieldnames]))
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         dir=result_path.parent,
                                         prefix=result_path.name + ".", suffix=".tmp",
                                         delete=False) as f:
            temp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=merged_fields)
            writer.writeheader()
            writer.writerows(previous_rows)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, result_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    logger.info("四折结果已追加到 {}，共 {} 项指标", result_path, len(rows))
    return result_path

def lr_at(cfg, epoch):
    warm = min(cfg.warmup_epochs, max(0, cfg.epochs - 1))
    if warm and epoch <= warm: return cfg.lr * epoch / warm
    p = (epoch - warm - 1) / max(1, cfg.epochs - warm - 1)
    return cfg.lr * (.01 + .99 * (1 + math.cos(math.pi * p)) / 2)

def make_loader(cache, idx, cfg, labels, batches, train, device, args):
    ds = SpectraDataset(cache, idx, train, batch_shift=args.batch_shift,
                        augmentation_strength=args.augmentation_strength)
    if train:
        if args.sampler == "joint":
            # Balance the observed (class, batch) combinations so one batch
            # cannot dominate the representation during leave-one-batch-out.
            keys = labels[idx] * cfg.n_batches + batches[idx]
            counts = np.bincount(keys, minlength=cfg.n_classes * cfg.n_batches)
            sample_weights = (1 / np.maximum(counts, 1))[keys]
        elif args.sampler == "class":
            counts = np.bincount(labels[idx], minlength=cfg.n_classes)
            sample_weights = (1 / np.maximum(counts, 1))[labels[idx]]
        else:
            sample_weights = np.ones(len(idx), dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(idx), replacement=True)
    else: sampler = None
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None and train),
                      num_workers=cfg.workers, pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0)


def gather_eval(model, loader, device, labels):
    model.eval(); ys=[]; ps=[]
    loss_sum = 0.0
    sample_count = 0
    eval_model = model
    with torch.inference_mode():
        for x,y,_ in loader:
            logits = eval_model(x.to(device))
            loss_sum += float(nn.functional.cross_entropy(logits, y.to(device), reduction="sum"))
            sample_count += len(y)
            ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(y.tolist())
    result = classification_metrics(ys,ps,labels)
    result["val_loss"] = float(loss_sum / max(1, sample_count))
    return result


def reduce_train_loss(loss_sum, sample_count, device):
    return float(loss_sum / max(1, sample_count))


def save_history_artifacts(history, output, labels):
    """Keep history.json unchanged and add a flat CSV plus a four-panel curve."""
    if not history:
        return
    fields = ["epoch", "loss", "train_loss", "val_loss", "lr", "learning_rate",
              "acc", "bacc", "macro_f1", "macro_precision", "macro_recall"]
    for label in labels:
        fields.extend([f"{label}_precision", f"{label}_recall", f"{label}_f1"])
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in fields}
            macro = row.get("report", {}).get("macro avg", {})
            flat["macro_precision"] = macro.get("precision", "")
            flat["macro_recall"] = macro.get("recall", "")
            for label in labels:
                values = row.get("report", {}).get(label, {})
                flat[f"{label}_precision"] = values.get("precision", "")
                flat[f"{label}_recall"] = values.get("recall", "")
                flat[f"{label}_f1"] = values.get("f1-score", "")
            writer.writerow(flat)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(epochs, [row.get("train_loss", row.get("loss")) for row in history], label="train loss")
        ax.plot(epochs, [row.get("val_loss", float("nan")) for row in history], label="val loss")
        ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.grid(alpha=.25); ax.legend()
        ax = axes[0, 1]
        ax.plot(epochs, [row.get("lr", row.get("learning_rate")) for row in history], label="learning rate", color="tab:orange")
        ax.set_title("Learning rate"); ax.set_xlabel("epoch"); ax.grid(alpha=.25); ax.legend()
        ax = axes[1, 0]
        for key in ("acc", "bacc", "macro_f1"):
            ax.plot(epochs, [row[key] for row in history], label=key)
        ax.plot(epochs, [row.get("report", {}).get("macro avg", {}).get("precision", float("nan")) for row in history], label="macro_precision", linestyle="--")
        ax.plot(epochs, [row.get("report", {}).get("macro avg", {}).get("recall", float("nan")) for row in history], label="macro_recall", linestyle="--")
        ax.set_title("Validation metrics"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend()
        ax = axes[1, 1]
        for index, label in enumerate(labels):
            values = [row.get("report", {}).get(label, {}).get("f1-score", float("nan")) for row in history]
            ax.plot(epochs, values, label=label)
        ax.set_title("Per-class validation F1"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1); ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(output / "curves.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        logger.warning("曲线 PNG 生成失败，history.csv 仍已保存: {}", exc)

def probe_batch(cfg, device, args):
    if args.batch_size is not None: return args.batch_size
    if device.type != "cuda": return Config().batch_size
    candidate = max(1, 2 ** int(math.log2(args.auto_batch_start)))
    model = PatchTSTClassifier(cfg.n_classes,cfg.n_points,args.tsai_root,patch_len=args.patch_len,stride=args.stride,n_layers=args.layers,n_heads=args.heads,d_model=args.d_model,d_ff=args.d_ff,dropout=args.dropout).to(device)
    model.train(); good=max(1,candidate//2); peak=0; low=good; high=None
    def fits(batch):
        nonlocal peak
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        try:
            x=torch.randn(batch,3,cfg.n_points,device=device); y=torch.zeros(batch,dtype=torch.long,device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                out=model(x); loss=nn.functional.cross_entropy(out,y)
            loss.backward()
            del x,y,out,loss; torch.cuda.synchronize(device)
            free,total=torch.cuda.mem_get_info(device); peak=max(peak,total-free)
            return free / total >= (1.0 - args.memory_target)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower(): raise
            torch.cuda.empty_cache(); return False
    while candidate <= args.auto_batch_max:
        if not fits(candidate):
            high=candidate; break
        low=good=candidate; candidate*=2
    if high is None: high=min(args.auto_batch_max, max(low+1, candidate))
    while high-low > 1:
        middle=(low+high)//2
        if fits(middle): low=good=middle
        else: high=middle
    del model; torch.cuda.empty_cache()
    logger.info("GPU {} 自动 batch={}，探测峰值显存占用约 {:.1f}%", device.index, good, 100*peak/max(1,torch.cuda.get_device_properties(device).total_memory))
    return good

def run(args, test_batch, meta, device, experiment_root, experiment_meta):
    cfg=Config(); cfg.cache_path=str(Path(args.cache).resolve()); cfg.data_root=str(Path(args.data_root or meta["data_root"]).resolve()); cfg.output_root=str(experiment_root)
    cfg.epochs=args.epochs; cfg.workers=args.workers; cfg.lr=args.lr; cfg.patience=args.patience; cfg.sampler=args.sampler; cfg.batch_shift=args.batch_shift; cfg.augmentation_strength=args.augmentation_strength; cfg.n_points=meta["n_points"]; cfg.n_classes=len(meta["labels"]); cfg.n_batches=len(meta["batch_values"]); cfg.labels=meta["labels"]; cfg.batch_values=meta["batch_values"]
    cfg.batch_size=probe_batch(cfg,device,args)
    idx,val_batch=make_splits(meta,test_batch,args.val_batch); seed_all(cfg.seed+test_batch)
    idx={k:limit_per_class(v,meta["y"],args.limit_per_class,cfg.seed+i) for i,(k,v) in enumerate(idx.items())}
    out=Path(cfg.output_root)/f"fold{test_batch}"; out.mkdir(parents=True,exist_ok=True)
    if (out/"best.pt").exists(): raise FileExistsError(f"{out} 已有 best.pt，请换 --output")
    if main_process():
        config_payload = {**asdict(cfg), "patch_len": args.patch_len, "stride": args.stride,
                          "layers": args.layers, "heads": args.heads, "d_model": args.d_model,
                          "d_ff": args.d_ff, "dropout": args.dropout, "tsai_root": args.tsai_root,
                          "sampler": args.sampler, "batch_shift": args.batch_shift,
                          "augmentation_strength": args.augmentation_strength}
        save_json(out / "config.json", config_payload)
        np.savez_compressed(out/"splits.npz", **idx)
        save_json(out/"split.json", {"test_batch": test_batch, "val_batch": val_batch,
                                     "counts": {k: len(v) for k, v in idx.items()},
                                     "cache_fingerprint": meta["fingerprint"],
                                     "batch_size_per_gpu": cfg.batch_size, "world_size": world(),
                                     "sampler": args.sampler, "batch_shift": args.batch_shift})
    loaders={k:make_loader(cfg.cache_path, v, cfg, meta["y"], meta["batch"],
                           k == "train", device, args) for k, v in idx.items()}
    model=PatchTSTClassifier(cfg.n_classes,cfg.n_points,args.tsai_root,patch_len=args.patch_len,stride=args.stride,n_layers=args.layers,n_heads=args.heads,d_model=args.d_model,d_ff=args.d_ff,dropout=args.dropout).to(device)
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(Path(cfg.output_root) / "tensorboard" /
                                       f"{experiment_meta['timestamp']}+{experiment_meta['commit'] or 'debug'}" /
                                       f"fold{test_batch}"))
        except ModuleNotFoundError:
            logger.warning("未安装 tensorboard，请运行 pip install tensorboard")
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1; stale=0; history=[]
    for epoch in range(1,cfg.epochs+1):
        model.train(); total=0.; seen=0
        it=tqdm(loaders["train"],desc=f"PatchTST fold {test_batch} epoch {epoch}/{cfg.epochs}",disable=not main_process(),leave=False)
        for x,y,_ in it:
            x=x.to(device,non_blocking=True); y=y.to(device); opt.zero_grad(set_to_none=True)
            for g in opt.param_groups: g["lr"]=lr_at(cfg,epoch)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()): loss=nn.functional.cross_entropy(model(x),y,label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); total+=float(loss)*len(y); seen+=len(y)
        train_loss = reduce_train_loss(total, seen, device)
        val=gather_eval(model,loaders["val"],device,cfg.labels)
        if main_process():
            current_lr = lr_at(cfg, epoch)
            val.update(epoch=epoch, loss=train_loss, train_loss=train_loss,
                       val_loss=val["val_loss"], lr=current_lr, learning_rate=current_lr)
            history.append(val)
            logger.info("fold={} epoch={}/{} train_loss={:.5f} val_loss={:.5f} acc={:.4f} bacc={:.4f} macro_f1={:.4f} lr={:.3e}",
                        test_batch, epoch, cfg.epochs, train_loss, val["val_loss"], val["acc"], val["bacc"], val["macro_f1"], current_lr)
            state=model.state_dict(); ck={"schema_version":SCHEMA_VERSION,"model":state,"config":{**asdict(cfg),"patch_len":args.patch_len,"stride":args.stride,"layers":args.layers,"heads":args.heads,"d_model":args.d_model,"d_ff":args.d_ff,"dropout":args.dropout,"tsai_root":args.tsai_root,"sampler":args.sampler,"batch_shift":args.batch_shift,"augmentation_strength":args.augmentation_strength},"labels":cfg.labels,"batch_values":cfg.batch_values,"wave":meta["wave"].tolist(),"preprocess":meta["preprocess"],"cache_fingerprint":meta["fingerprint"],"test_batch":test_batch,"val_batch":val_batch,"test_indices":idx["test"].tolist(),"limit_per_class":args.limit_per_class}
            if val["macro_f1"]>best: best=val["macro_f1"]; stale=0; torch.save(ck,out/"best.pt")
            else: stale+=1
            save_json(out/"history.json",history)
            save_history_artifacts(history, out, cfg.labels)
        if stale >= cfg.patience:
            break
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch); writer.add_scalar("loss/validation", val["val_loss"], epoch)
            writer.add_scalar("metrics/accuracy", val["acc"], epoch); writer.add_scalar("metrics/balanced_accuracy", val["bacc"], epoch); writer.add_scalar("metrics/macro_f1", val["macro_f1"], epoch)
            writer.add_scalar("metrics/macro_precision", val["report"]["macro avg"]["precision"], epoch); writer.add_scalar("metrics/macro_recall", val["report"]["macro avg"]["recall"], epoch); writer.add_scalar("optimization/learning_rate", current_lr, epoch)
            for label in cfg.labels:
                writer.add_scalar(f"class/{label}/precision", val["report"][label]["precision"], epoch); writer.add_scalar(f"class/{label}/recall", val["report"][label]["recall"], epoch); writer.add_scalar(f"class/{label}/f1", val["report"][label]["f1-score"], epoch)
            writer.flush()
    ck=torch.load(out/"best.pt",map_location=device,weights_only=True)
    model.load_state_dict(ck["model"])
    result=gather_eval(model,loaders["test"],device,cfg.labels)
    if main_process():
        result["test_loss"] = result.pop("val_loss")
        result.update(test_batch=test_batch,val_batch=val_batch,best_val_f1=best); save_json(out/"test_metrics.json",result); print(json.dumps({k:v for k,v in result.items() if k not in ("cm","report")},ensure_ascii=False))
    if writer: writer.close()
    return result

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(); g.add_argument("--fold",type=int); g.add_argument("--all-folds",action="store_true"); p.add_argument("--val-batch",type=int); p.add_argument("--data-root",default=Config().data_root); p.add_argument("--cache",default=Config().cache_path); p.add_argument("--output-root",help="实验父目录，实验名会自动追加时间戳"); p.add_argument("--epochs",type=int,default=300); p.add_argument("--batch-size",type=int); p.add_argument("--workers",type=int,default=4); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--patience",type=int,default=40); p.add_argument("--tsai-root",default="/home/wjx/CodeData/code/tsai-main"); p.add_argument("--patch-len",type=int,default=32); p.add_argument("--stride",type=int,default=16); p.add_argument("--layers",type=int,default=3); p.add_argument("--heads",type=int,default=8); p.add_argument("--d-model",type=int,default=128); p.add_argument("--d-ff",type=int,default=256); p.add_argument("--dropout",type=float,default=.1); p.add_argument("--limit-per-class",type=int); p.add_argument("--sampler",choices=["joint","class","uniform"],default="joint",help="训练采样：联合类别-批次、仅类别或均匀"); p.add_argument("--batch-shift",action=argparse.BooleanOptionalAction,default=True,help="启用批次增益/基线漂移增强"); p.add_argument("--augmentation-strength",type=float,default=1.0); p.add_argument("--auto-batch-start",type=int,default=8); p.add_argument("--auto-batch-max",type=int,default=4096); p.add_argument("--memory-target",type=float,default=.92); p.add_argument("--gpu-id",type=int); p.add_argument("--tensorboard",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--debug",action=argparse.BooleanOptionalAction,default=False,help="debug 模式写入 debug/，不执行 git commit"); p.add_argument("--commit-id",help="复用已提交的 commit，适合并行正式实验"); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    a=p.parse_args(); os.environ["PATCHTST_DEVICE"]=a.device; device=select_gpu(a); setup_logger(); experiment_root, experiment_meta=prepare_experiment(a); logger.info("实验={} debug={} commit={} device={} epochs={} data={}", experiment_meta["experiment_id"], experiment_meta["debug"], experiment_meta["commit"] or "none", device, a.epochs, a.data_root); meta=load_meta(a.cache,check_sources=True,data_root=a.data_root); folds=meta["batch_values"] if a.all_folds else [a.fold or meta["batch_values"][0]]
    for fold in folds: make_splits(meta,fold,a.val_batch)
    fold_results = {fold: run(a, fold, meta, device, experiment_root, experiment_meta) for fold in folds}
    append_results_csv(fold_results, meta["labels"], experiment_meta)
    cleanup_ddp()
if __name__=="__main__": main()
