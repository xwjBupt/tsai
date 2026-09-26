"""Train tsai PatchTST on the Cell cache with torchrun/DDP.

Example:
  torchrun --standalone --nproc_per_node=8 train_patchtst.py --all-folds
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, Sampler
from tqdm import tqdm

from config import Config
from dataset import SpectraDataset
from metadata import load_meta
from metrics import classification_metrics
from prepare_data import SCHEMA_VERSION
from splits import make_splits, limit_per_class
from patchtst_model import PatchTSTClassifier


def is_dist(): return dist.is_available() and dist.is_initialized()
def rank(): return dist.get_rank() if is_dist() else 0
def world(): return dist.get_world_size() if is_dist() else 1
def main_process(): return rank() == 0

def setup_ddp():
    n = int(os.environ.get("WORLD_SIZE", "1"))
    if n > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        backend = os.environ.get("PATCHTST_BACKEND", "nccl")
        if backend == "nccl":
            # Avoid common single-node P2P/IB initialization failures. The
            # setting can be overridden with PATCHTST_NCCL_P2P=1.
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
        try:
            dist.init_process_group(backend, device_id=torch.device("cuda", local))
        except RuntimeError as exc:
            if backend == "nccl" and ("driver version" in str(exc).lower() or "nccl" in str(exc).lower()):
                raise RuntimeError(
                    "NCCL 初始化失败。请确认 torch CUDA 版本与 NVIDIA 驱动匹配；"
                    "可先用 PATCHTST_BACKEND=gloo 诊断，或安装支持 CUDA 12.4 的驱动。"
                ) from exc
            raise
        return torch.device("cuda", local)
    requested = os.environ.get("PATCHTST_DEVICE", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_ddp():
    if is_dist(): dist.destroy_process_group()
def seed_all(seed):
    random.seed(seed + rank()); np.random.seed(seed + rank()); torch.manual_seed(seed + rank())

def save_json(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def lr_at(cfg, epoch):
    warm = min(cfg.warmup_epochs, max(0, cfg.epochs - 1))
    if warm and epoch <= warm: return cfg.lr * epoch / warm
    p = (epoch - warm - 1) / max(1, cfg.epochs - warm - 1)
    return cfg.lr * (.01 + .99 * (1 + math.cos(math.pi * p)) / 2)

def make_loader(cache, idx, cfg, labels, train, device):
    ds = SpectraDataset(cache, idx, train)
    if is_dist() and train:
        sampler = DistributedSampler(ds, shuffle=True, seed=cfg.seed, drop_last=False)
    elif is_dist():
        sampler = DistributedEvalSampler(ds)
    elif train:
        count = np.bincount(labels[idx], minlength=cfg.n_classes)
        sampler = WeightedRandomSampler(torch.as_tensor((1 / np.maximum(count, 1))[labels[idx]], dtype=torch.double), len(idx), replacement=True)
    else: sampler = None
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None and train),
                      num_workers=cfg.workers, pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0)


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices without padding or duplicate samples."""
    def __init__(self, dataset):
        self.length = len(dataset)
        self.rank = rank()
        self.world = world()

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world))

    def __len__(self):
        return max(0, (self.length - self.rank + self.world - 1) // self.world)

def gather_eval(model, loader, device, labels):
    model.eval(); ys=[]; ps=[]
    eval_model = model.module if isinstance(model, DDP) else model
    with torch.inference_mode():
        for x,y,_ in loader:
            ps.extend(eval_model(x.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    if is_dist():
        payload=[ys,ps]; all_payload=[None]*world(); dist.all_gather_object(all_payload,payload)
        ys=sum((p[0] for p in all_payload), []); ps=sum((p[1] for p in all_payload), [])
    return classification_metrics(ys,ps,labels)

def probe_batch(cfg, device, args):
    if args.batch_size is not None: return args.batch_size
    if device.type != "cuda": return Config().batch_size
    candidate = max(1, 2 ** int(math.log2(args.auto_batch_start)))
    model = PatchTSTClassifier(cfg.n_classes,cfg.n_points,args.tsai_root,patch_len=args.patch_len,stride=args.stride,n_layers=args.layers,n_heads=args.heads,d_model=args.d_model,d_ff=args.d_ff,dropout=args.dropout).to(device)
    model.train(); good=max(1,candidate//2)
    while candidate <= args.auto_batch_max:
        try:
            torch.cuda.empty_cache(); x=torch.randn(candidate,3,cfg.n_points,device=device); y=torch.zeros(candidate,dtype=torch.long,device=device)
            out=model(x); loss=nn.functional.cross_entropy(out,y); loss.backward(); del x,y,out,loss
            torch.cuda.synchronize(device); good=candidate; candidate*=2
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            torch.cuda.empty_cache(); break
    del model; torch.cuda.empty_cache(); return good

def run(args, test_batch, meta, device):
    cfg=Config(); cfg.cache_path=str(Path(args.cache).resolve()); cfg.data_root=str(Path(args.data_root or meta["data_root"]).resolve()); cfg.output_root=str(Path(args.output or Config().output_root).resolve())
    cfg.epochs=args.epochs; cfg.workers=args.workers; cfg.lr=args.lr; cfg.patience=args.patience; cfg.n_points=meta["n_points"]; cfg.n_classes=len(meta["labels"]); cfg.n_batches=len(meta["batch_values"]); cfg.labels=meta["labels"]; cfg.batch_values=meta["batch_values"]
    cfg.batch_size=probe_batch(cfg,device,args)
    if is_dist():
        # DDP needs every rank to execute the same number of training steps.
        batch_tensor = torch.tensor(cfg.batch_size, device=device, dtype=torch.int64)
        dist.all_reduce(batch_tensor, op=dist.ReduceOp.MIN)
        cfg.batch_size = int(batch_tensor.item())
    idx,val_batch=make_splits(meta,test_batch,args.val_batch); seed_all(cfg.seed+test_batch)
    idx={k:limit_per_class(v,meta["y"],args.limit_per_class,cfg.seed+i) for i,(k,v) in enumerate(idx.items())}
    out=Path(cfg.output_root)/f"fold{test_batch}"; out.mkdir(parents=True,exist_ok=True)
    if (out/"best.pt").exists(): raise FileExistsError(f"{out} 已有 best.pt，请换 --output")
    if main_process():
        cfg.save(out/"config.json"); np.savez_compressed(out/"splits.npz",**idx); save_json(out/"split.json",{"test_batch":test_batch,"val_batch":val_batch,"counts":{k:len(v) for k,v in idx.items()},"cache_fingerprint":meta["fingerprint"],"batch_size_per_gpu":cfg.batch_size,"world_size":world()})
    if is_dist(): dist.barrier()
    loaders={k:make_loader(cfg.cache_path,v,cfg,meta["y"],k=="train",device) for k,v in idx.items()}
    model=PatchTSTClassifier(cfg.n_classes,cfg.n_points,args.tsai_root,patch_len=args.patch_len,stride=args.stride,n_layers=args.layers,n_heads=args.heads,d_model=args.d_model,d_ff=args.d_ff,dropout=args.dropout).to(device)
    if is_dist(): model=DDP(model,device_ids=[device.index],find_unused_parameters=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1; stale=0; history=[]
    for epoch in range(1,cfg.epochs+1):
        if isinstance(loaders["train"].sampler,DistributedSampler): loaders["train"].sampler.set_epoch(epoch)
        model.train(); total=0.; seen=0
        it=tqdm(loaders["train"],desc=f"PatchTST fold {test_batch} epoch {epoch}/{cfg.epochs}",disable=not main_process(),leave=False)
        for x,y,_ in it:
            x=x.to(device,non_blocking=True); y=y.to(device); opt.zero_grad(set_to_none=True)
            for g in opt.param_groups: g["lr"]=lr_at(cfg,epoch)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()): loss=nn.functional.cross_entropy(model(x),y,label_smoothing=.05)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); total+=float(loss)*len(y); seen+=len(y)
        val=gather_eval(model,loaders["val"],device,cfg.labels)
        if main_process():
            val.update(epoch=epoch,loss=total/max(1,seen),lr=lr_at(cfg,epoch)); history.append(val); print(json.dumps({k:v for k,v in val.items() if k not in ("cm","report")},ensure_ascii=False),flush=True)
            state=model.module.state_dict() if is_dist() else model.state_dict(); ck={"schema_version":SCHEMA_VERSION,"model":state,"config":{**asdict(cfg),"patch_len":args.patch_len,"stride":args.stride,"layers":args.layers,"heads":args.heads,"d_model":args.d_model,"d_ff":args.d_ff,"dropout":args.dropout,"tsai_root":args.tsai_root},"labels":cfg.labels,"batch_values":cfg.batch_values,"wave":meta["wave"].tolist(),"preprocess":meta["preprocess"],"cache_fingerprint":meta["fingerprint"],"test_batch":test_batch,"val_batch":val_batch,"test_indices":idx["test"].tolist(),"limit_per_class":args.limit_per_class}
            if val["macro_f1"]>best: best=val["macro_f1"]; stale=0; torch.save(ck,out/"best.pt")
            else: stale+=1
            save_json(out/"history.json",history)
        # Rank 0 decides early stopping; every process must receive the same
        # decision before entering the next epoch/barrier.
        stop = torch.tensor([int(main_process() and stale >= cfg.patience)], device=device)
        if is_dist(): dist.broadcast(stop, src=0)
        if bool(stop.item()): break
        if is_dist(): dist.barrier()
    if is_dist(): dist.barrier()
    ck=torch.load(out/"best.pt",map_location=device,weights_only=True)
    (model.module if is_dist() else model).load_state_dict(ck["model"])
    result=gather_eval(model,loaders["test"],device,cfg.labels)
    if main_process():
        result.update(test_batch=test_batch,val_batch=val_batch,best_val_f1=best); save_json(out/"test_metrics.json",result); print(json.dumps({k:v for k,v in result.items() if k not in ("cm","report")},ensure_ascii=False))
    if is_dist(): dist.barrier()

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(); g.add_argument("--fold",type=int); g.add_argument("--all-folds",action="store_true"); p.add_argument("--val-batch",type=int); p.add_argument("--data-root",default=Config().data_root); p.add_argument("--cache",default=Config().cache_path); p.add_argument("--output",default=Config().output_root); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int); p.add_argument("--workers",type=int,default=4); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--patience",type=int,default=20); p.add_argument("--tsai-root",default="/home/wjx/CodeData/code/tsai-main"); p.add_argument("--patch-len",type=int,default=32); p.add_argument("--stride",type=int,default=16); p.add_argument("--layers",type=int,default=3); p.add_argument("--heads",type=int,default=8); p.add_argument("--d-model",type=int,default=128); p.add_argument("--d-ff",type=int,default=256); p.add_argument("--dropout",type=float,default=.1); p.add_argument("--limit-per-class",type=int); p.add_argument("--auto-batch-start",type=int,default=8); p.add_argument("--auto-batch-max",type=int,default=128); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", help="DDP 后端；GPU 推荐 nccl，诊断可用 gloo")
    a=p.parse_args(); os.environ["PATCHTST_DEVICE"]=a.device; os.environ["PATCHTST_BACKEND"]=a.backend; device=setup_ddp(); meta=load_meta(a.cache,check_sources=True,data_root=a.data_root); folds=meta["batch_values"] if a.all_folds else [a.fold or meta["batch_values"][0]]
    for fold in folds: make_splits(meta,fold,a.val_batch)
    for fold in folds: run(a,fold,meta,device)
    cleanup_ddp()
if __name__=="__main__": main()
