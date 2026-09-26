"""Evaluate a PatchTST checkpoint on its recorded held-out batch."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpectraDataset
from metadata import load_meta
from metrics import evaluate
from patchtst_model import PatchTSTClassifier


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--cache'); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--tsai-root',default='/home/wjx/CodeData/code/tsai-main')
    a=p.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); c=ck['config']; cache=a.cache or c['cache_path']
    meta=load_meta(cache,check_sources=False)
    if ck['cache_fingerprint'] != meta['fingerprint']: raise ValueError('缓存与 checkpoint 不匹配')
    idx=np.asarray(ck['test_indices'],dtype=np.int64); ds=SpectraDataset(cache,idx); dl=DataLoader(ds,batch_size=a.batch_size)
    dev=torch.device(a.device if a.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    m=PatchTSTClassifier(c['n_classes'],c['n_points'],c.get('tsai_root',a.tsai_root),patch_len=c['patch_len'],stride=c['stride'],n_layers=c['layers'],n_heads=c['heads'],d_model=c['d_model'],d_ff=c['d_ff'],dropout=c['dropout']).to(dev); m.load_state_dict(ck['model'])
    r=evaluate(m,dl,dev,ck['labels']); print({'test_batch':ck['test_batch'],'acc':r['acc'],'bacc':r['bacc'],'macro_f1':r['macro_f1'],'cm':r['cm']}); ds.close()
if __name__=='__main__': main()
