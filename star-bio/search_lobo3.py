"""Standard four-fold leave-one-batch-out RBF search.

One entire batch is test. The other three batches are training data; a small
stratified subset of those three batches is used only for hyperparameter
selection. No sample or statistic from the test batch is used before scoring.
"""
import argparse, json, time
from pathlib import Path
import h5py, numpy as np
from metadata import load_meta
from metrics import classification_metrics
from search_domain import rbf_scores, row_snv, fast_macro_f1


def split_train_val(y, batch, train_idx, fraction, seed):
    rng = np.random.default_rng(seed); val=[]; tr=[]
    keys = sorted(set(zip(y[train_idx].tolist(), batch[train_idx].tolist())))
    for c,b in keys:
        ids = train_idx[(y[train_idx] == c) & (batch[train_idx] == b)]
        n = max(1, int(round(len(ids) * fraction)))
        choose = rng.choice(ids, min(n, len(ids)), replace=False)
        val.extend(choose.tolist()); mask=np.ones(len(ids),dtype=bool); mask[np.isin(ids,choose)] = False; tr.extend(ids[mask].tolist())
    return np.asarray(sorted(tr),dtype=np.int64), np.asarray(sorted(val),dtype=np.int64)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--feature-path',default='/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy');p.add_argument('--channels',default='0');p.add_argument('--centers',type=int,default=4096);p.add_argument('--val-fraction',type=float,default=.15);p.add_argument('--seed',type=int,default=3407);p.add_argument('--cache',default='/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5');p.add_argument('--data-root',default='/home/wjx/CodeData/data/Star-Com/7class-4patch');a=p.parse_args()
    meta=load_meta(a.cache,True,a.data_root);y=meta['y'];labels=meta['labels'];channels=[int(i) for i in a.channels.split(',')]
    with h5py.File(a.cache,'r') as h: batch=h['batch'][:]
    arr=np.load(a.feature_path,mmap_mode='r');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    # Build the requested representation without mutating the memmap.
    x=np.asarray(arr[:,channels,:],dtype='float32').reshape(len(arr),-1).copy()
    x=row_snv(x)
    for fold in meta['batch_values']:
        t0=time.perf_counter(); fold_id=meta['batch_values'].index(fold); test=np.flatnonzero(batch==fold_id); observed=np.flatnonzero(batch!=fold_id); tr,va=split_train_val(y,batch,observed,a.val_fraction,a.seed+fold)
        tx=x[tr];vx=x[va];ex=x[test];mu=tx.mean(0);tx-=mu;vx-=mu;ex-=mu
        keys=y[tr]*len(meta['batch_values'])+batch[tr];counts=np.bincount(keys);weights=1/np.maximum(counts[keys],1);rng=np.random.default_rng(a.seed+fold);centers=tx[rng.choice(len(tx),min(a.centers,len(tx)),replace=False,p=weights/weights.sum())]
        best=None
        for gamma in (.01,.03,.1,.3,1.,3.):
            for alpha in (1e-6,1e-5,1e-4,1e-3,.01):
                sv=rbf_scores(tx,y[tr],vx,centers,gamma,alpha,weights);score=fast_macro_f1(y[va],sv.argmax(1),len(labels))
                if best is None or score>best[0]:best=(score,gamma,alpha)
        st=rbf_scores(tx,y[tr],ex,centers,best[1],best[2],weights);m=classification_metrics(y[test].tolist(),st.argmax(1).tolist(),labels);m.update(test_batch=fold,val_count=len(va),train_count=len(tr),best_val_f1=best[0],gamma=best[1],alpha=best[2],channels=channels,elapsed_seconds=time.perf_counter()-t0);fd=out/f'fold{fold}';fd.mkdir(exist_ok=True);(fd/'test_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));rows.append(m);print(json.dumps({k:v for k,v in m.items() if k not in ('cm','report')},ensure_ascii=False),flush=True)
    avg={k:float(np.mean([m[k] for m in rows])) for k in ('acc','bacc','macro_f1')};(out/'summary.json').write_text(json.dumps({'folds':rows,'AVG':avg,'channels':channels,'val_fraction':a.val_fraction},ensure_ascii=False,indent=2));print('SUMMARY',avg,flush=True)
if __name__=='__main__':main()
