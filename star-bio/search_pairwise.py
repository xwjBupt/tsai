"""One-vs-one RBF ridge ensemble for cross-batch bacterial spectra."""
import argparse, json, time
from pathlib import Path
import h5py, numpy as np, torch
from metadata import load_meta
from metrics import classification_metrics
from search_domain import row_snv
from splits import make_splits


def features(x, centers, gamma):
    d = torch.cdist(x, centers).square(); denom = torch.cdist(centers[:256], centers[:256]).square().median().clamp_min(1e-6)
    return torch.cat([torch.exp(-gamma * d / denom), torch.ones(len(x), 1, device=x.device)], 1)


def pair_scores(phi, qphi, y, n_cls, alpha):
    scores = torch.zeros(len(qphi), n_cls, device=phi.device)
    for c in range(n_cls):
        for d in range(c + 1, n_cls):
            mask = (y == c) | (y == d)
            a, yy = phi[mask], (y[mask] == c).float() * 2 - 1
            cov = a.T @ a; rhs = a.T @ yy
            coef = torch.linalg.solve(cov + alpha * cov.diag().mean().clamp_min(1e-6) * torch.eye(cov.shape[0], device=phi.device), rhs)
            score = qphi @ coef
            prob = torch.sigmoid(score)
            scores[:, c] += prob
            scores[:, d] += 1 - prob
    return scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--centers',type=int,default=1024);p.add_argument('--cache',default='/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5');p.add_argument('--data-root',default='/home/wjx/CodeData/data/Star-Com/7class-4patch');p.add_argument('--feature-path',default='/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy');a=p.parse_args()
    meta=load_meta(a.cache,True,a.data_root);y=meta['y'];labels=meta['labels'];n_cls=len(labels)
    with h5py.File(a.cache,'r') as h: batch=h['batch'][:]
    x=np.asarray(np.load(a.feature_path,mmap_mode='r')[:,0,:],dtype='float32');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in meta['batch_values']:
        t0=time.perf_counter();idx,val_batch=make_splits(meta,fold);tr,va,te=idx['train'],idx['val'],idx['test'];train=row_snv(x[tr]);val=row_snv(x[va]);test=row_snv(x[te]);mu=train.mean(0);train-=mu;val-=mu;test-=mu
        tx=torch.tensor(train,device=device);ty=torch.tensor(y[tr],device=device);vx=torch.tensor(val,device=device);ex=torch.tensor(test,device=device);rng=np.random.default_rng(3407+fold);centers=tx[rng.choice(len(tx),min(a.centers,len(tx)),replace=False)]
        best=None
        for gamma in (.03,.1,.3,1.,3.):
            phi=features(tx,centers,gamma);pv=features(vx,centers,gamma)
            for alpha in (1e-5,1e-4,1e-3,.01):
                sv=pair_scores(phi,pv,ty,n_cls,alpha);pred=sv.argmax(1).cpu().numpy();score=classification_metrics(y[va].tolist(),pred.tolist(),labels)['macro_f1']
                if best is None or score>best[0]:best=(score,gamma,alpha)
        phi=features(tx,centers,best[1]);pe=features(ex,centers,best[1]);scores=pair_scores(phi,pe,ty,n_cls,best[2]);pred=scores.argmax(1).cpu().numpy();m=classification_metrics(y[te].tolist(),pred.tolist(),labels);m.update(test_batch=fold,val_batch=val_batch,best_val_f1=best[0],gamma=best[1],alpha=best[2],elapsed_seconds=time.perf_counter()-t0);fd=out/f'fold{fold}';fd.mkdir(exist_ok=True);(fd/'test_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));rows.append(m);print(fold,m['macro_f1'],flush=True)
    avg={k:float(np.mean([m[k] for m in rows])) for k in ('acc','bacc','macro_f1')};(out/'summary.json').write_text(json.dumps({'folds':rows,'AVG':avg},ensure_ascii=False,indent=2));print('SUMMARY',avg,flush=True)
if __name__=='__main__':main()
