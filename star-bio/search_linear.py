"""Train-only ridge discriminants; regularization chosen on held-out validation batch."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch,h5py
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    spec=json.loads(Path(a.spec).read_text());out=Path(a.out);torch.set_num_threads(2)
    meta=load_meta(spec['cache'],True,spec['data_root'])
    if spec.get('feature_path'):
        fm=json.loads(Path(spec['feature_path']).with_suffix('.json').read_text());assert fm['fingerprint']==meta['fingerprint']
        xx=np.load(spec['feature_path'])
    else:
        with h5py.File(spec['cache']) as h:xx=h['x'][:]
    # Selected channel vs full representation is an explicit experiment choice.
    if spec.get('channel') is not None:xx=xx[:,spec['channel']:spec['channel']+1,:]
    if spec.get('downsample',1)>1:xx=xx[:,:,::spec['downsample']]
    x=torch.from_numpy(xx.reshape(len(xx),-1).copy()).to('cuda',torch.float64)
    y=torch.from_numpy(meta['y']).cuda();labels=meta['labels'];n=len(labels);results=[]
    for fold in meta['batch_values']:
        start=time.perf_counter();idx,valbatch=make_splits(meta,fold);folder=out/f'fold{fold}';folder.mkdir()
        tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda')
        tx=x[tr];ty=y[tr]
        mean=tx.mean(0);std=tx.std(0).clamp_min(1e-5) if spec.get('standardize',False) else torch.ones_like(mean)
        tx=(tx-mean)/std
        keys=meta['y'][idx['train']]*len(meta['batch_values'])+meta['batch'][idx['train']];counts=np.bincount(keys)
        w=torch.tensor(1/counts[keys],device='cuda');w=w/w.sum()
        if spec.get('method')=='lda':
            means=torch.stack([tx[ty==i].mean(0) for i in range(n)])
            centered=tx-means[ty];cov=(centered.T*w)@centered
            rhs=means.T
        else:
            tx=torch.cat([tx,torch.ones(len(tx),1,device='cuda',dtype=x.dtype)],1)
            cov=(tx.T*w)@tx;rhs=(tx.T*w)@torch.nn.functional.one_hot(ty,n).to(x.dtype)
        eye=torch.eye(cov.shape[0],device='cuda',dtype=x.dtype);scale=cov.diag().mean();best=-1;history=[];bestcoef=None
        for alpha in [1e-5,1e-4,1e-3,.01,.1,1.,10.,100.]:
            coef=torch.linalg.solve(cov+alpha*scale*eye,rhs)
            vx=(x[va]-mean)/std
            if spec.get('method')=='lda':
                intercept=-.5*(means*coef.T).sum(1);scores=vx@coef+intercept
            else:
                intercept=None;scores=torch.cat([vx,torch.ones(len(vx),1,device='cuda',dtype=x.dtype)],1)@coef
            m=classification_metrics(y[va].cpu().tolist(),scores.argmax(1).cpu().tolist(),labels)
            history.append({'alpha':alpha,'macro_f1':m['macro_f1']})
            if m['macro_f1']>best:best=m['macro_f1'];bestcoef=coef;bestint=intercept;bestalpha=alpha
        vx=(x[te]-mean)/std
        if spec.get('method')=='lda':scores=vx@bestcoef+bestint
        else:scores=torch.cat([vx,torch.ones(len(vx),1,device='cuda',dtype=x.dtype)],1)@bestcoef
        m=classification_metrics(y[te].cpu().tolist(),scores.argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=valbatch,best_val_f1=best,best_alpha=bestalpha,elapsed_seconds=time.perf_counter()-start)
        (folder/'test_metrics.json').write_text(json.dumps(m,indent=2));(folder/'history.json').write_text(json.dumps(history,indent=2))
        np.savez_compressed(folder/'splits.npz',**idx)
        torch.save({'mean':mean.cpu(),'std':std.cpu(),'coef':bestcoef.cpu(),'intercept':bestint.cpu() if bestint is not None else None,'spec':spec,'alpha':bestalpha,'labels':labels},folder/'best.pt')
        results.append(m);print('TEST',fold,m['macro_f1'],flush=True)
    summary={'spec':spec,'folds':results,'AVG':{k:float(np.mean([m[k] for m in results])) for k in ['acc','bacc','macro_f1']}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    append_results_csv({m['test_batch']:m for m in results},labels,{'experiment_id':spec['name'],'timestamp':spec['timestamp'],'commit':spec['commit'],'debug':False})
    print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
