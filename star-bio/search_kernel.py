"""Train-only RBF landmark features with validation-selected bandwidth/ridge."""
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
    if spec.get('channel') is not None:xx=xx[:,spec['channel']:spec['channel']+1,:]
    x=torch.from_numpy(xx.reshape(len(xx),-1).copy()).to('cuda',torch.float64)
    y=torch.from_numpy(meta['y']).cuda();labels=meta['labels'];n=len(labels);results=[]
    for fold in meta['batch_values']:
        start=time.perf_counter();idx,valbatch=make_splits(meta,fold);folder=out/f'fold{fold}';folder.mkdir();torch.manual_seed(spec.get('seed',3407)+fold)
        tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda')
        tx=x[tr];ty=y[tr];mean=tx.mean(0);std=tx.std(0).clamp_min(1e-5) if spec.get('standardize',False) else torch.ones_like(mean)
        tx=(tx-mean)/std;vx=(x[va]-mean)/std
        keys=meta['y'][idx['train']]*len(meta['batch_values'])+meta['batch'][idx['train']];counts=np.bincount(keys)
        w=torch.tensor(1/counts[keys],device='cuda');w=w/w.sum()
        centers=tx[torch.multinomial(w,min(spec.get('centers',1024),len(tr)),replacement=False)]
        denom=torch.cdist(centers[:256],centers[:256]).square().median().clamp_min(1e-6)
        dt=torch.cdist(tx,centers).square()/denom;dv=torch.cdist(vx,centers).square()/denom
        best=-1;history=[]
        for gamma in spec.get('gammas',[.1,.3,1.,3.,10.]):
            z=torch.exp(-gamma*dt);vz=torch.exp(-gamma*dv)
            if spec.get('linear',False):
                z=torch.cat([z,tx/tx.shape[1]**.5],1);vz=torch.cat([vz,vx/vx.shape[1]**.5],1)
            z=torch.cat([z,torch.ones(len(z),1,device='cuda',dtype=x.dtype)],1);vz=torch.cat([vz,torch.ones(len(vz),1,device='cuda',dtype=x.dtype)],1)
            cov=(z.T*w)@z;rhs=(z.T*w)@torch.nn.functional.one_hot(ty,n).to(x.dtype)
            eigen,u=torch.linalg.eigh(cov);ur=u.T@rhs
            for alpha in [1e-6,1e-5,1e-4,.001,.01,.1,1.]:
                coef=u@(ur/(eigen.clamp_min(0)+alpha).unsqueeze(1));scores=vz@coef
                m=classification_metrics(y[va].cpu().tolist(),scores.argmax(1).cpu().tolist(),labels)
                history.append({'gamma':gamma,'alpha':alpha,'macro_f1':m['macro_f1']})
                if m['macro_f1']>best:best=m['macro_f1'];bestcoef=coef;bestgamma=gamma;bestalpha=alpha
        ex=(x[te]-mean)/std;z=torch.exp(-bestgamma*torch.cdist(ex,centers).square()/denom)
        if spec.get('linear',False):z=torch.cat([z,ex/ex.shape[1]**.5],1)
        z=torch.cat([z,torch.ones(len(z),1,device='cuda',dtype=x.dtype)],1);scores=z@bestcoef
        m=classification_metrics(y[te].cpu().tolist(),scores.argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=valbatch,best_val_f1=best,best_alpha=bestalpha,best_gamma=bestgamma,elapsed_seconds=time.perf_counter()-start)
        (folder/'test_metrics.json').write_text(json.dumps(m,indent=2));(folder/'history.json').write_text(json.dumps(history,indent=2));np.savez_compressed(folder/'splits.npz',**idx)
        torch.save({'mean':mean.cpu(),'std':std.cpu(),'centers':centers.cpu(),'denom':denom.cpu(),'coef':bestcoef.cpu(),'spec':spec,'alpha':bestalpha,'gamma':bestgamma,'labels':labels},folder/'best.pt')
        results.append(m);print('TEST',fold,m['macro_f1'],'VAL',best,flush=True)
    summary={'spec':spec,'folds':results,'AVG':{k:float(np.mean([m[k] for m in results])) for k in ['acc','bacc','macro_f1']}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    append_results_csv({m['test_batch']:m for m in results},labels,{'experiment_id':spec['name'],'timestamp':spec['timestamp'],'commit':spec['commit'],'debug':False})
    print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
