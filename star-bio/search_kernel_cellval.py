"""RBF kernel with cell-level validation inside the three non-test batches."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def cell_split(meta,test_batch,frac=.15,seed=3407):
 base,_=make_splits(meta,test_batch,test_batch-999 if False else None)
 # make_splits' validation batch is discarded; retain all non-test batches.
 test=np.flatnonzero(meta['batch']==meta['batch_values'].index(test_batch));remain=np.flatnonzero(meta['batch']!=meta['batch_values'].index(test_batch));rng=np.random.default_rng(seed+test_batch);val=[]
 for c in range(len(meta['labels'])):
  q=remain[meta['y'][remain]==c].copy();rng.shuffle(q);val.extend(q[:max(1,int(len(q)*frac))])
 val=np.asarray(val,dtype=np.int64);train=np.setdiff1d(remain,val)
 return {'train':train,'val':val,'test':test}, None

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.loads(Path(a.spec).read_text());out=Path(a.out);meta=load_meta(s['cache'],True,s['data_root']);labels=meta['labels'];res=[]
 with h5py.File(s['cache']) as h:arr=h['x'][:]
 x=torch.from_numpy(arr[:,s.get('channel',0),:].reshape(len(arr),-1).copy()).to('cuda',torch.float64);y=torch.tensor(meta['y'],device='cuda')
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=cell_split(meta,fold,s.get('val_fraction',.15),s.get('seed',3407));fd=out/f'fold{fold}';fd.mkdir();tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda');tx=x[tr];mean=tx.mean(0);std=tx.std(0).clamp_min(1e-5);z=(x-mean)/std;keys=meta['y'][idx['train']]*len(meta['batch_values'])+meta['batch'][idx['train']];counts=np.bincount(keys);w=torch.tensor(1/counts[keys],device='cuda');w=w/w.sum();centers=z[tr][torch.multinomial(w,min(s.get('centers',4096),len(tr)),replacement=False)];denom=torch.cdist(centers[:512],centers[:512]).square().median().clamp_min(1e-6);dt=torch.cdist(z[tr],centers).square()/denom;dv=torch.cdist(z[va],centers).square()/denom;best=-1
  for gamma in s.get('gammas',[.01,.03,.1,.3,1.]):
   phi=torch.exp(-gamma*dt);vp=torch.exp(-gamma*dv);phi=torch.cat([phi,z[tr]/z.shape[1]**.5,torch.ones(len(tr),1,device='cuda')],1);vp=torch.cat([vp,z[va]/z.shape[1]**.5,torch.ones(len(va),1,device='cuda')],1);cov=phi.T@phi;rhs=phi.T@torch.nn.functional.one_hot(y[tr],len(labels)).to(x.dtype);coef=torch.linalg.solve(cov+.001*cov.diag().mean()*torch.eye(cov.shape[0],device='cuda'),rhs);m=classification_metrics(y[va].cpu().tolist(),(vp@coef).argmax(1).cpu().tolist(),labels)
   if m['macro_f1']>best:best=m['macro_f1'];bg=gamma;bc=coef
  tp=torch.exp(-bg*torch.cdist(z[te],centers).square()/denom);tp=torch.cat([tp,z[te]/z.shape[1]**.5,torch.ones(len(te),1,device='cuda')],1);m=classification_metrics(y[te].cpu().tolist(),(tp@bc).argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch='cell_within_train',best_val_f1=best,gamma=bg,elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(m,indent=2));res.append(m);print('TEST',fold,m['macro_f1'],flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
