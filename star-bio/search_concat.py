"""RBF ridge on concatenated train-only normalized spectral representations."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.loads(Path(a.spec).read_text());out=Path(a.out);meta=load_meta(s['cache'],True,s['data_root']);labels=meta['labels'];res=[];arrays=[np.load(q,mmap_mode='r') for q in s['feature_paths']]
 with h5py.File(s['cache']) as h:batch=h['batch'][:]
 y=torch.tensor(meta['y'],device='cuda')
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);fd=out/f'fold{fold}';fd.mkdir();tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda');x=torch.from_numpy(np.concatenate([a[:,0,:] for a in arrays],1).copy()).cuda().float();mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5);x=(x-mu)/sd;keys=meta['y'][idx['train']]*len(meta['batch_values'])+batch[idx['train']];counts=np.bincount(keys);w=torch.tensor(1/counts[keys],device='cuda');w=w/w.sum();best=-1
  for centers_n in [2048,4096,8192]:
   centers=x[tr][torch.multinomial(w,min(centers_n,len(tr)),replacement=False)];denom=torch.cdist(centers[:256],centers[:256]).square().median().clamp_min(1e-6)
   for gamma in [.01,.03,.1,.3,1.]:
    tp=torch.exp(-gamma*torch.cdist(x[tr],centers).square()/denom);vp=torch.exp(-gamma*torch.cdist(x[va],centers).square()/denom);tp=torch.cat([tp,torch.ones(len(tr),1,device='cuda')],1);vp=torch.cat([vp,torch.ones(len(va),1,device='cuda')],1);cov=tp.T@tp;rhs=tp.T@torch.nn.functional.one_hot(y[tr],len(labels)).float();coef=torch.linalg.solve(cov+.001*cov.diag().mean()*torch.eye(cov.shape[0],device='cuda'),rhs);m=classification_metrics(y[va].cpu().tolist(),(vp@coef).argmax(1).cpu().tolist(),labels)
    if m['macro_f1']>best:best=m['macro_f1'];bc=centers;bd=denom;bg=gamma;bcoef=coef
  ep=torch.exp(-bg*torch.cdist(x[te],bc).square()/bd);ep=torch.cat([ep,torch.ones(len(ep),1,device='cuda')],1);m=classification_metrics(y[te].cpu().tolist(),(ep@bcoef).argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,centers=bc.shape[0],gamma=bg,elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(m,indent=2));res.append(m);print('TEST',fold,m['macro_f1'],flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':s['name'],'timestamp':s['timestamp'],'commit':s['commit'],'debug':False});print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
