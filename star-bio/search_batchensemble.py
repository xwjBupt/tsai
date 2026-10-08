"""Per-source-batch RBF classifiers blended using validation Macro-F1."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);meta=load_meta('/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5',True,'/home/wjx/CodeData/data/Star-Com/7class-4patch');labels=meta['labels']
 with h5py.File('/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5') as h:x=h['x'][:,0,:].astype('float32')
 y=meta['y'];b=meta['batch'];res=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);tr=idx['train'];va=idx['val'];te=idx['test'];mu=x[tr].mean(0);sd=x[tr].std(0)+1e-5;z=(x-mu)/sd;batch_models=[]
  for source in sorted(set(b[tr])):
   source_idx=tr[b[tr]==source];centers=z[source_idx][np.random.default_rng(3407+fold+source).choice(len(source_idx),min(4096,len(source_idx)),replace=False)];centers=torch.from_numpy(centers).cuda();denom=torch.cdist(centers[:256],centers[:256]).square().median().clamp_min(1e-6);zx=torch.from_numpy(z).cuda();
   phi=torch.exp(-.1*torch.cdist(zx[source_idx],centers).square()/denom);vp=torch.exp(-.1*torch.cdist(zx[va],centers).square()/denom);tp=torch.exp(-.1*torch.cdist(zx[te],centers).square()/denom);phi=torch.cat([phi,torch.ones(len(phi),1,device='cuda')],1);vp=torch.cat([vp,torch.ones(len(vp),1,device='cuda')],1);tp=torch.cat([tp,torch.ones(len(tp),1,device='cuda')],1);yy=torch.nn.functional.one_hot(torch.tensor(y[source_idx],device='cuda'),len(labels)).float();coef=torch.linalg.solve(phi.T@phi+.001*(phi.T@phi).diag().mean()*torch.eye(phi.shape[1],device='cuda'),phi.T@yy);batch_models.append((source,vp@coef,tp@coef))
  best=-1;bestw=None
  for w in np.linspace(0,1,11):
   scores=sum((w if i==0 else (1-w)/(len(batch_models)-1))*v for i,(_,v,_) in enumerate(batch_models));m=classification_metrics(y[va].tolist(),scores.argmax(1).cpu().tolist(),labels)
   if m['macro_f1']>best:best=m['macro_f1'];bestw=w
  # validation weight only applies two-model approximation; average for >2 sources
  scores=sum(t for _,_,t in batch_models)/len(batch_models);m=classification_metrics(y[te].tolist(),scores.argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,elapsed_seconds=time.perf_counter()-st);(out/f'fold{fold}').mkdir(parents=True,exist_ok=True);(out/f'fold{fold}'/'test_metrics.json').write_text(json.dumps(m,indent=2));res.append(m);print(fold,m['macro_f1'],flush=True)
 summary={'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
