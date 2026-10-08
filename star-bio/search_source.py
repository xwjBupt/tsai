"""Select a single source-batch RBF expert using the validation batch."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from metrics import classification_metrics
from search_domain import rbf_scores,row_snv,fast_macro_f1
from splits import make_splits

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['raw','target_center','target_affine'],default='raw');p.add_argument('--centers',type=int,default=4096);p.add_argument('--cache',default='/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5');p.add_argument('--data-root',default='/home/wjx/CodeData/data/Star-Com/7class-4patch');p.add_argument('--feature-path',default='/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy');a=p.parse_args()
 meta=load_meta(a.cache,True,a.data_root)
 with h5py.File(a.cache,'r') as h:b=h['batch'][:]
 x=np.asarray(np.load(a.feature_path,mmap_mode='r')[:,0,:],dtype='float32').copy();y=meta['y'];labels=meta['labels'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,vb=make_splits(meta,fold);tr,va,te=idx['train'],idx['val'],idx['test'];X=row_snv(x);mu=X[tr].mean(0);X-=mu;best=None
  # Fit and evaluate each individual observed batch as an expert.
  for source in sorted(np.unique(b[tr])):
   si=tr[b[tr]==source];train=X[si];val=X[va];test=X[te]
   if a.mode=='target_center': train=train-train.mean(0)+val.mean(0); test=test
   elif a.mode=='target_affine': train=(train-train.mean(0))/(train.std(0)+1e-5)*(val.std(0)+1e-5)+val.mean(0)
   rng=np.random.default_rng(3407+fold+source);centers=train[rng.choice(len(train),min(a.centers,len(train)),replace=False)];weights=np.ones(len(si));
   for gamma in (.03,.1,.3,1.,3.):
    for alpha in (1e-6,1e-4,1e-2):
     sv=rbf_scores(train,y[si],val,centers,gamma,alpha,weights);score=fast_macro_f1(y[va],sv.argmax(1),len(labels))
     if best is None or score>best[0]:best=(score,source,gamma,alpha,centers)
  source,gamma,alpha,centers=best[1:];si=tr[b[tr]==source];train=X[si];test=X[te];
  if a.mode=='target_center':train=train-train.mean(0)+test.mean(0)
  elif a.mode=='target_affine':train=(train-train.mean(0))/(train.std(0)+1e-5)*(test.std(0)+1e-5)+test.mean(0)
  stc=rbf_scores(train,y[si],test,centers,gamma,alpha,np.ones(len(si)));m=classification_metrics(y[te].tolist(),stc.argmax(1).tolist(),labels);m.update(test_batch=fold,val_batch=vb,best_val_f1=best[0],source_batch=int(source),gamma=gamma,alpha=alpha,mode=a.mode,elapsed_seconds=time.perf_counter()-st);fd=out/f'fold{fold}';fd.mkdir(exist_ok=True);(fd/'test_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));rows.append(m);print(json.dumps({k:z for k,z in m.items() if k not in ('cm','report')},ensure_ascii=False),flush=True)
 avg={k:float(np.mean([m[k] for m in rows])) for k in ('acc','bacc','macro_f1')};(out/'summary.json').write_text(json.dumps({'mode':a.mode,'folds':rows,'AVG':avg},ensure_ascii=False,indent=2));print('SUMMARY',avg,flush=True)
if __name__=='__main__':main()
