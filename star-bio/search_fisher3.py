"""Three-source LOBO with train-only Fisher wavelength weighting."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np
from metadata import load_meta
from metrics import classification_metrics
from search_domain import rbf_scores,row_snv,fast_macro_f1
from search_lobo3 import split_train_val

def fisher_scale(x,y,n_cls,mode,topk):
    overall=x.mean(0); between=np.zeros(x.shape[1]); within=np.zeros(x.shape[1])
    for c in range(n_cls):
        z=x[y==c];between += len(z)*(z.mean(0)-overall)**2; within += ((z-z.mean(0))**2).sum(0)
    f=between/np.maximum(within,1e-8);f=f/np.median(f[f>0])
    if topk and topk < len(f):
        ids=np.argpartition(f,-topk)[-topk:];mask=np.zeros(len(f),bool);mask[ids]=True
    else: mask=np.ones(len(f),bool)
    if mode == 'sqrt': w=np.sqrt(np.clip(f,0,20))
    elif mode == 'quarter': w=np.sqrt(np.sqrt(np.clip(f,0,20)))
    elif mode == 'linear': w=np.clip(f,0,20)
    else: w=np.ones(len(f))
    return w*mask,mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--weight',choices=['none','quarter','sqrt','linear'],required=True);p.add_argument('--topk',type=int,default=1105);p.add_argument('--centers',type=int,default=4096);p.add_argument('--cache',default='/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5');p.add_argument('--data-root',default='/home/wjx/CodeData/data/Star-Com/7class-4patch');p.add_argument('--feature-path',default='/home/wjx/CodeData/code/tsai-main/star-bio/outputs/search_features/raw_snv.npy');a=p.parse_args();meta=load_meta(a.cache,True,a.data_root);y=meta['y'];labels=meta['labels'];
 with h5py.File(a.cache,'r') as h:b=h['batch'][:]
 arr=np.load(a.feature_path,mmap_mode='r');x=row_snv(np.asarray(arr[:,0,:],dtype='float32'));out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for fold in meta['batch_values']:
  t0=time.perf_counter();fid=meta['batch_values'].index(fold);test=np.flatnonzero(b==fid);observed=np.flatnonzero(b!=fid);tr,va=split_train_val(y,b,observed,.15,3407+fold);mu=x[tr].mean(0);tx=x[tr]-mu;vx=x[va]-mu;ex=x[test]-mu;w,mask=fisher_scale(tx,y[tr],len(labels),a.weight,a.topk);tx=tx[:,mask]*w[mask];vx=vx[:,mask]*w[mask];ex=ex[:,mask]*w[mask];keys=y[tr]*len(meta['batch_values'])+b[tr];counts=np.bincount(keys);weights=1/np.maximum(counts[keys],1);rng=np.random.default_rng(3407+fold);centers=tx[rng.choice(len(tx),min(a.centers,len(tx)),replace=False,p=weights/weights.sum())];best=None
  for gamma in (.01,.03,.1,.3,1.,3.):
   for alpha in (1e-6,1e-5,1e-4,1e-3,.01):
    sv=rbf_scores(tx,y[tr],vx,centers,gamma,alpha,weights);score=fast_macro_f1(y[va],sv.argmax(1),len(labels));
    if best is None or score>best[0]:best=(score,gamma,alpha)
  st=rbf_scores(tx,y[tr],ex,centers,best[1],best[2],weights);m=classification_metrics(y[test].tolist(),st.argmax(1).tolist(),labels);m.update(test_batch=fold,val_count=len(va),train_count=len(tr),best_val_f1=best[0],gamma=best[1],alpha=best[2],weight=a.weight,topk=a.topk,selected=int(mask.sum()),elapsed_seconds=time.perf_counter()-t0);fd=out/f'fold{fold}';fd.mkdir(exist_ok=True);(fd/'test_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));rows.append(m);print(json.dumps({k:v for k,v in m.items() if k not in ('cm','report')},ensure_ascii=False),flush=True)
 avg={k:float(np.mean([m[k] for m in rows])) for k in ('acc','bacc','macro_f1')};(out/'summary.json').write_text(json.dumps({'folds':rows,'AVG':avg,'weight':a.weight,'topk':a.topk},ensure_ascii=False,indent=2));print('SUMMARY',avg,flush=True)
if __name__=='__main__':main()
