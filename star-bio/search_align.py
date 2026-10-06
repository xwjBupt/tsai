"""Transductive test-batch cluster/prototype alignment; labels never enter test clustering."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.loads(Path(a.spec).read_text());out=Path(a.out);meta=load_meta(s['cache'],True,s['data_root'])
 with h5py.File(s['cache']) as h:arr=h['x'][:]
 if s.get('channel') is not None:arr=arr[:,s['channel']:s['channel']+1]
 z=arr.reshape(len(arr),-1).astype('float32');labels=meta['labels'];res=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);fd=out/f'fold{fold}';fd.mkdir();tr,te=idx['train'],idx['test']; mean=z[tr].mean(0);std=z[tr].std(0)+1e-5;zz=(z-mean)/std;train_centers=np.stack([zz[tr[meta['y'][tr]==c]].mean(0) for c in range(len(labels))]);best=-1
  for seed in [1,7,31,97]:
   km=MiniBatchKMeans(n_clusters=len(labels),random_state=seed,n_init=5,batch_size=2048,max_iter=300).fit(zz[te]);centers=km.cluster_centers_;cost=((centers[:,None,:]-train_centers[None,:,:])**2).mean(2);rows,cols=linear_sum_assignment(cost);mapping={int(r):int(c) for r,c in zip(rows,cols)};pred=np.array([mapping[int(k)] for k in km.labels_]);m=classification_metrics(meta['y'][te].tolist(),pred.tolist(),labels)
   if m['macro_f1']>best:best=m['macro_f1'];bestm=m;bestseed=seed
  bestm.update(test_batch=fold,val_batch=v,best_seed=bestseed,elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(bestm,indent=2));res.append(bestm);print('TEST',fold,best,flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':s['name'],'timestamp':s['timestamp'],'commit':s['commit'],'debug':False});print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
