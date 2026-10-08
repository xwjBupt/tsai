"""GPU chunked nearest-neighbor spectral classifier with validation-selected k."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.loads(Path(a.spec).read_text());out=Path(a.out);meta=load_meta(s['cache'],True,s['data_root'])
 with h5py.File(s['cache']) as h:arr=h['x'][:]
 if s.get('channel') is not None:arr=arr[:,s['channel']:s['channel']+1]
 if s.get('downsample',1)>1:arr=arr[:,:,::s['downsample']]
 x=torch.from_numpy(arr.reshape(len(arr),-1).copy()).cuda().float(); y=torch.tensor(meta['y'],device='cuda');labels=meta['labels'];res=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);fd=out/f'fold{fold}';fd.mkdir();tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda')
  mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5); z=torch.nn.functional.normalize((x-mu)/sd,dim=1); best=-1;hist=[]
  for k in s.get('ks',[1,3,5,11,21,51]):
   pred=[]
   for q in z[va].split(128):
    vals,ids=(q@z[tr].T).topk(k,dim=1); votes=torch.zeros(len(q),len(labels),device='cuda');votes.scatter_add_(1,y[tr][ids],torch.softmax(vals/s.get('temperature',.03),1));pred.extend(votes.argmax(1).cpu().tolist())
   m=classification_metrics(y[va].cpu().tolist(),pred,labels);hist.append({'k':k,'macro_f1':m['macro_f1']})
   if m['macro_f1']>best:best=m['macro_f1'];bk=k
  pred=[]
  for q in z[te].split(128):
   vals,ids=(q@z[tr].T).topk(bk,dim=1);votes=torch.zeros(len(q),len(labels),device='cuda');votes.scatter_add_(1,y[tr][ids],torch.softmax(vals/s.get('temperature',.03),1));pred.extend(votes.argmax(1).cpu().tolist())
  m=classification_metrics(y[te].cpu().tolist(),pred,labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,best_k=bk,elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(m,indent=2));(fd/'history.json').write_text(json.dumps(hist,indent=2));res.append(m);print('TEST',fold,m['macro_f1'],bk,flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':s['name'],'timestamp':s['timestamp'],'commit':s['commit'],'debug':False});print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
