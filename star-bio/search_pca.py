"""Train-only PCA + RBF ridge classifier; PCA fit never sees validation/test rows."""
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
 x=torch.from_numpy(arr.reshape(len(arr),-1).copy()).to('cuda',torch.float32);y=torch.tensor(meta['y'],device='cuda');labels=meta['labels'];res=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);fd=out/f'fold{fold}';fd.mkdir();tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda');mu=x[tr].mean(0);xc=x[tr]-mu
  # randomized low rank via covariance eigenvectors; training set only.
  q=min(s['components'],xc.shape[0]-1,xc.shape[1]);_,_,vmat=torch.pca_lowrank(xc,q=q,center=False);
  z=torch.matmul(x-mu,vmat[:,:q]);scale=z[tr].std(0).clamp_min(1e-5);z=z/scale
  centers=z[tr][torch.randperm(len(tr),device='cuda')[:min(s.get('centers',4096),len(tr))]];denom=torch.cdist(centers[:256],centers[:256]).square().median().clamp_min(1e-6);best=-1;hist=[]
  for gamma in [.01,.03,.1,.3,1.,3.]:
   trainphi=torch.exp(-gamma*torch.cdist(z[tr],centers).square()/denom);valphi=torch.exp(-gamma*torch.cdist(z[va],centers).square()/denom);trainphi=torch.cat([trainphi,torch.ones(len(tr),1,device='cuda')],1);valphi=torch.cat([valphi,torch.ones(len(va),1,device='cuda')],1);cov=trainphi.T@trainphi;rhs=trainphi.T@torch.nn.functional.one_hot(y[tr],len(labels)).float();
   for alpha in [1e-3,.01,.1,1.]:
    coef=torch.linalg.solve(cov+alpha*cov.diag().mean()*torch.eye(cov.shape[0],device='cuda'),rhs);m=classification_metrics(y[va].cpu().tolist(),(valphi@coef).argmax(1).cpu().tolist(),labels);hist.append({'gamma':gamma,'alpha':alpha,'macro_f1':m['macro_f1']});
    if m['macro_f1']>best:best=m['macro_f1'];bg=gamma;ba=alpha;bcoef=coef
  testphi=torch.exp(-bg*torch.cdist(z[te],centers).square()/denom);testphi=torch.cat([testphi,torch.ones(len(te),1,device='cuda')],1);m=classification_metrics(y[te].cpu().tolist(),(testphi@bcoef).argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,best_gamma=bg,best_alpha=ba,elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(m,indent=2));(fd/'history.json').write_text(json.dumps(hist,indent=2));res.append(m);print('TEST',fold,m['macro_f1'],flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':s['name'],'timestamp':s['timestamp'],'commit':s['commit'],'debug':False});print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
