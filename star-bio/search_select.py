"""Train-only Fisher spectral feature selection + RBF ridge search."""
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
 x=torch.from_numpy(arr[:,s.get('channel',0),:].copy()).cuda().float();y=torch.tensor(meta['y'],device='cuda');labels=meta['labels'];res=[]
 for fold in meta['batch_values']:
  st=time.perf_counter();idx,v=make_splits(meta,fold);fd=out/f'fold{fold}';fd.mkdir();tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['val'],device='cuda');te=torch.tensor(idx['test'],device='cuda');mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5);z=(x-mu)/sd
  means=torch.stack([z[tr[y[tr]==c]].mean(0) for c in range(len(labels))]);between=((means-means.mean(0))**2).mean(0);within=torch.stack([z[tr[y[tr]==c]].var(0,unbiased=False) for c in range(len(labels))]).mean(0);score=between/(within+1e-5);best=-1
  for k in s.get('ks',[32,64,128,256,512]):
   selected=score.topk(min(k,z.shape[1])).indices;zz=z[:,selected];centers=zz[tr][torch.randperm(len(tr),device='cuda')[:min(4096,len(tr))]];denom=torch.cdist(centers[:256],centers[:256]).square().median().clamp_min(1e-6);tp=torch.exp(-.3*torch.cdist(zz[tr],centers).square()/denom);vp=torch.exp(-.3*torch.cdist(zz[va],centers).square()/denom);tp=torch.cat([tp,torch.ones(len(tr),1,device='cuda')],1);vp=torch.cat([vp,torch.ones(len(va),1,device='cuda')],1);cov=tp.T@tp;rhs=tp.T@torch.nn.functional.one_hot(y[tr],len(labels)).float();coef=torch.linalg.solve(cov+.001*cov.diag().mean()*torch.eye(cov.shape[0],device='cuda'),rhs);m=classification_metrics(y[va].cpu().tolist(),(vp@coef).argmax(1).cpu().tolist(),labels)
   if m['macro_f1']>best:best=m['macro_f1'];bk=k;bcoef=coef;bsel=selected;bcenters=centers; bdenom=denom
  tephi=torch.exp(-.3*torch.cdist(z[te][:,bsel],bcenters).square()/bdenom);tephi=torch.cat([tephi,torch.ones(len(te),1,device='cuda')],1);m=classification_metrics(y[te].cpu().tolist(),(tephi@bcoef).argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,selected_features=int(bk),elapsed_seconds=time.perf_counter()-st);(fd/'test_metrics.json').write_text(json.dumps(m,indent=2));res.append(m);print('TEST',fold,m['macro_f1'],bk,flush=True)
 summary={'spec':s,'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':s['name'],'timestamp':s['timestamp'],'commit':s['commit'],'debug':False});print('SUMMARY',summary['AVG'],flush=True)
if __name__=='__main__':main()
