"""Validation-selected score blending across independently trained RBF heads."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from train_patchtst import append_results_csv

def score(ck,feature,device='cuda'):
 s=ck['spec'];z=torch.from_numpy(feature.reshape(len(feature),-1).copy()).to(device,torch.float32)
 if s.get('channel') is not None:z=z[:,s['channel']*feature.shape[2]:(s['channel']+1)*feature.shape[2]]
 mean=ck['mean'].to(device).float();std=ck['std'].to(device).float();centers=ck['centers'].to(device).float();denom=ck['denom'].to(device).float();coef=ck['coef'].to(device).float();z=(z-mean)/std;phi=torch.exp(-ck['gamma']*torch.cdist(z,centers).square()/denom)
 if s.get('linear',False):phi=torch.cat([phi,z/z.shape[1]**.5],1)
 phi=torch.cat([phi,torch.ones(len(z),1,device=device)],1);return phi@coef

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);runs=[Path(x) for x in a.runs];meta=load_meta('/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5',True,'/home/wjx/CodeData/data/Star-Com/7class-4patch');labels=meta['labels'];res=[]
 for fold in meta['batch_values']:
  idx,v=make_splits(meta,fold);fsets=[];cks=[]
  for run in runs:
   cks.append(torch.load(run/f'fold{fold}'/'best.pt',map_location='cpu',weights_only=True));f=json.loads(Path(cks[-1]['spec']['feature_path']).with_suffix('.json').read_text());fsets.append(np.load(cks[-1]['spec']['feature_path'],mmap_mode='r'))
  yv=torch.tensor(meta['y'][idx['val']],device='cuda');yt=torch.tensor(meta['y'][idx['test']],device='cuda');val_scores=[score(c,f[idx['val']]) for c,f in zip(cks,fsets)];test_scores=[score(c,f[idx['test']]) for c,f in zip(cks,fsets)];best=-1
  for w in np.linspace(0,1,21):
   sv=w*val_scores[0]+(1-w)*val_scores[1];m=classification_metrics(yv.cpu().tolist(),sv.argmax(1).cpu().tolist(),labels)
   if m['macro_f1']>best:best=m['macro_f1'];bw=w
  st=bw*test_scores[0]+(1-bw)*test_scores[1];m=classification_metrics(yt.cpu().tolist(),st.argmax(1).cpu().tolist(),labels);m.update(test_batch=fold,val_batch=v,best_val_f1=best,blend_weight=float(bw));res.append(m);print('fold',fold,'w',bw,'f1',m['macro_f1'],flush=True)
 summary={'runs':[str(r) for r in runs],'folds':res,'AVG':{k:float(np.mean([m[k] for m in res])) for k in ['acc','bacc','macro_f1']}};out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));append_results_csv({m['test_batch']:m for m in res},labels,{'experiment_id':'rbf_ensemble','timestamp':'26-10-06','commit':'search','debug':False});print('SUMMARY',summary['AVG'])
if __name__=='__main__':main()
