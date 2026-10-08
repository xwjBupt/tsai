"""Validation-only class-prior calibration for an existing RBF kernel checkpoint."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from metadata import load_meta
from prepare_data import read_csv,discover_files
from splits import make_splits
from metrics import classification_metrics
from search_features import channels
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def feats(w,raw,g,mode):
 w=w[220:1260];raw=raw[:,220:1260];o=np.argsort(w);z=interp1d(w[o],raw[:,o],axis=1,bounds_error=True)(g).astype('float32');z=savgol_filter(z,9,2,axis=1)
 return channels(z)
@torch.inference_mode()
def score(ck,x):
 s=ck['spec'];xx=x
 if s.get('channel') is not None:xx=xx[:,s['channel']:s['channel']+1]
 z=torch.from_numpy(xx.reshape(len(xx),-1).copy()).cuda().float();mean,std,centers,denom,coef=[ck[k].cuda() for k in ['mean','std','centers','denom','coef']];z=(z-mean)/std;z=torch.exp(-ck['gamma']*torch.cdist(z,centers).square()/denom)
 if s.get('linear',False):z=torch.cat([z,z/z.shape[1]**.5],1)
 z=torch.cat([z,torch.ones(len(z),1,device='cuda')],1);return z@coef
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--feature-mode',default='raw_snv');a=p.parse_args();root=Path(a.run);meta=load_meta('/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5',True,'/home/wjx/CodeData/data/Star-Com/7class-4patch');labels=meta['labels'];out=[]
 for fold in meta['batch_values']:
  fd=root/f'fold{fold}';ck=torch.load(fd/'best.pt',map_location='cpu',weights_only=True);idx,v=make_splits(meta,fold);ys=[];valx=[];testx=[]
  for path,label,batch,name in discover_files('/home/wjx/CodeData/data/Star-Com/7class-4patch')[0]:
   if batch not in (v,fold):continue
   w,raw=read_csv(path);xx=feats(w,raw,np.asarray(ck['spec'].get('wave',meta['wave'])),a.feature_mode) if ck['spec'].get('wave') else feats(w,raw,meta['wave'],a.feature_mode)
   (valx if batch==v else testx).append(xx);ys.extend([labels.index(label)]*len(xx))
  # Use cached feature matrix and saved indices for consistency.
  feature=np.load(ck['spec']['feature_path'],mmap_mode='r');vscore=score(ck,feature[idx['val']]);tscore=score(ck,feature[idx['test']]);vy=torch.tensor(meta['y'][idx['val']],device='cuda');ty=torch.tensor(meta['y'][idx['test']],device='cuda')
  base=vscore;bias=torch.zeros(len(labels),device='cuda');best=-1
  grid=np.linspace(-1.5,1.5,13)
  for _ in range(5):
   changed=False
   for c in range(len(labels)):
    local=best
    for delta in grid:
     b=bias.clone();b[c]+=float(delta);m=classification_metrics(vy.cpu().tolist(),(base+b).argmax(1).cpu().tolist(),labels)
     if m['macro_f1']>local:local=m['macro_f1'];best_bias=b;changed=True
    if changed:bias=best_bias;best=local
   if not changed:break
  bm=classification_metrics(vy.cpu().tolist(),(base).argmax(1).cpu().tolist(),labels);tm=classification_metrics(ty.cpu().tolist(),(tscore+bias).argmax(1).cpu().tolist(),labels);tm.update(test_batch=fold,val_batch=v,base_val_f1=bm['macro_f1'],calibrated_val_f1=best,bias=bias.cpu().tolist());out.append(tm);print('fold',fold,'base',bm['macro_f1'],'val',best,'test',tm['macro_f1'],'bias',bias.cpu().tolist(),flush=True)
 print('AVG',float(np.mean([x['macro_f1'] for x in out])))
 (root/'calibrated_metrics.json').write_text(json.dumps({'folds':out,'AVG':float(np.mean([x['macro_f1'] for x in out]))},indent=2))
if __name__=='__main__':main()
