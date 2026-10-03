"""Predict raw spectra, or independently verify all four saved RBF checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from metadata import load_meta
from prepare_data import read_csv,discover_files
from search_features import channels
from splits import make_splits
from metrics import classification_metrics


def features(wave,raw,grid,mode):
    wave=wave[220:1260];raw=raw[:,220:1260];order=np.argsort(wave)
    z=interp1d(wave[order],raw[:,order],axis=1,bounds_error=True)(grid).astype('float32')
    z=savgol_filter(z,9,2,axis=1)
    if mode.startswith('signed'):
        z=z-savgol_filter(z,int(mode[6:]),3,axis=1)
    elif mode!='raw_snv':raise ValueError('Unsupported feature mode')
    return channels(z)

@torch.inference_mode()
def predict(ck,xx,device):
    spec=ck['spec']
    if spec.get('channel') is not None:xx=xx[:,spec['channel']:spec['channel']+1,:]
    x=torch.from_numpy(xx.reshape(len(xx),-1).copy()).to(device,torch.float64)
    mean,std,centers,denom,coef=[ck[k].to(device) for k in ['mean','std','centers','denom','coef']]
    x=(x-mean)/std
    z=torch.exp(-ck['gamma']*torch.cdist(x,centers).square()/denom)
    if spec.get('linear',False):z=torch.cat([z,x/x.shape[1]**.5],1)
    z=torch.cat([z,torch.ones(len(z),1,device=device,dtype=x.dtype)],1)
    # Regression discriminant scores; these are not calibrated probabilities.
    return (z@coef).cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify-run');g.add_argument('--checkpoint')
    p.add_argument('--csv');p.add_argument('--row',type=int,default=1);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(2)
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        fm=json.loads(Path(ck['spec']['feature_path']).with_suffix('.json').read_text())
        w,x=read_csv(a.csv)
        if not 1<=a.row<=len(x):raise ValueError('row out of bounds')
        z=features(w,x[a.row-1:a.row],np.array(fm['wave']),fm['mode']);scores=predict(ck,z,a.device)[0]
        print(json.dumps({'class':ck['labels'][int(scores.argmax())],'scores':dict(zip(ck['labels'],scores.tolist()))},ensure_ascii=False,indent=2));return
    root=Path(a.verify_run);report=[]
    for fold in (1,2,3,4):
        folder=root/f'fold{fold}';ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);spec=ck['spec']
        meta=load_meta(spec['cache'],True,spec['data_root'])
        fm=json.loads(Path(spec['feature_path']).with_suffix('.json').read_text());assert fm['fingerprint']==meta['fingerprint']
        cached=np.load(spec['feature_path'],mmap_mode='r');ix,val=make_splits(meta,fold)
        with np.load(folder/'splits.npz') as saved:
            for key in ix:np.testing.assert_array_equal(saved[key],ix[key])
        assert not set(ix['train'])&set(ix['test']) and not set(ix['val'])&set(ix['test'])
        ys=[];ps=[];offset=0
        # Rebuild raw features for every test CSV independently of feature cache.
        for path,label,batch,name in discover_files(spec['data_root'])[0]:
            count=next(c['cells'] for c in meta['counts'] if c['class']==label and c['batch']==batch)
            if batch==fold:
                w,raw=read_csv(path);assert len(raw)==count
                xx=features(w,raw,np.array(fm['wave']),fm['mode'])
                np.testing.assert_array_equal(xx,cached[offset:offset+count])
                pred=np.concatenate([predict(ck,xx[j:j+256],a.device).argmax(1) for j in range(0,count,256)])
                ys.extend([meta['labels'].index(label)]*count);ps.extend(pred.tolist())
            offset+=count
        m=classification_metrics(ys,ps,ck['labels']);old=json.loads((folder/'test_metrics.json').read_text())
        for key in ['acc','bacc','macro_f1']:assert abs(m[key]-old[key])<1e-12,(fold,key,m[key],old[key])
        assert m['cm']==old['cm']
        assert len(ys)==len(ix['test'])
        np.savez_compressed(folder/'verified_predictions.npz',y=np.array(ys),pred=np.array(ps),indices=ix['test'])
        report.append({'fold':fold,'test_cells':len(ys),'val_batch':val,**m})
        print(f'fold={fold} raw CSV + checkpoint verified F1={m["macro_f1"]:.8f}',flush=True)
    result={'verified':True,'source':'raw test CSVs, rebuilt features, saved checkpoint, disjoint saved splits', 'folds':report,'AVG':{k:float(np.mean([m[k] for m in report])) for k in ['acc','bacc','macro_f1']}}
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(result['AVG'],flush=True)
if __name__=='__main__':main()
