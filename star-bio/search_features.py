"""Per-spectrum alternative preprocessing. No labels or batch statistics fitted."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from prepare_data import read_csv,discover_files
from metadata import load_meta


def snv(x): return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)
def channels(x):
    x=snv(x)
    d=savgol_filter(x,9,2,deriv=1,axis=1)
    dd=savgol_filter(x,9,2,deriv=2,axis=1)
    return np.stack([x,snv(d),snv(dd)],1).astype('float32')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    root=Path('/home/wjx/CodeData/data/Star-Com/7class-4patch');meta=load_meta(root/'cache_v2.h5',True,root)
    grid=meta['wave']; fp=(grid>=1000)&(grid<=1800)
    modes={'raw_snv':len(grid),'signed401':len(grid),'fingerprint':int(fp.sum()),'signed101':len(grid)}
    arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype='float32',shape=(len(meta['y']),3,n)) for k,n in modes.items()}
    offset=0
    with h5py.File(root/'cache_v2.h5') as h: names=h['file'].asstr()[:]
    for path,label,batch,name in discover_files(root)[0]:
        wave,raw=read_csv(path); wave=wave[220:1260];raw=raw[:,220:1260];order=np.argsort(wave)
        raw=interp1d(wave[order],raw[:,order],axis=1,bounds_error=True)(grid).astype('float32')
        end=offset+len(raw)
        assert (names[offset:end]==name).all()
        smooth=savgol_filter(raw,9,2,axis=1)
        arrays['raw_snv'][offset:end]=channels(smooth)
        arrays['signed401'][offset:end]=channels(smooth-savgol_filter(smooth,401,3,axis=1))
        arrays['signed101'][offset:end]=channels(smooth-savgol_filter(smooth,101,3,axis=1))
        arrays['fingerprint'][offset:end]=channels(smooth[:,fp])
        offset=end;print(name,end,flush=True)
    assert offset==len(meta['y'])
    for mode,array in arrays.items():
        array.flush()
        (out/f'{mode}.json').write_text(json.dumps({'mode':mode,'fingerprint':meta['fingerprint'],'wave':grid[fp].tolist() if mode=='fingerprint' else grid.tolist(),'shape':list(array.shape),'preprocess':'smooth9order2 + per-spectrum SNV and SNV derivatives; signed SG background, no clipping'},indent=2))
if __name__=='__main__':main()
