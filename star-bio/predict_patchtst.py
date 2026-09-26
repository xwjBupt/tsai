"""Predict one cell spectrum with a PatchTST checkpoint."""
import argparse,csv,json
import numpy as np, torch
from patchtst_model import PatchTSTClassifier
from prepare_data import process

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--csv',required=True); p.add_argument('--row',type=int,default=1); p.add_argument('--tsai-root',default='/home/wjx/CodeData/code/tsai-main'); a=p.parse_args()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); c=ck['config']
 with open(a.csv,newline='',encoding='utf-8-sig') as f:
  r=csv.reader(f); wave=np.asarray(next(r),dtype=np.float64); rows=[x for x in r if x]
 if a.row<1 or a.row>len(rows): raise ValueError(f'row 必须在 1..{len(rows)}')
 x=torch.from_numpy(process(wave,np.asarray([rows[a.row-1]],dtype=np.float32),np.asarray(ck['wave']),ck['preprocess']))
 m=PatchTSTClassifier(c['n_classes'],c['n_points'],c.get('tsai_root',a.tsai_root),patch_len=c['patch_len'],stride=c['stride'],n_layers=c['layers'],n_heads=c['heads'],d_model=c['d_model'],d_ff=c['d_ff'],dropout=c['dropout']); m.load_state_dict(ck['model']); m.eval()
 with torch.inference_mode(): q=torch.softmax(m(x),1)[0]
 print(json.dumps({'class':ck['labels'][q.argmax().item()],'confidence':float(q.max()),'probabilities':dict(zip(ck['labels'],q.tolist()))},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
