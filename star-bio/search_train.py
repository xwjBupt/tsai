"""GPU-resident four-fold training, validation-only selection, test once per fold."""
import argparse,json,math,random,time,sys
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from metadata import load_meta
from splits import make_splits
from metrics import classification_metrics
from search_models import SearchModel


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

@torch.inference_mode()
def evaluate(model,x,y,indices,labels):
    model.eval(); ps=[]; loss=0.
    for ix in indices.split(256):
        with torch.autocast('cuda',dtype=torch.float16): out=model(x[ix])
        loss+=nn.functional.cross_entropy(out.float(),y[ix],reduction='sum').item()
        ps.extend(out.argmax(1).cpu().tolist())
    result=classification_metrics(y[indices].cpu().tolist(),ps,labels)
    result['loss']=loss/len(indices)
    return result,ps


def run(spec,outdir):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    meta=load_meta(spec['cache'],check_sources=True,data_root=spec['data_root'])
    if spec.get('feature_path'):
        feature_meta=json.loads(Path(spec['feature_path']).with_suffix('.json').read_text())
        assert feature_meta['fingerprint']==meta['fingerprint']
        array=np.load(spec['feature_path'])
        assert array.shape[0]==len(meta['y']) and array.shape[1]==3
        meta['n_points']=array.shape[-1]
        meta['wave']=np.asarray(feature_meta['wave'])
    else:
        feature_meta=None
        with h5py.File(spec['cache']) as h: array=h['x'][:]
    x=torch.from_numpy(array).cuda();del array
    y=torch.from_numpy(meta['y']).cuda()
    labels=meta['labels']; results=[]
    save(outdir/'experiment.json',spec)
    for fold in meta['batch_values']:
        start=time.perf_counter(); seed=spec.get('seed',3407)+fold
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        idx,v=make_splits(meta,fold)
        folder=outdir/f'fold{fold}';folder.mkdir()
        save(folder/'split.json',{'test_batch':fold,'val_batch':v,'counts':{k:len(a) for k,a in idx.items()},'fingerprint':meta['fingerprint']})
        np.savez_compressed(folder/'splits.npz',**idx)
        tr=torch.as_tensor(idx['train'],device='cuda');va=torch.as_tensor(idx['val'],device='cuda');te=torch.as_tensor(idx['test'],device='cuda')
        keys=meta['y'][idx['train']]*len(meta['batch_values'])+meta['batch'][idx['train']]
        counts=np.bincount(keys); weights=torch.tensor(1/counts[keys],device='cuda',dtype=torch.float32)
        model=SearchModel(spec,meta['n_points'],len(labels)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=spec.get('lr',.0003),weight_decay=spec.get('wd',.01))
        scaler=torch.amp.GradScaler('cuda');best=-1;stale=0;hist=[];steps=0
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(outdir/'tensorboard'/f'fold{fold}'))
        for ep in range(1,spec.get('epochs',80)+1):
            model.train(); total=0.; seen=0
            lr=spec.get('lr',.0003)*(.05+.95*(1+math.cos(math.pi*(ep-1)/spec.get('epochs',80)))/2)
            for g in opt.param_groups:g['lr']=lr
            order=tr[torch.multinomial(weights,len(tr),replacement=True)]
            for ix in order.split(spec.get('batch',128)):
                xx=x[ix]
                if spec.get('noise',0):xx=xx+torch.randn_like(xx)*spec['noise']
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(xx); loss=nn.functional.cross_entropy(logits,y[ix],label_smoothing=spec.get('smoothing',.05))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.)
                scale=scaler.get_scale();scaler.step(opt);scaler.update();steps+=int(scaler.get_scale()>=scale)
                total+=loss.item()*len(ix);seen+=len(ix)
            val,_=evaluate(model,x,y,va,labels)
            record={**val,'epoch':ep,'train_loss':total/seen,'val_loss':val['loss'],'lr':lr,'steps':steps,'elapsed_seconds':time.perf_counter()-start}
            hist.append(record);save(folder/'history.json',hist)
            for k in ['acc','bacc','macro_f1','train_loss','val_loss','lr']:writer.add_scalar(k,record[k],ep)
            writer.flush()
            print(json.dumps({'fold':fold,'epoch':ep,'train_loss':total/seen,'val_f1':val['macro_f1'],'seconds':record['elapsed_seconds']}),flush=True)
            if val['macro_f1']>best:
                best=val['macro_f1'];stale=0
                torch.save({'model':model.state_dict(),'spec':spec,'n_points':meta['n_points'],'labels':labels,'fold':fold,'val_batch':v,'epoch':ep,'best_val_f1':best,'wave':meta['wave'].tolist(),'preprocess':meta['preprocess'],'fingerprint':meta['fingerprint'],'feature_meta':feature_meta},folder/'best.pt')
            else: stale+=1
            if stale>=spec.get('patience',15):break
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
        test,pred=evaluate(model,x,y,te,labels)
        test.update(test_batch=fold,val_batch=v,best_epoch=ck['epoch'],best_val_f1=best,elapsed_seconds=time.perf_counter()-start)
        save(folder/'test_metrics.json',test);np.savez_compressed(folder/'predictions.npz',indices=idx['test'],y=meta['y'][idx['test']],pred=pred)
        results.append(test);writer.close()
        print('TEST '+json.dumps({k:v for k,v in test.items() if k not in ['cm','report']}),flush=True)
        save(outdir/'progress.json',{'completed_folds':len(results),'folds':results})
        del model,opt,scaler,ck;torch.cuda.empty_cache()
    summary={'experiment_id':spec['name'],'timestamp':spec['timestamp'],'commit':spec['commit'],'spec':spec,'folds':results,
             'AVG':{k:float(np.mean([r[k] for r in results])) for k in ['acc','bacc','macro_f1']},
             'STD':{k:float(np.std([r[k] for r in results])) for k in ['acc','bacc','macro_f1']}}
    save(outdir/'summary.json',summary)
    # Reuse locked results registry without changing existing experiments.
    from train_patchtst import append_results_csv
    append_results_csv({r['test_batch']:r for r in results},labels,{'experiment_id':spec['name'],'timestamp':spec['timestamp'],'commit':spec['commit'],'debug':False})
    print('SUMMARY '+json.dumps(summary['AVG']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run(json.loads(Path(a.spec).read_text()),Path(a.out))
