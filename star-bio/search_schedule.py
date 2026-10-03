"""Persistent single-host GPU queue. Prefer <10% memory occupancy, then capacity."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def cards():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in text.strip().splitlines():
        i,total,used,util=map(int,line.split(','));rows.append((used/total>=.1,-(total-used),i,total-used,util))
    return sorted(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    state_path=out/'scheduler.json'
    if state_path.exists():raise RuntimeError('已有调度记录，请检查原进程，不可重复启动')
    queue=json.loads(Path(a.queue).read_text());active={};done=[];launched=set()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()
    while True:
        # Queue can be extended while the scheduler runs.
        queue=json.loads(Path(a.queue).read_text())
        for pid,job in list(active.items()):
            status=job['process'].poll()
            if status is not None:
                job['log'].close();done.append({'name':job['name'],'gpu':job['gpu'],'out':job['out'],'returncode':status})
                del active[pid]
        pending=[s for s in queue if s['name'] not in launched]
        occupied={j['gpu'] for j in active.values()}
        for busy,negative_free,gpu,free,util in cards():
            if not pending:break
            if gpu in occupied or free<12000:continue
            spec=pending.pop(0).copy();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip();stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%y-%m-%d@%H-%M-%S')
            spec.update(timestamp=stamp,commit=commit[:7])
            folder=out/spec['name']/f'{stamp}+commit-{commit[:7]}';folder.mkdir(parents=True)
            save(folder/'spec.json',spec)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            log=(folder/'launcher.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/('search_linear.py' if spec.get('runner')=='linear' else 'search_train.py')),'--spec',str(folder/'spec.json'),'--out',str(folder)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[proc.pid]={'process':proc,'log':log,'name':spec['name'],'gpu':gpu,'out':str(folder)};launched.add(spec['name'])
            print(f'LAUNCH gpu={gpu} free_mib={free} pid={proc.pid} {spec["name"]}',flush=True)
        save(state_path,{'pid':os.getpid(),'active':[{'pid':pid,**{k:v for k,v in j.items() if k not in ['process','log']}} for pid,j in active.items()],'finished':done,'pending':[s['name'] for s in queue if s['name'] not in launched]})
        if not active and all(s['name'] in launched for s in queue):break
        time.sleep(10)
if __name__=='__main__':main()
