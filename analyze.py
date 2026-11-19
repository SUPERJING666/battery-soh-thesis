"""Evaluate accepted checkpoints only; no model fitting in this module."""
from pathlib import Path
import argparse,json,pickle,time,os,threading
import numpy as np,pandas as pd,torch,psutil
from src.data import Dataset,ROOT,CFG,metrics
from src.models import Net
from run import predict

def load(folder,device='cuda'):
    meta=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    with (folder/'transform.pkl').open('rb') as f:t=pickle.load(f)
    if (folder/'model.pt').exists():
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);m=Net(ck['kind'],*ck['dims'],input_layout=ck.get('input_layout','legacy'));m.load_state_dict(ck['state_dict']);m.to(device).eval()
    else:
        with (folder/'model.pkl').open('rb') as f:m=pickle.load(f)
    return meta,t,m

def accepted():
    for p in sorted((ROOT/'results').glob('*/result.json')):
        m=json.loads(p.read_text(encoding='utf-8'))
        if m['scenario']=='holdout' and m['track']=='fair' and m['input_mode']=='joint':yield p.parent,m

def robustness(data):
    path=ROOT/'reports/robustness.csv';rows=[]
    for folder,meta in accepted():
        if meta['model'] not in ['XGBoost','CNN','Early','Middle','Late']:continue
        m,t,model=load(folder);split=data.split('holdout');idx=split['test'];stage=m['stage'];raw=data.curves[stage][idx]
        sigma=float(np.std(data.curves[stage][split['train']],ddof=0))
        rng=np.random.RandomState(7301);base_noise=rng.normal(size=raw.shape)
        for fraction in CFG['noise_fractions']:
            noisy=raw+base_noise*(sigma*fraction)
            pred=predict(model,t.arrays(data,idx,m['model'],curve_override=noisy))
            rows.append({'run_id':m['run_id'],'stage':stage,'model':m['model'],'seed':m['seed'],'test_type':'gaussian','level':fraction,'noise_sd_physical':sigma*fraction,'physical_unit':'mA' if stage=='CV' else 'V',**metrics(data.y[idx],pred)})
        for fraction in [.05,.10,.20]:
            rng=np.random.RandomState(9201);changed=raw.copy();grid=np.arange(151)
            for k in range(len(raw)):
                drop=rng.choice(grid[1:-1],int(149*fraction),replace=False);keep=np.ones(151,dtype=bool);keep[drop]=False
                changed[k]=np.interp(grid,grid[keep],raw[k,keep])
            pred=predict(model,t.arrays(data,idx,m['model'],curve_override=changed))
            rows.append({'run_id':m['run_id'],'stage':stage,'model':m['model'],'seed':m['seed'],'test_type':'synthetic_missing_points_interpolated','level':fraction,'noise_sd_physical':None,**metrics(data.y[idx],pred)})
        del model
    pd.DataFrame(rows).to_csv(path,index=False)
    print('robustness rows',len(rows),flush=True)

def importance(data):
    rows=[];split=data.split('holdout');idx=split['test'];conditions=data.conditions[idx]
    blocks=[np.flatnonzero(np.all(conditions==x,axis=1)) for x in np.unique(conditions,axis=0)]
    for folder,m in accepted():
        if m['model'] not in ['Early','Middle','Late']:continue
        meta,t,model=load(folder);arrays=t.arrays(data,idx,m['model']);base=metrics(data.y[idx],predict(model,arrays))['rmse_pp']
        for branch in [0,1]:
            for repeat in range(CFG['permutation_repeats']):
                rng=np.random.RandomState(4100+repeat);order=np.arange(len(idx))
                for block in blocks:order[block]=rng.permutation(block)
                xs=list(arrays);xs[branch]=xs[branch][order]
                err=metrics(data.y[idx],predict(model,xs))['rmse_pp']
                rows.append({'run_id':m['run_id'],'stage':m['stage'],'model':m['model'],'seed':m['seed'],'branch':'curve' if branch==0 else 'handcrafted_with_conditions','repeat':repeat,'base_rmse_pp':base,'permuted_rmse_pp':err,'delta_rmse_pp':err-base})
        del model
    pd.DataFrame(rows).to_csv(ROOT/'reports/permutation_importance.csv',index=False)
    print('importance rows',len(rows),flush=True)

def benchmark_one(folder,device):
    data=Dataset();proc=psutil.Process();base_rss=proc.memory_info().rss;peak=[base_rss];stop=threading.Event()
    def sample():
        while not stop.wait(.001):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    m,t,model=load(folder,device);idx=data.split(m['scenario'])['test'][:1];arrays=t.arrays(data,idx,m['model'])
    neural=isinstance(model,torch.nn.Module)
    if device=='cuda' and neural:torch.cuda.reset_peak_memory_stats()
    def forward():return predict(model,arrays)
    def complete():return predict(model,t.arrays(data,idx,m['model']))
    times={}
    for name,fn in [('forward',forward),('preprocessing_and_forward',complete)]:
        for _ in range(20):fn()
        samples=[]
        for _ in range(200):
            if device=='cuda' and neural:torch.cuda.synchronize()
            start=time.perf_counter();fn()
            if device=='cuda' and neural:torch.cuda.synchronize()
            samples.append(1000*(time.perf_counter()-start))
        times[name+'_median_ms']=float(np.median(samples));times[name+'_p95_ms']=float(np.percentile(samples,95))
    stop.set();thread.join();peak[0]=max(peak[0],proc.memory_info().rss)
    ck=folder/('model.pt' if neural else 'model.pkl')
    row={'run_id':m['run_id'],'stage':m['stage'],'model':m['model'],'device':device,'batch_size':1,'warmup':20,'repeats':200,'parameters':m['parameters'],'serialized_bytes':ck.stat().st_size,'process_baseline_rss_bytes':base_rss,'process_peak_rss_bytes':peak[0],'rss_increment_bytes':max(0,peak[0]-base_rss),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' and neural else None,**times}
    suffix=device+'.json';(folder/('benchmark_'+suffix)).write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)

def summarize():
    rows=[];per_battery=[]
    for p in sorted((ROOT/'results').glob('*/result.json')):
        m=json.loads(p.read_text(encoding='utf-8'));row={k:m[k] for k in ['run_id','scenario','stage','model','seed','track','input_mode','training_seconds','epochs_trained','parameters']};row.update(m['metrics']);rows.append(row)
        d=pd.read_csv(p.parent/'predictions.csv')
        for bid,g in d.groupby('id'):per_battery.append({k:row[k] for k in ['run_id','scenario','stage','model','seed','track','input_mode']}|{'id':bid}|metrics(g.true,g.pred))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'reports/all_runs.csv',index=False)
    pd.DataFrame(per_battery).to_csv(ROOT/'reports/per_battery.csv',index=False)
    keys=['scenario','track','input_mode','stage','model']
    summary=df.groupby(keys).agg(n_runs=('seed','size'),rmse_mean=('rmse_pp','mean'),rmse_sd=('rmse_pp','std'),mae_mean=('mae_pp','mean'),mae_sd=('mae_pp','std'),r2_mean=('r2','mean'),r2_sd=('r2','std')).reset_index()
    summary.to_csv(ROOT/'reports/summary.csv',index=False)
    print('accepted training runs',len(df),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['robustness','importance','summary','benchmark'],default='summary');p.add_argument('--run');p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.task=='summary':summarize()
    elif a.task=='benchmark':benchmark_one(ROOT/'results'/a.run,a.device)
    elif a.task=='robustness':robustness(Dataset())
    elif a.task=='importance':importance(Dataset())
if __name__=='__main__':main()
