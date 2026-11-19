"""Resumable training entry point. All accepted results have full provenance."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse, json, hashlib, time, random, pickle, sys, traceback, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.data import Dataset,Transform,STAGES,CFG,ROOT,metrics
from src.models import Net

torch.set_num_threads(2)
warnings.filterwarnings('ignore',category=UserWarning)
TRAD=['RF','GBDT','NN','AdaBoost','SVR','XGBoost']
NEURAL=['CNN','Early','Middle','Late']
MAIN=['XGBoost']+NEURAL

def environment():
    import sklearn,scipy,xgboost
    return {'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__}

def code_hash():
    paths=[Path(__file__),ROOT/'configs/protocol.json']+sorted((ROOT/'src').glob('*.py'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def learner(name,seed):
    return {'RF':lambda:RandomForestRegressor(n_estimators=100,random_state=seed,n_jobs=4),
            'GBDT':lambda:GradientBoostingRegressor(random_state=seed),
            'NN':lambda:MLPRegressor(random_state=seed,max_iter=200),
            'AdaBoost':lambda:AdaBoostRegressor(random_state=seed),
            'SVR':lambda:SVR(),
            'XGBoost':lambda:XGBRegressor(random_state=seed,n_jobs=4,tree_method='hist')}[name]()

def neural_train(name,train,val,ytrain,yval,seed,epochs=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda'
    assert torch.cuda.is_available(),'GPU required by the accepted protocol; CPU fallback is not silently enabled.'
    dims=(train[0].shape[1] if train[0] is not None else 151, train[1].shape[1] if train[1] is not None else 3)
    model=Net(name,*dims).to(device);model.initialize(ytrain.mean())
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['neural']['lr'],capturable=True)
    to=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32) if x is not None else None
    tx=[to(x) for x in train];vx=[to(x) for x in val];ty=to(ytrain);vy=to(yval)
    best=float('inf');best_state=None;stale=0;history=[]
    bs=CFG['neural']['batch_size'];max_epoch=epochs or CFG['neural']['epochs']
    # A fixed-size CUDA graph preserves masked last-minibatch MSE while removing
    # Python launch overhead. Warmup updates are restored before epoch one.
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    static_x=[torch.empty((bs,)+tuple(x.shape[1:]),device=device) if x is not None else None for x in tx]
    static_y=torch.empty(bs,device=device);static_mask=torch.ones(bs,device=device)
    fill=torch.arange(bs,device=device)%len(ty)
    for dst,x in zip(static_x,tx):
        if dst is not None:dst.copy_(x[fill])
    static_y.copy_(ty[fill])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=False);z=model(*static_x);(((z-static_y)**2)*static_mask).sum().div(static_mask.sum()).backward();optimizer.step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        static_pred=model(*static_x)
        static_loss=(((static_pred-static_y)**2)*static_mask).sum()/static_mask.sum()
        static_loss.backward();optimizer.step()
    model.load_state_dict(initial)
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value):value.zero_()
    for epoch in range(max_epoch):
        model.train();order=torch.randperm(len(ty),device=device);loss_sum=torch.zeros((),device=device)
        for batch in order.split(bs):
            n=len(batch)
            if n<bs:batch=torch.cat([batch,order[:bs-n]])
            for dst,x in zip(static_x,tx):
                if dst is not None:dst.copy_(x[batch])
            static_y.copy_(ty[batch]);static_mask.fill_(1)
            if n<bs:static_mask[n:]=0
            graph.replay();loss_sum+=static_loss.detach()*n
        model.eval()
        with torch.no_grad():vl=float(((model(*vx)-vy)**2).mean())
        history.append({'epoch':epoch+1,'train_mse':float(loss_sum/len(ty)),'val_mse':vl})
        if vl<best-1e-10:
            best=vl;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=CFG['neural']['patience']:break
    assert best_state is not None and np.isfinite(best)
    model.load_state_dict(best_state);model.eval()
    return model,history,dims

def predict(model,arrays):
    if isinstance(model,torch.nn.Module):
        device=next(model.parameters()).device
        with torch.no_grad():
            n=len(next(x for x in arrays if x is not None));outputs=[]
            for start in range(0,n,4096):
                xs=[torch.as_tensor(x[start:start+4096],device=device,dtype=torch.float32) if x is not None else None for x in arrays]
                outputs.append(model(*xs).cpu().numpy())
        return np.concatenate(outputs)
    return model.predict(arrays[1])

def execute(data,scenario,stage,name,seed,track='fair',mode='joint',epochs=None):
    runid=f'{scenario}__{track}__{mode}__{stage}__{name}__s{seed}'
    folder=ROOT/'results'/runid;result_path=folder/'result.json'
    ch=code_hash()
    if result_path.exists():
        existing=json.loads(result_path.read_text(encoding='utf-8'))
        assert existing['code_hash']==ch,f'Code changed since run {runid}; use a new output tree instead of mixing versions.'
        return existing
    folder.mkdir(parents=True,exist_ok=True)
    split=data.split(scenario);transform=Transform(data,stage,split,track,mode)
    arrays={k:transform.arrays(data,v,name) for k,v in split.items()}
    start=time.perf_counter()
    if name in TRAD:
        model=learner(name,seed)
        model.fit(arrays['train'][1],data.y[split['train']]);history=[];dims=None
    else:model,history,dims=neural_train(name,arrays['train'],arrays['val'],data.y[split['train']],data.y[split['val']],seed,epochs)
    training_seconds=time.perf_counter()-start
    pred=predict(model,arrays['test']);assert np.isfinite(pred).all()
    p=data.meta.iloc[split['test']].reset_index(drop=True);p['true']=data.y[split['test']];p['pred']=pred
    p.to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
    sm={k:{'ids':sorted(set(data.ids[idx])),'samples':len(idx)} for k,idx in split.items()}
    metadata={'run_id':runid,'scenario':scenario,'stage':stage,'model':name,'seed':seed,'track':track,'input_mode':mode,'code_hash':ch,'environment':environment(),'split':sm,'transform':transform.state(),'training_seconds':training_seconds,'epochs_trained':len(history),'best_val_mse':min((r['val_mse'] for r in history),default=None),'metrics':metrics(p.true,p.pred),'parameters':sum(p.numel() for p in model.parameters()) if isinstance(model,torch.nn.Module) else None,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'training_config':CFG['neural'] if history else model.get_params()}
    if isinstance(model,torch.nn.Module):
        torch.save({'kind':name,'dims':dims,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},folder/'model.pt')
    else:
        with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
    with (folder/'transform.pkl').open('wb') as f:pickle.dump(transform,f)
    (folder/'split.json').write_text(json.dumps(sm,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    # Last write is the completion marker. Incomplete directories are rerun safely.
    result_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'done':runid,'seconds':round(training_seconds,2),'epochs':len(history),'rmse_pp':round(metadata['metrics']['rmse_pp'],5)},ensure_ascii=False),flush=True)
    return metadata

def schedule(suite):
    jobs=[]
    add=lambda scenarios,models,seeds,track='fair',mode='joint':jobs.extend((sc,s,m,z,track,mode) for sc in scenarios for s in STAGES for m in models for z in seeds)
    if suite in ['main','all']:add(['holdout'],TRAD+NEURAL,CFG['seeds'])
    if suite in ['legacy','all']:add(['holdout'],TRAD+NEURAL,[2025],'legacy')
    if suite in ['cv','all']:add([f'cv{i}' for i in range(5)],MAIN,[2025])
    if suite in ['domain','all']:add(['temp35','temp45','rate025','rate05','rate1'],MAIN,CFG['seeds'])
    if suite in ['ablation','all']:
        add(['holdout'],['conditions'],CFG['seeds'])
        add(['holdout'],['MLP'],CFG['seeds'],mode='hi')
        add(['holdout'],['MLP'],CFG['seeds'])
        add(['holdout'],['CNN'],CFG['seeds'],mode='curve')
        add(['holdout'],['Early','Middle','Late'],CFG['seeds'],mode='no_conditions')
    return jobs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['main','legacy','cv','domain','ablation','all'],default='main');parser.add_argument('--scenario',default='holdout');parser.add_argument('--model');parser.add_argument('--stage',default='RE');parser.add_argument('--seed',type=int,default=2025);parser.add_argument('--epochs',type=int);parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();jobs=[(args.scenario,args.stage,args.model,args.seed,'fair','joint')] if args.model else schedule(args.suite)
    if args.plan:print(json.dumps({'jobs':len(jobs),'configuration':CFG,'schedule':jobs},ensure_ascii=False,indent=2));return
    print(json.dumps(environment(),ensure_ascii=False),flush=True);data=Dataset()
    for job in jobs:execute(data,*job,epochs=args.epochs)
if __name__=='__main__':main()
