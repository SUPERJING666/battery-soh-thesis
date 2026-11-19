"""Transparent public-description comparators; NOT exact paper reproductions.

Publication architecture/hyperparameter details are incomplete. Every result is
labelled adapted_public_description and the explicit implementation is archived.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,time,pickle,hashlib,random,argparse
import numpy as np,pandas as pd,torch
from torch import nn
from scipy.stats import wasserstein_distance
from sklearn.model_selection import GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data import Dataset,Transform,STAGES,ROOT,CFG,metrics
from run import predict,code_hash

torch.set_num_threads(2)

class ELM:
    def __init__(self,seed,width=64,ridge=.001):self.seed=seed;self.width=width;self.ridge=ridge
    def fit(self,x,y):
        rng=np.random.RandomState(self.seed);self.w=rng.normal(size=(x.shape[1],self.width))/np.sqrt(x.shape[1]);self.b=rng.uniform(-1,1,self.width)
        h=np.tanh(x@self.w+self.b);h=np.column_stack([h,np.ones(len(h))]);self.beta=np.linalg.solve(h.T@h+self.ridge*np.eye(h.shape[1]),h.T@y);return self
    def predict(self,x):return np.column_stack([np.tanh(x@self.w+self.b),np.ones(len(x))])@self.beta

class FeatureLSTM(nn.Module):
    def __init__(self,features):super().__init__();self.rnn=nn.LSTM(1,16,batch_first=True);self.out=nn.Linear(16,1)
    def forward(self,c,h):return self.out(self.rnn(h.unsqueeze(-1))[0][:,-1]).flatten()

class MultiScale(nn.Module):
    def __init__(self,features):
        super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,16,k,padding=k//2),nn.ReLU(),nn.Conv1d(16,32,k,padding=k//2),nn.ReLU()) for k in [3,5,7]])
        self.att=nn.Sequential(nn.Linear(96,12),nn.ReLU(),nn.Linear(12,96),nn.Sigmoid());self.out=nn.Linear(96+features,1)
    def embedding(self,c,h):
        x=torch.cat([b(c.unsqueeze(1)) for b in self.branches],dim=1).mean(2);return torch.cat([x*self.att(x),h],dim=1)
    def forward(self,c,h):return self.out(self.embedding(c,h)).flatten()

class TBA(nn.Module):
    def __init__(self,features):
        super().__init__();self.tcn=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,16,3,padding=2,dilation=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=4,dilation=4),nn.ReLU())
        self.rnn=nn.LSTM(16,16,batch_first=True,bidirectional=True);self.att=nn.Linear(32,1);self.out=nn.Linear(32+features,1)
    def forward(self,c,h):
        x=self.rnn(self.tcn(c.unsqueeze(1)).transpose(1,2))[0];a=torch.softmax(self.att(x),dim=1);return self.out(torch.cat([(a*x).sum(1),h],dim=1)).flatten()

def train_custom(model,train,val,y,yval,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=model.cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    model.out.bias.data.fill_(float(np.mean(y)))
    tx=[torch.tensor(a,dtype=torch.float32,device='cuda') if a is not None else None for a in train]
    vx=[torch.tensor(a,dtype=torch.float32,device='cuda') if a is not None else None for a in val]
    ty=torch.tensor(y,dtype=torch.float32,device='cuda');vy=torch.tensor(yval,dtype=torch.float32,device='cuda')
    best=float('inf');state=None;stale=0;history=[]
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(ty),device='cuda').split(512):
            opt.zero_grad(set_to_none=True);p=model(*[x[batch] if x is not None else None for x in tx]);loss=((p-ty[batch])**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():v=float(((model(*vx)-vy)**2).mean())
        history.append({'epoch':epoch+1,'val_mse':v})
        if v<best-1e-10:best=v;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=15:break
    model.load_state_dict(state);model.eval();return model,history

def embed(model,arrays):
    with torch.no_grad():
        c,h=arrays;return model.embedding(torch.tensor(c,dtype=torch.float32,device='cuda'),torch.tensor(h,dtype=torch.float32,device='cuda')).cpu().numpy()

def base_estimators(data,stage,split,seed):
    t=Transform(data,stage,split);a={k:t.arrays(data,idx,'MLP') for k,idx in split.items()};x=a['train'][1];y=data.y[split['train']]
    rng=np.random.RandomState(seed);chosen=rng.choice(len(x),min(2000,len(x)),replace=False)
    gp=GaussianProcessRegressor(kernel=RBF(1.)+WhiteKernel(.001),optimizer=None,normalize_y=True).fit(x[chosen],y[chosen])
    svr=SVR().fit(x,y);elm=ELM(seed).fit(x,y)
    torch.manual_seed(seed);lstm,hist=train_custom(FeatureLSTM(x.shape[1]),a['train'],a['val'],y,data.y[split['val']],seed)
    def outputs(idx):
        ar=t.arrays(data,idx,'MLP');xx=ar[1]
        return np.column_stack([gp.predict(xx),svr.predict(xx),elm.predict(xx),predict(lstm,ar)])
    return t,(gp,svr,elm,lstm),outputs,hist

def finish(folder,data,idx,pred,metadata,artifacts):
    p=data.meta.iloc[idx].reset_index(drop=True);p['true']=data.y[idx];p['pred']=pred;p.to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
    metadata['status']='adapted_public_description_not_exact_reproduction';metadata['metrics']=metrics(data.y[idx],pred)
    metadata['pipeline_code_hash']=code_hash();metadata['protocol_version']=CFG['protocol_version']
    metadata['custom_neural_training']={'max_epochs':100,'patience':15,'batch_size':512,'optimizer':'Adam','lr':.001,'final_bias':'source_training_label_mean'}
    with (folder/'artifacts.pkl').open('wb') as f:pickle.dump(artifacts,f)
    (folder/'result.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'literature_done':metadata['run_id'],'rmse_pp':metadata['metrics']['rmse_pp']},ensure_ascii=False),flush=True)

def run55(data,stage,seed):
    runid=f'ref55__holdout__{stage}__s{seed}';folder=ROOT/'literature_results'/runid
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);split=data.split('holdout');t=Transform(data,stage,split);a={k:t.arrays(data,idx,'Middle') for k,idx in split.items()}
    torch.manual_seed(seed);model,hist=train_custom(MultiScale(a['train'][1].shape[1]),a['train'],a['val'],data.y[split['train']],data.y[split['val']],seed)
    x=embed(model,a['train']);reg=XGBRegressor(n_estimators=225,max_depth=10,learning_rate=.08,subsample=.94,reg_alpha=.18,reg_lambda=57.3,n_jobs=4,random_state=seed).fit(x,data.y[split['train']])
    pred=reg.predict(embed(model,a['test']))
    torch.save(model.cpu().state_dict(),folder/'feature_model.pt')
    finish(folder,data,split['test'],pred,{'run_id':runid,'reference':55,'stage':stage,'seed':seed,'scenario':'holdout','limitations':['Public abstract confirms multiscale attention + statistical features + XGBoost; exact architecture/training and Bayesian search unavailable.','Explicit local comparator: kernels 3/5/7, two conv layers per branch, global pooling, channel SE attention.','Reported XGBoost parameters used as fixed values; this is not a reproduction of the paper Bayesian optimization.'],'test_ids':sorted(set(data.ids[split['test']])),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},{'transform':t,'regressor':reg,'history':hist})

def run61(data,stage,seed):
    runid=f'ref61__holdout__{stage}__s{seed}';folder=ROOT/'literature_results'/runid
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);split=data.split('holdout');tr=split['train'];oof=np.zeros((len(tr),4));folds=[]
    for fold,(fit,held) in enumerate(GroupKFold(5).split(tr,groups=data.ids[tr])):
        sp={'train':tr[fit],'val':split['val'],'test':tr[held]}
        t,models,fn,hist=base_estimators(data,stage,sp,seed+fold)
        oof[held]=fn(tr[held]);folds.append({'fold':fold,'train_ids':sorted(set(data.ids[tr[fit]])),'oof_ids':sorted(set(data.ids[tr[held]])),'gp_training_cap':2000})
        del models
    # Unsupervised supplementary PCA, fitted only to outer training data.
    scaler=StandardScaler().fit(np.column_stack([data.hi[stage][tr],data.conditions[tr]]))
    pca=PCA(n_components=1).fit(scaler.transform(np.column_stack([data.hi[stage][tr],data.conditions[tr]])))
    supplement=lambda idx:pca.transform(scaler.transform(np.column_stack([data.hi[stage][idx],data.conditions[idx]])))
    meta=ELM(seed,width=32).fit(np.column_stack([oof,supplement(tr)]),data.y[tr])
    t,models,fn,hist=base_estimators(data,stage,split,seed);pred=meta.predict(np.column_stack([fn(split['test']),supplement(split['test'])]))
    gp,svr,elm,lstm=models;torch.save(lstm.cpu().state_dict(),folder/'lstm.pt')
    pd.DataFrame(oof,columns=['GPR','SVR','ELM','LSTM']).to_csv(folder/'oof_predictions.csv',index=False)
    finish(folder,data,split['test'],pred,{'run_id':runid,'reference':61,'stage':stage,'seed':seed,'scenario':'holdout','limitations':['GPR/SVR/ELM/LSTM with ELM meta learner and PCA supplement follows publicly visible structure only.','Stage-specific statistical features replace the paper full multidimensional charge/discharge feature set.','GPR capped at 2000 training-only sampled rows with fixed kernel and no optimizer; exact reproduction is not claimed.','LSTM 16 units, ELM 64/32 hidden units, PCA 1 component are local explicit choices.'],'oof_folds':folds,'test_ids':sorted(set(data.ids[split['test']])),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},{'transform':t,'gpr':gp,'svr':svr,'elm':elm,'meta':meta,'pca':pca,'pca_scaler':scaler,'history':hist})

def run15(data,scenario,stage,seed):
    runid=f'ref15__{scenario}__{stage}__s{seed}';folder=ROOT/'literature_results'/runid
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);split=data.split(scenario);target_ids=np.unique(data.ids[split['test']]);rng=np.random.RandomState(2025);rng.shuffle(target_ids)
    n=max(1,int(.2*len(target_ids)));adapt_ids=target_ids[:n];test_ids=target_ids[n:];assert len(test_ids)>0
    adapt=split['test'][np.isin(data.ids[split['test']],adapt_ids)];test=split['test'][np.isin(data.ids[split['test']],test_ids)]
    train=split['train'];scale=StandardScaler().fit(data.hi[stage][train]);source=scale.transform(data.hi[stage][train]);target=scale.transform(data.hi[stage][adapt]);rank=[]
    for bid in np.unique(data.ids[train]):
        xx=source[data.ids[train]==bid];rank.append((float(np.mean([wasserstein_distance(xx[:,i],target[:,i]) for i in range(6)])),bid))
    selected=[bid for _,bid in sorted(rank)[:max(2,int(.5*len(rank)))]]
    split={'train':train[np.isin(data.ids[train],selected)],'val':split['val'],'test':test}
    t=Transform(data,stage,split);a={k:t.arrays(data,idx,'Middle') for k,idx in split.items()};torch.manual_seed(seed)
    model,hist=train_custom(TBA(a['train'][1].shape[1]),a['train'],a['val'],data.y[split['train']],data.y[split['val']],seed);pred=predict(model,a['test'])
    torch.save(model.cpu().state_dict(),folder/'tba.pt')
    finish(folder,data,test,pred,{'run_id':runid,'reference':15,'stage':stage,'seed':seed,'scenario':scenario,'limitations':['Wasserstein source selection + TCN/BiLSTM/attention is a public-description comparator, not original SSTL reproduction.','Uses only unlabeled target adaptation cells for distribution distances; no target labels used in fitting.','Closest 50% of source training cells selected; TCN 16 channels and BiLSTM 16 units are explicit local choices.','No unknown original target fine-tuning schedule is invented.'],'adaptation_ids':adapt_ids.tolist(),'test_ids':test_ids.tolist(),'selected_source_ids':selected,'source_distance_ranking':rank,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},{'transform':t,'history':hist,'split':{k:v.tolist() for k,v in split.items()}})

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',choices=['55','61','15','all'],default='all');a=p.parse_args();data=Dataset()
    if a.reference in ['55','all']:
        for seed in CFG['seeds']:run55(data,'RE',seed)
    if a.reference in ['61','all']:
        for stage in STAGES:
            for seed in CFG['seeds']:run61(data,stage,seed)
    if a.reference in ['15','all']:
        for sc in ['temp35','temp45','rate025','rate05','rate1']:
            for stage in STAGES:
                for seed in CFG['seeds']:run15(data,sc,stage,seed)
if __name__=='__main__':main()
