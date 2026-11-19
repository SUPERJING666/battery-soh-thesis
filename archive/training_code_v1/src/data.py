from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
STAGES=['CC','CV','RE']
STATS=['Var','Ske','Max','Min','Mean','Kur']

def features(curves):
    x=np.asarray(curves,dtype=np.float64)[:,1:150]
    return np.column_stack([np.var(x,axis=1),skew(x,axis=1,bias=False),x.max(1),x.min(1),x.mean(1),kurtosis(x,axis=1,bias=False)])

class Dataset:
    def __init__(self):
        self.df=pd.read_pickle(ROOT/'data/data.pkl')
        assert not self.df.duplicated(['id','cycle']).any()
        self.ids=self.df.id.to_numpy()
        self.y=self.df.capacity.to_numpy(np.float64)/3500
        self.curves={s:self.df[[f'{s}_{i}' for i in range(151)]].to_numpy(np.float64) for s in STAGES}
        self.hi={s:features(self.curves[s]) for s in STAGES}
        def condition(x):
            a,b=x.split('-')[:2];r=b.split('_')[0]
            return float(a),{'025':.25,'05':.5,'1':1.}[r]
        self.conditions=np.array([condition(x) for x in self.ids],np.float64)
        self.meta=self.df[['id','cycle','capacity']].copy()

    def split(self,name):
        ids=self.df.id.unique().copy(); rng=np.random.RandomState(2025);rng.shuffle(ids)
        if name=='holdout':groups=[ids[:39],ids[39:52],ids[52:]]
        elif name.startswith('cv'):
            fold=int(name[2:]); splitter=GroupKFold(n_splits=5)
            train,test=list(splitter.split(self.ids,groups=self.ids))[fold]
            dev=np.unique(self.ids[train]);rng.shuffle(dev);n=max(1,int(len(dev)*.2))
            groups=[dev[n:],dev[:n],np.unique(self.ids[test])]
        elif name.startswith('temp'):
            target=int(name[4:]); source_ids=np.unique(self.ids[(self.conditions[:,0]==25)&(self.conditions[:,1]==.5)])
            rng.shuffle(source_ids);n=max(1,int(len(source_ids)*.2))
            groups=[source_ids[n:],source_ids[:n],np.unique(self.ids[(self.conditions[:,0]==target)&(self.conditions[:,1]==.5)])]
        elif name.startswith('rate'):
            target={'025':.25,'05':.5,'1':1.}[name[4:]]
            source_ids=np.unique(self.ids[(self.conditions[:,0]==25)&(self.conditions[:,1]!=target)])
            rng.shuffle(source_ids);n=max(1,int(len(source_ids)*.2))
            groups=[source_ids[n:],source_ids[:n],np.unique(self.ids[(self.conditions[:,0]==25)&(self.conditions[:,1]==target)])]
        else:raise ValueError(name)
        sets=[set(g) for g in groups]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        return {k:np.flatnonzero(np.isin(self.ids,g)) for k,g in zip(['train','val','test'],groups)}

def correlations(x,y):
    def c(a,b):
        a=a-a.mean(0);b=b-b.mean();den=np.sqrt((a*a).sum(0)*(b*b).sum())
        return np.divide((a*b[:,None]).sum(0),den,out=np.zeros(a.shape[1]),where=den>1e-16)
    p=np.abs(c(x,y));s=np.abs(c(np.apply_along_axis(rankdata,0,x),rankdata(y)))
    def norm(a): return (a-a.min())/(a.max()-a.min()) if a.max()>a.min() else np.zeros_like(a)
    return p,s,norm(p)+norm(s)

class Transform:
    def __init__(self,data,stage,split,track='fair',input_mode='joint'):
        self.stage=stage;self.track=track;self.input_mode=input_mode
        tr=split['train'];p,s,score=correlations(data.hi[stage][tr],data.y[tr])
        self.selected=np.argsort(-score,kind='stable')[:3]
        self.feature_names=[stage+'_'+STATS[i] for i in self.selected]
        self.correlations={'pearson_abs':p.tolist(),'spearman_abs':s.tolist(),'score':score.tolist()}
        self.curve_scaler=StandardScaler().fit(data.curves[stage][tr])
        self.hi_scaler=StandardScaler().fit(data.hi[stage][tr][:,self.selected])
        self.condition_scaler=StandardScaler().fit(data.conditions[tr])

    def arrays(self,data,idx,model,curve_override=None,conditions_override=None):
        raw=data.curves[self.stage][idx] if curve_override is None else curve_override
        h=features(raw)[:,self.selected]
        c=self.curve_scaler.transform(raw)
        h=self.hi_scaler.transform(h)
        cond=data.conditions[idx] if conditions_override is None else conditions_override
        cond=self.condition_scaler.transform(cond)
        use_cond=self.track=='fair' and self.input_mode!='no_conditions'
        if self.track=='legacy':use_cond=model in ['RF','GBDT','NN','AdaBoost','SVR','XGBoost','CNN']
        if self.input_mode in ['hi','curve']:use_cond=False
        if model in ['RF','GBDT','NN','AdaBoost','SVR','XGBoost','MLP','conditions']:
            inp=cond if model=='conditions' else np.column_stack([h,cond]) if use_cond else h
            if self.track=='legacy' and model!='CNN':inp=np.column_stack([inp,np.zeros(len(inp))])
            return None,np.asarray(inp,np.float32)
        if model=='CNN':return np.asarray(np.column_stack([c,cond]) if use_cond else c,np.float32),None
        return np.asarray(c,np.float32),np.asarray(np.column_stack([h,cond]) if use_cond else h,np.float32)

    def state(self):
        state={'stage':self.stage,'track':self.track,'input_mode':self.input_mode,'selected':self.selected.tolist(),'feature_names':self.feature_names,'correlations':self.correlations}
        for name in ['curve_scaler','hi_scaler','condition_scaler']:
            sc=getattr(self,name);state[name]={'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist()}
        return state

def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64);e=p-y
    mse=float(np.mean(e*e));den=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'mse':mse,'rmse_pp':float(100*np.sqrt(mse)),'mae_pp':float(100*np.mean(abs(e))),'r2':float(1-np.sum(e*e)/den) if den>0 else None}
