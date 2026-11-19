"""Independent forward reference + leakage and provenance acceptance checks."""
from pathlib import Path
import json,hashlib
import h5py,numpy as np,torch
from src.data import Dataset,Transform,ROOT,metrics
from src.models import Net

def inbound(x):
    if isinstance(x,dict):
        if x.get('class_name')=='__keras_tensor__':yield x['config']['keras_history'][0]
        else:
            for v in x.values():yield from inbound(v)
    elif isinstance(x,list):
        if len(x)>=3 and isinstance(x[0],str) and isinstance(x[1],int):yield x[0]
        else:
            for v in x:yield from inbound(v)

def read_h5(p):
    with h5py.File(p) as f:
        cfg=json.loads(f.attrs['model_config'])['config'];weights={}
        for l in cfg['layers']:
            if l['class_name'] not in ['Conv1D','Dense']:continue
            datasets={}
            f['model_weights'][l['name']].visititems(lambda n,o:datasets.update({n.split('/')[-1].split(':')[0]:o[()]}) if isinstance(o,h5py.Dataset) else None)
            weights[l['name']]=datasets
    return cfg,weights

def reference(cfg,weights,inputs):
    values={};pos=0
    for l in cfg['layers']:
        kind=l['class_name'];c=l['config'];name=l['name'];names=list(inbound(l.get('inbound_nodes',[])))
        if kind=='InputLayer':values[name]=inputs[pos];pos+=1;continue
        xs=[values[n] for n in names];x=xs[0]
        if kind=='Conv1D':
            k=weights[name]['kernel'];b=weights[name]['bias'];stride=c['strides'][0]
            x=np.stack([np.einsum('nkc,kco->no',x[:,j:j+len(k)],k)+b for j in range(0,x.shape[1]-len(k)+1,stride)],axis=1)
        elif kind=='MaxPooling1D':
            w=c['pool_size'][0];s=c['strides'][0]
            x=np.stack([x[:,j:j+w].max(1) for j in range(0,x.shape[1]-w+1,s)],axis=1)
        elif kind=='Flatten':x=x.reshape(len(x),-1)
        elif kind=='Dense':x=x@weights[name]['kernel']+weights[name]['bias']
        elif kind=='Concatenate':x=np.concatenate(xs,axis=c['axis'])
        elif kind=='Dropout':pass
        else:raise ValueError(kind)
        if c.get('activation')=='relu':x=np.maximum(x,0)
        values[name]=x
    return values[cfg['layers'][-1]['name']].reshape(-1)

def convert(cfg,weights,model):
    layer_by_name={l['name']:l for l in cfg['layers']}
    inputs=[l['name'] for l in cfg['layers'] if l['class_name']=='InputLayer']
    def ancestors(name):
        if name in inputs:return {name}
        deps=list(inbound(layer_by_name[name].get('inbound_nodes',[])))
        return set().union(*(ancestors(d) for d in deps))
    convs=[l for l in cfg['layers'] if l['class_name']=='Conv1D']
    for src,dst in zip(convs,[m for m in model.modules() if isinstance(m,torch.nn.Conv1d)]):
        w=weights[src['name']];dst.weight.data.copy_(torch.tensor(w['kernel'].transpose(2,1,0)));dst.bias.data.copy_(torch.tensor(w['bias']))
    dense=[l for l in cfg['layers'] if l['class_name']=='Dense']
    if model.kind in ['CNN','Early']:groups=[(dense,model.head)]
    else:
        branch1=[l for l in dense if ancestors(l['name'])=={inputs[0]}]
        branch2=[l for l in dense if ancestors(l['name'])=={inputs[1]}]
        head=[l for l in dense if len(ancestors(l['name']))==2]
        groups=[(branch1,model.chead),(branch2,model.hhead),(head,model.head)]
    for source,module in groups:
        targets=[m for m in module.modules() if isinstance(m,torch.nn.Linear)];assert len(source)==len(targets)
        for src,dst in zip(source,targets):
            w=weights[src['name']];dst.weight.data.copy_(torch.tensor(w['kernel'].T));dst.bias.data.copy_(torch.tensor(w['bias']))

def main():
    records=[]
    for kind in ['CNN','Early','Middle','Late']:
        for stage in ['CC','CV','RE']:
            rel={'CNN':f'4-SOH_Estimation/深度学习模型结果/model_{stage}.h5','Early':f'5-fusion/model/early/model_{stage}.h5','Middle':f'5-fusion/rusult/middle/model_{stage}.h5','Late':f'5-fusion/rusult/late/model_{stage}.h5'}[kind]
            cfg,weights=read_h5(ROOT/'archive/submission'/rel)
            rng=np.random.RandomState(821);a=rng.normal(size=(7,153 if kind=='CNN' else 154 if kind=='Early' else 151,1)).astype(np.float32)
            h=rng.normal(size=(7,3)).astype(np.float32)
            inputs=[a] if kind in ['CNN','Early'] else [a,h]
            yref=reference(cfg,weights,inputs)
            model=Net(kind,153 if kind=='CNN' else 151,3);convert(cfg,weights,model);model.eval()
            c=torch.tensor(a[:,:,0] if kind!='Early' else a[:,:151,0]);hi=torch.tensor(a[:,151:,0] if kind=='Early' else h)
            with torch.no_grad():yp=model(c,hi).numpy()
            error=float(abs(yp-yref).max());assert error<5e-5,(kind,stage,error)
            records.append({'model':kind,'stage':stage,'reference':'independent NumPy evaluation of saved Keras H5 weights/config','max_abs_forward_difference':error,'parameters':sum(p.numel() for p in model.parameters())})
    data=Dataset();split=data.split('holdout');t=Transform(data,'RE',split)
    selected=t.selected.copy();before=t.state();old_y=data.y[split['test']].copy();data.y[split['test']]=np.linspace(-100,100,len(split['test']))
    new=Transform(data,'RE',split);assert new.state()==before
    data.y[split['test']]=old_y
    assert len(split['test'])==4040 and len(split['train'])==13762
    for name in ['holdout']+[f'cv{i}' for i in range(5)]+['temp35','temp45','rate025','rate05','rate1']:
        sp=data.split(name);assert all(len(x)>0 for x in sp.values())
        groups=[set(data.ids[x]) for x in sp.values()];assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    cv_tests=np.concatenate([data.split(f'cv{i}')['test'] for i in range(5)])
    assert sorted(cv_tests.tolist())==list(range(len(data.y)))
    saved=np.asarray(__import__('pandas').read_pickle(ROOT/'data/HI.pkl')['RE_Max'])
    assert np.max(abs(saved-data.curves['RE'][:,1]))==0
    domain_split=data.split('temp45');dt=Transform(data,'RE',domain_split)
    _,domain_hi=dt.arrays(data,domain_split['test'][:10],'Late')
    assert np.all(domain_hi[:,-2:]==0), 'Source-constant temperature and rate must remain masked at target inference.'
    torch.manual_seed(730)
    cnn=Net('CNN',153,3,input_layout='fair_v3');cnn.initialize(.85);cnn.eval()
    curve=torch.randn(32,153,requires_grad=True);cnn(curve,None).sum().backward()
    assert (curve.grad[:,-2:].abs().sum(0)>1e-7).all(), 'Both CNN conditions must have a live path to output.'
    early=Net('Early',151,5,input_layout='fair_v3');early.initialize(.85);early.eval()
    curve=torch.randn(32,151);static=torch.randn(32,5,requires_grad=True);early(curve,static).sum().backward()
    assert (static.grad.abs().sum(0)>1e-7).all(), 'All early static inputs must have a live path through convolution.'
    legacy=Net('Early',151,5);legacy.initialize(.85);legacy.eval();old_static=torch.randn(32,5,requires_grad=True);legacy(curve,old_static).sum().backward()
    assert (old_static.grad==0).all(), 'The archived tail-discard issue must remain reproducible.'
    report={'passed':True,'parity_models':records,'no_test_labels_used_in_selection_or_scaling':True,'cv_test_partition_covers_every_sample_once':True,'historical_RE_Max_equals_RE_1_for_all_samples':True,'source_constant_conditions_masked_without_target_labels':True,'fair_cnn_and_early_static_input_gradient_paths_verified':True,'archived_early_tail_discard_bug_reproduced':True,'source_archives_unchanged':True}
    for record in json.loads((ROOT/'archive/source_manifest.json').read_text(encoding='utf-8')):
        assert hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()==record['sha256']
    (ROOT/'reports/acceptance_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
