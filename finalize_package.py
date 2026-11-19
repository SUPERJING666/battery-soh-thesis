from pathlib import Path
import json,hashlib,shutil
import pandas as pd
from src.data import ROOT,Dataset
from run import code_hash

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    data=Dataset();records=[];scenarios=['holdout']+[f'cv{i}' for i in range(5)]+['temp35','temp45','rate025','rate05','rate1']
    for scenario in scenarios:
        for part,idx in data.split(scenario).items():
            for bid,g in data.meta.iloc[idx].groupby('id'):
                records.append({'scenario':scenario,'split':part,'battery_id':bid,'samples':len(g),'min_cycle':int(g.cycle.min()),'max_cycle':int(g.cycle.max())})
    pd.DataFrame(records).to_csv(ROOT/'reports/battery_split_manifest.csv',index=False,encoding='utf-8-sig')
    source=ROOT/'archive/training_code_v1';text=(ROOT/'literature.py').read_text(encoding='utf-8').replace('from run import predict,code_hash','from run import predict')
    text=text.replace("    metadata['pipeline_code_hash']=code_hash();metadata['protocol_version']=CFG['protocol_version']\n",'')
    text=text.replace("    metadata['custom_neural_training']={'max_epochs':100,'patience':15,'batch_size':512,'optimizer':'Adam','lr':.001,'final_bias':'source_training_label_mean'}\n",'')
    (source/'literature.py').write_bytes(text.encode('utf-8'))
    reference=json.loads((ROOT/'literature_results/ref55__holdout__RE__s2025/result.json').read_text(encoding='utf-8'))['source_code_sha256'];assert sha(source/'literature.py')==reference
    # Earlier snapshots are immutable, not reconstructed from later revisions.
    for version in ['training_code_v1','training_code_v2']:
        old=ROOT/'archive'/version;paths=[old/'run.py',old/'configs/protocol.json']+sorted((old/'src').glob('*.py'))
        expected=json.loads((old/'fingerprint.json').read_text(encoding='utf-8'))['code_hash']
        assert hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()==expected
    dest=ROOT/'archive/current_code_snapshot';dest.mkdir(exist_ok=True);manifest=[]
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'requirements.lock.txt']+list((ROOT/'src').glob('*.py'))+list((ROOT/'configs').glob('*.json')):
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);manifest.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    (ROOT/'reports/current_code_manifest.json').write_text(json.dumps({'active_code_hash':code_hash(),'files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{'filename':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in (ROOT/'data').glob('*.pkl')]).to_csv(ROOT/'reports/data_manifest.csv',index=False)
    print('Code snapshots v1/v2 and current, data/split manifests verified.',flush=True)
if __name__=='__main__':main()
