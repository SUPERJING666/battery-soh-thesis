import json,pickle
import numpy as np
from src.data import ROOT

def main():
    for p in (ROOT/'results').glob('*/benchmark_cpu.json'):
        row=json.loads(p.read_text(encoding='utf-8'));folder=p.parent
        if row['parameters'] is not None:row['structure_summary']=str(int(row['parameters']))+' parameters'
        else:
            with (folder/'model.pkl').open('rb') as f:m=pickle.load(f)
            if row['model']=='NN':
                n=sum(x.size for x in m.coefs_)+sum(x.size for x in m.intercepts_);row['parameters']=n;row['structure_summary']=str(n)+' parameters'
            elif row['model']=='SVR':row['support_vectors']=len(m.support_);row['structure_summary']=str(len(m.support_))+' support vectors'
            elif row['model']=='XGBoost':
                trees=m.get_booster().get_dump(dump_format='json')
                def count(x):return 1+sum(count(c) for c in x.get('children',[]))
                row['tree_count']=len(trees);row['tree_nodes']=sum(count(json.loads(t)) for t in trees);row['structure_summary']=f'{row["tree_count"]} trees / {row["tree_nodes"]} nodes'
            else:
                trees=np.asarray(m.estimators_,dtype=object).ravel();row['tree_count']=len(trees);row['tree_nodes']=sum(x.tree_.node_count for x in trees);row['structure_summary']=f'{len(trees)} trees / {row["tree_nodes"]} nodes'
        p.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('CPU structure measures completed.',flush=True)
if __name__=='__main__':main()
