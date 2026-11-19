"""Independent checkpoint inference on its recorded, untouched test cohort."""
import argparse,json
from pathlib import Path
import torch
from src.data import ROOT,Dataset,metrics
from analyze import load
from run import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',required=True);a=p.parse_args()
    folder=(ROOT/'results'/a.run).resolve();assert folder.is_relative_to(ROOT/'results')
    output=Path(a.output).resolve();assert output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'results'),'Write a new derivative under this code package; accepted results cannot be overwritten.'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    meta,t,model=load(folder,a.device);data=Dataset();idx=data.split(meta['scenario'])['test']
    assert set(data.ids[idx])==set(meta['split']['test']['ids'])
    values=predict(model,t.arrays(data,idx,meta['model']));df=data.meta.iloc[idx].reset_index(drop=True);df['true']=data.y[idx];df['pred']=values
    output.parent.mkdir(parents=True,exist_ok=True);df.to_csv(output,index=False,float_format='%.17g')
    print(json.dumps({'run_id':a.run,'device':a.device,'output':str(output),'metrics':metrics(df.true,df.pred)},ensure_ascii=False))
if __name__=='__main__':main()
