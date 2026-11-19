from pathlib import Path
import json
import numpy as np,pandas as pd,torch
from src.data import ROOT,Dataset
from analyze import load
from run import predict

def main():
    data=Dataset();lookup={(x.id,x.cycle):i for i,x in data.meta.iterrows()};records=[]
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for p in sorted((ROOT/'results').glob('*/result.json')):
        meta,t,model=load(p.parent,'cpu');d=pd.read_csv(p.parent/'predictions.csv');sample=d.iloc[[0,len(d)//2,len(d)-1]]
        idx=np.array([lookup[(r.id,r.cycle)] for _,r in sample.iterrows()]);arrays=t.arrays(data,idx,meta['model'])
        if isinstance(model,torch.nn.Module):
            with torch.no_grad():pred=model(*[torch.tensor(x) if x is not None else None for x in arrays]).numpy()
        else:pred=model.predict(arrays[1])
        error=float(abs(pred-sample.pred.to_numpy()).max())
        # CPU vs NVIDIA TF32 convolution is a platform-precision diagnostic,
        # not a serialization check. Re-evaluate on the original device and
        # with the original batch shape for the actual checkpoint assertion.
        neural=isinstance(model,torch.nn.Module)
        if neural:model=model.cuda()
        idx_all=np.array([lookup[(r.id,r.cycle)] for _,r in d.iterrows()]);all_arrays=t.arrays(data,idx_all,meta['model'])
        repeat=predict(model,all_arrays);same_device_error=float(abs(repeat-d.pred.to_numpy()).max())
        assert same_device_error<1e-6,(meta['run_id'],same_device_error)
        records.append({'run_id':meta['run_id'],'samples_checked_on_original_device':len(d),'original_device_reload_max_abs_difference':same_device_error,'cpu_three_sample_max_abs_difference':error})
    (ROOT/'reports/checkpoint_roundtrip_checks.json').write_text(json.dumps({'passed':True,'runs_checked':len(records),'original_device_tolerance_normalized_SOH':1e-6,'cpu_comparison_note':'CPU versus GPU convolution precision differs because original GPU training/inference permits cuDNN TF32; CPU diagnostic is not claimed to be bit-identical.','cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'records':records},indent=2),encoding='utf-8')
    print('Checkpoint reload checks passed:',len(records),flush=True)
if __name__=='__main__':main()
