"""Independent process per CPU measurement; GPU measurements stay serial."""
from pathlib import Path
import json,subprocess,sys,argparse
from src.data import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--force',action='store_true');a=p.parse_args()
    count=0
    for f in sorted((ROOT/'results').glob('*/result.json')):
        m=json.loads(f.read_text(encoding='utf-8'))
        if m['scenario']!='holdout' or m['track']!='fair' or m['input_mode']!='joint' or m['seed']!=2025:continue
        if m['model'] not in ['RF','GBDT','NN','AdaBoost','SVR','XGBoost','CNN','Early','Middle','Late']:continue
        if a.device=='cuda' and m['model'] not in ['CNN','Early','Middle','Late']:continue
        if (f.parent/f'benchmark_{a.device}.json').exists() and not a.force:count+=1;continue
        subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--task','benchmark','--run',m['run_id'],'--device',a.device],check=True)
        count+=1
    print('benchmark completed',a.device,count,flush=True)
if __name__=='__main__':main()
