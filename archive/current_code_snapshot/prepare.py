"""Snapshot the submission sources; this script never changes upstream files."""
from pathlib import Path
import hashlib, json, shutil
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent.parent/'4'/'code'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=[]
    for p in sorted(SOURCE.rglob('*')):
        if not p.is_file() or '.ipynb_checkpoints' in p.parts: continue
        rel=p.relative_to(SOURCE)
        if p.suffix.lower() not in {'.py','.ipynb','.pkl','.csv','.h5','.xlsx'}:continue
        dest=ROOT/'archive'/'submission'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
        assert digest(dest)==digest(p)
        manifest.append({'source':str(p.resolve()),'archive':str(dest.relative_to(ROOT)), 'sha256':digest(p),'bytes':p.stat().st_size})
    (ROOT/'archive'/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['data.pkl','HI.pkl']:
        dest=ROOT/'data'/name
        if not dest.exists():shutil.copy2(SOURCE/'data'/name,dest)
        assert digest(dest)==digest(SOURCE/'data'/name)
    print(f'Archived and verified {len(manifest)} source files.',flush=True)
if __name__=='__main__':main()
