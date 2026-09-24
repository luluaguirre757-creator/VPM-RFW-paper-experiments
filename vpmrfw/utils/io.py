from __future__ import annotations
from pathlib import Path
import json, shutil, hashlib, platform, sys
import yaml

def load_yaml(p): return yaml.safe_load(Path(p).read_text())
def dump_yaml(x,p): Path(p).write_text(yaml.safe_dump(x,sort_keys=False))
def source_manifest(root,out):
    root=Path(root).resolve(); out=Path(out).resolve(); rows=[]
    skip_parts={'results','.pytest_cache','__pycache__','.venv','.git'}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel=p.relative_to(root)
        if p.resolve()==out or any(part in skip_parts or part.startswith('results_') for part in rel.parts):
            continue
        rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(rows)+'\n')
def environment(out):
    Path(out).write_text(json.dumps({'python':sys.version,'platform':platform.platform()},indent=2))
def archive_dir(path): return shutil.make_archive(str(path),'zip',root_dir=str(path))
