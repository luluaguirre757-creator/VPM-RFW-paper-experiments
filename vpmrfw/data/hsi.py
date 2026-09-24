from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.io import loadmat

DATASET_SPECS={
 "KSC": {
     "file":"KSC.mat","key":"KSC","shape":(512,614,176),
     "canonical_url":"https://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
     "sensor":"AVIRIS","note":"176 retained bands after standard noisy/water-absorption removal"
 },
 "Salinas": {
     "file":"Salinas_corrected.mat","key":"salinas_corrected","shape":(512,217,204),
     "canonical_url":"https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
     "sensor":"AVIRIS","note":"standard corrected 204-band cube"
 },
}


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def audit_hsi(name,data_dir):
    spec=DATASET_SPECS[name]; p=Path(data_dir)/spec["file"]
    row={"dataset":name,"path":str(p),"exists":p.exists(),"canonical_url":spec["canonical_url"],
         "expected_key":spec["key"],"expected_shape":list(spec["shape"]),"sensor":spec["sensor"]}
    if not p.exists(): return row
    row["sha256"]=sha256_file(p); row["size_bytes"]=p.stat().st_size
    try:
        d=loadmat(p,variable_names=[spec["key"]])
        row["key_present"]=spec["key"] in d
        if row["key_present"]:
            x=np.asarray(d[spec["key"]]); row["actual_shape"]=list(x.shape)
            row["shape_ok"]=tuple(x.shape)==tuple(spec["shape"])
            row["finite"]=bool(np.all(np.isfinite(x)))
    except Exception as e: row["load_error"]=repr(e)
    return row


def write_data_audit(names,data_dir,out):
    rows=[audit_hsi(n,data_dir) for n in names]
    Path(out).write_text(json.dumps(rows,indent=2))
    return rows


def load_hsi(name,data_dir):
    spec=DATASET_SPECS[name]; p=Path(data_dir)/spec["file"]
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Use scripts/download_hsi.py or place the canonical MAT file manually; see data/README.md")
    d=loadmat(p,variable_names=[spec["key"]])
    if spec["key"] not in d: raise KeyError(f"{p} lacks key {spec['key']}")
    cube=np.asarray(d[spec["key"]],dtype=np.float32)
    if cube.shape!=spec["shape"]: raise ValueError(f"{name} expected shape {spec['shape']}, got {cube.shape}")
    if not np.all(np.isfinite(cube)): raise ValueError(f"{name} contains non-finite values")
    return cube


def rank_factors_from_band(band,rank):
    # SVD is performed in float64 for numerical stability, one band at a time.
    U,s,Vt=np.linalg.svd(np.asarray(band,dtype=np.float64),full_matrices=False)
    k=min(int(rank),len(s)); root=np.sqrt(s[:k])
    return [U[:,:k]*root[None,:], Vt[:k,:].T*root[None,:]], s


def validate_no_leakage(design_band_indices,current):
    if any(int(i)>=int(current) for i in design_band_indices):
        raise AssertionError("future/current band used in sampling design")
