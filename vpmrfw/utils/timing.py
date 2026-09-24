from __future__ import annotations
import os, time
import numpy as np

_THREAD_VARS = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
]

def pin_threads(n=1):
    """Force the experiment process to a common BLAS/OpenMP thread count."""
    for k in _THREAD_VARS:
        os.environ[k] = str(int(n))

def median_runtime(fn, warmups=1, repeats=5):
    """Run ``fn`` after warm-up and return (median seconds, all timed seconds)."""
    for _ in range(int(warmups)):
        fn()
    vals=[]
    for _ in range(int(repeats)):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return float(np.median(vals)), np.asarray(vals,float)
