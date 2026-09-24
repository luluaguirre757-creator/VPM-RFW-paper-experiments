from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def normalize_rows(U):
    n=np.linalg.norm(U,axis=1,keepdims=True)
    return U/np.maximum(n,1e-15)


def apply_energy_preserving_lognormal_amplitudes(directions,rng,sigma):
    N=directions.shape[0]
    if sigma==0: return directions.copy()
    a=np.exp(sigma*rng.standard_normal(N))
    a=np.sqrt(N)*a/np.sqrt(np.sum(a*a))
    return directions*a[:,None]


def orthonormal_like_directions(N,K,rng):
    # gaussian directions; row-normalization is experimental control, not column orthogonalization
    return normalize_rows(rng.standard_normal((N,K)))

@dataclass
class SyntheticTask:
    nominal:list[np.ndarray]
    design_scenarios:list[list[np.ndarray]]
    test_scenarios:list[list[np.ndarray]]
    regime:str


def _perturb(base,rng,scale,row_normalized,amps=None):
    # Perturb *directions* rather than amplitude-scaled rows.  This is crucial for the
    # controlled normalized/unnormalized comparison: with the same seed, both settings
    # see identical directional/scenario perturbations and differ only in row norms.
    base_dirs=normalize_rows(base)
    dirs=normalize_rows(base_dirs + scale*rng.standard_normal(base.shape)/np.sqrt(base.shape[1]))
    if row_normalized: return dirs
    if amps is None: amps=np.linalg.norm(base,axis=1)
    return dirs*amps[:,None]


def make_sequence(Ns,Ks,S,seed,row_sigma=0.0,sequence="recurrent",drift=0.02,
                  design_scales=(0,.005,.01,.02,.04),J_test=50,test_max=.05):
    # Independent deterministic RNG streams prevent held-out evaluation choices (e.g. J_test)
    # from changing the task trajectory or design scenarios.
    rng=np.random.default_rng(seed)
    # Scenario streams are keyed by task below, so changing the number of held-out
    # scenarios on one task cannot change later tasks.
    labels=[]
    if sequence=="recurrent":
        base_labels=["A","B","A","C","B"]
        block=int(np.ceil(S/5)); labels=(sum(([x]*block for x in base_labels),[]))[:S]
    elif sequence=="smooth": labels=["A"]*S
    elif sequence=="abrupt": labels=["A"]*(S//2)+["C"]*(S-S//2)
    else: raise ValueError(sequence)
    anchors={lab:[orthonormal_like_directions(N,K,rng) for N,K in zip(Ns,Ks)] for lab in sorted(set(labels))}
    amp_rng=np.random.default_rng(seed+1000003)
    amps={lab:[] for lab in anchors}
    for lab,fac in anchors.items():
        for U in fac:
            if row_sigma==0: amps[lab].append(np.ones(U.shape[0]))
            else:
                a=np.exp(row_sigma*amp_rng.standard_normal(U.shape[0])); a=np.sqrt(U.shape[0])*a/np.sqrt(np.sum(a*a)); amps[lab].append(a)
    states={lab:[U.copy() for U in anchors[lab]] for lab in anchors}
    tasks=[]; prev_lab=None
    for s,lab in enumerate(labels):
        # recurrence restarts at stored regime state; each occurrence drifts locally
        fac=[]
        for r,(U0,a) in enumerate(zip(states[lab],amps[lab])):
            dirs=normalize_rows(U0 + drift*rng.standard_normal(U0.shape)/np.sqrt(U0.shape[1]))
            states[lab][r]=dirs
            fac.append(dirs if row_sigma==0 else dirs*a[:,None])
        design_rng=np.random.default_rng(seed+2000003+104729*s)
        test_rng=np.random.default_rng(seed+3000017+130363*s)
        design=[]
        for sc in design_scales:
            design.append([_perturb(U,design_rng,sc,row_sigma==0) for U in fac])
        test=[]
        for _ in range(J_test):
            sc=test_rng.uniform(0,test_max)
            test.append([_perturb(U,test_rng,sc,row_sigma==0) for U in fac])
        tasks.append(SyntheticTask(fac,design,test,lab))
        prev_lab=lab
    return tasks
