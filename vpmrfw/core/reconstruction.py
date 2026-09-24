from __future__ import annotations
import numpy as np


def mode_product(X: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    """Return X ×_mode M, with M shape (new_dim, old_dim)."""
    X=np.asarray(X,float); M=np.asarray(M,float)
    if M.shape[1] != X.shape[mode]:
        raise ValueError("mode-product dimension mismatch")
    Y=np.tensordot(M,X,axes=(1,mode))
    # tensordot puts the new dimension first; move it to `mode`.
    return np.moveaxis(Y,0,mode)


def synthesize_tucker(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    X=np.asarray(core,float)
    for r,U in enumerate(factors):
        X=mode_product(X,np.asarray(U,float),r)
    return X


def structured_reconstruct(observed_subtensor: np.ndarray,
                           factors: list[np.ndarray],
                           sets: list[np.ndarray]) -> np.ndarray:
    """LS Tucker reconstruction from a Cartesian-product sample set.

    For sampled factors A_r=U_r[L_r,:], the LS core is obtained by multiplying
    the observed subtensor by A_r^dagger in every mode.  This is algebraically
    equivalent to the vectorized LS solution for the Kronecker design.
    """
    G=np.asarray(observed_subtensor,float)
    for r,(U,idx) in enumerate(zip(factors,sets)):
        A=np.asarray(U,float)[np.asarray(idx,dtype=int),:]
        if np.linalg.matrix_rank(A) < A.shape[1]:
            raise np.linalg.LinAlgError("selected factor is rank deficient; LS reconstruction is not admissible")
        G=mode_product(G,np.linalg.pinv(A),r)
    return synthesize_tucker(G,factors)


def empirical_structured_nmse(factors:list[np.ndarray], sets:list[np.ndarray], rng:np.random.Generator,
                              snr_db:float=20.0, trials:int=10) -> np.ndarray:
    """Dense-core Monte-Carlo NMSE with a method-independent noise convention.

    For a fixed RNG seed this function generates the same latent core and the same *full*
    noise tensor for every sampling method. Noise variance is set from the power of the full
    clean tensor, not from method-dependent sampled entries. Calling this function with a
    freshly initialized RNG using the same seed therefore gives a genuinely paired comparison.
    """
    Ks=[U.shape[1] for U in factors]
    sets=[np.asarray(s,dtype=int) for s in sets]
    out=[]
    for _ in range(int(trials)):
        core=rng.standard_normal(Ks)
        X=synthesize_tucker(core,factors)
        p=float(np.mean(X*X))
        nv=p/(10.0**(float(snr_db)/10.0))
        noise=rng.normal(0.0,np.sqrt(max(nv,1e-30)),size=X.shape)
        Y=(X+noise)[np.ix_(*sets)]
        Xh=structured_reconstruct(Y,factors,sets)
        nm=float(np.sum((Xh-X)**2)/(np.sum(X*X)+1e-30))
        out.append(10.0*np.log10(max(nm,1e-30)))
    return np.asarray(out,float)
