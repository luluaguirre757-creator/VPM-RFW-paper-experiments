from __future__ import annotations
import numpy as np
from scipy.linalg import svdvals


def structured_theoretical_mse(factors, sets, noise_var=1.0, rank_tol=1e-10):
    """sigma^2 tr[(A^T A)^-1] for A=⊗_r U_r[L_r,:]."""
    traces=[]
    for U,idx in zip(factors,sets):
        A=U[np.asarray(idx,dtype=int),:]
        s=svdvals(A)
        if s.size < U.shape[1] or s[-1] <= rank_tol*s[0]: return float("inf")
        G=A.T@A
        traces.append(float(np.trace(np.linalg.inv(G))))
    return float(noise_var*np.prod(traces))



def structured_full_rank(factors, sets, rank_tol=1e-10):
    """Check the paper's downstream full-column-rank requirement mode by mode."""
    for U,idx in zip(factors,sets):
        A=np.asarray(U,float)[np.asarray(idx,dtype=int),:]
        s=svdvals(A)
        if s.size < U.shape[1] or s.size==0 or s[-1] <= rank_tol*max(float(s[0]),1e-30):
            return False
    return True

def unstructured_theoretical_mse(factors, tuples, noise_var=1.0, rank_tol=1e-10):
    rows=[]
    for tup in tuples:
        row=factors[-1][tup[-1]]
        for r in range(len(factors)-2,-1,-1): row=np.kron(row,factors[r][tup[r]])
        rows.append(row)
    A=np.asarray(rows)
    s=svdvals(A)
    K=int(np.prod([u.shape[1] for u in factors]))
    if len(s)<K or s[-1] <= rank_tol*s[0]: return float("inf")
    return float(noise_var*np.trace(np.linalg.inv(A.T@A)))


def nmse_db(xhat,x,eps=1e-30):
    val=np.sum((xhat-x)**2)/(np.sum(x*x)+eps)
    return float(10*np.log10(max(val,eps)))


def psnr(xhat,x,data_range=None):
    if data_range is None: data_range=float(x.max()-x.min())
    mse=float(np.mean((xhat-x)**2))
    return float(20*np.log10(max(data_range,1e-12))-10*np.log10(max(mse,1e-30)))


def spectral_angle_cube(xhat,x,eps=1e-12):
    A=x.reshape(-1,x.shape[-1]); B=xhat.reshape(-1,xhat.shape[-1])
    cs=np.sum(A*B,axis=1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+eps)
    return float(np.mean(np.arccos(np.clip(cs,-1,1))))
