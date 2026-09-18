"""Neighborhood agreement, displacement bounds and coordinate-consistent fusion."""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def guarded(s,n,groups):
    residual=n-s
    overlap=np.zeros(len(s));radius=np.zeros(len(s))
    for group in np.unique(groups):
        idx=np.flatnonzero(groups==group);k=min(20,len(idx)-1)
        if k<1:continue
        dist,si=NearestNeighbors(n_neighbors=k+1,n_jobs=1).fit(s[idx]).kneighbors(s[idx])
        _,ni=NearestNeighbors(n_neighbors=k+1,n_jobs=1).fit(n[idx]).kneighbors(n[idx])
        for j in range(len(idx)):
            a=si[j][si[j]!=j][:k];b=ni[j][ni[j]!=j][:k]
            overlap[idx[j]]=len(set(a)&set(b))/k
            radius[idx[j]]=np.median(dist[j][si[j]!=j][:k])
    weight=.25*overlap
    norm=np.linalg.norm(residual,axis=1)
    weight=np.minimum(weight,.25*radius/np.maximum(norm,1e-12))
    result=s+weight[:,None]*residual
    assert np.all(np.linalg.norm(result-s,axis=1)<=.25*radius+1e-5)
    return result,dict(mean_weight=float(weight.mean()),mean_neighbor_agreement=float(overlap.mean()),max_displacement_radius_ratio=float(np.max(np.linalg.norm(result-s,axis=1)/np.maximum(radius,1e-12))))


def frame_consistent_fusion(semantic,neural,reliability,intervention,neural_fraction=.9):
    s=StandardScaler().fit_transform(semantic).astype(np.float64)
    n=StandardScaler().fit_transform(neural).astype(np.float64)
    if s.shape!=n.shape:raise ValueError('Paired representation shapes differ')
    rho=np.asarray(reliability).reshape(-1)
    if len(rho)!=len(n) or not np.isfinite(rho).all():raise ValueError('Invalid reliability')
    rotation,_=orthogonal_procrustes(n,s)
    aligned_neural=n@rotation
    weight=np.clip(neural_fraction*rho,0,1)[:,None]
    inner=(1-weight)*s+weight*aligned_neural
    t=float(np.clip(intervention,0,1))
    # Both terms are in the semantic frame; return to the original neural frame.
    fused=((1-t)*aligned_neural+t*inner)@rotation.T
    if not np.isfinite(fused).all():raise FloatingPointError('Nonfinite fusion')
    return fused.astype(np.float32)
