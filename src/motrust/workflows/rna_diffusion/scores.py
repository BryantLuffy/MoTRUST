"""Finite empirical CRPS and energy scores after common mean projection."""
import torch
from .model import project_mean

def _iid_scores(samples,truth):
    if samples.ndim!=3 or samples.shape[1:]!=truth.shape:raise ValueError('Expected samples, cells, features')
    s,n,f=samples.shape
    absolute=(samples-truth[None]).abs().mean(0)
    energy=torch.linalg.vector_norm(samples-truth[None],dim=2).mean(0)/f**.5
    if s>1:
        weights=(2*torch.arange(1,s+1,device=samples.device)-s-1)[:,None,None]
        absolute=absolute-(weights*torch.sort(samples,dim=0).values).sum(0)/(s*(s-1))
        distances=torch.cdist(samples.permute(1,0,2).contiguous(),samples.permute(1,0,2).contiguous(),compute_mode='donot_use_mm_for_euclid_dist')
        energy=energy-distances.sum((1,2))/(2*s*(s-1)*f**.5)
    return absolute.mean(1),energy


def empirical_scores(samples,truth):
    # Projection couples sample members: score the issued empirical distribution,
    # whose pairwise denominator is S^2, not the independent-draw S(S-1).
    a,b=_iid_scores(samples,truth);s=len(samples)
    if s>1:
        first=(samples-truth).abs().mean((0,2));norm=torch.linalg.vector_norm(samples-truth,dim=2).mean(0)/truth.shape[1]**.5
        a=a+(first-a)/s;b=b+(norm-b)/s
    return a,b


def check():
    torch.manual_seed(922);p=torch.rand(3,4,dtype=torch.float64)*5;s=project_mean(torch.randn(7,3,4,dtype=torch.float64),p);y=torch.randn(3,4,dtype=torch.float64)
    a,b=empirical_scores(s,y);ea=(s-y).abs().mean((0,2));eb=torch.linalg.vector_norm(s-y,dim=2).mean(0)/2
    for i in range(7):
        for j in range(7):ea-=(s[i]-s[j]).abs().mean(1)/98;eb-=torch.linalg.vector_norm(s[i]-s[j],dim=1)/196
    torch.testing.assert_close(a,ea);torch.testing.assert_close(b,eb)


def feature_crps(samples, truth):
    s=len(samples); weights=(2*torch.arange(1,s+1,device=samples.device)-s-1)[:,None,None]
    return (samples-truth).abs().mean(0)-(weights*samples.sort(dim=0).values).sum(0)/(s*s)
