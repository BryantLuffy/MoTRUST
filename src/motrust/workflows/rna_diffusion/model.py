"""Conditional molecular residual DDPM and common bounded mean projection."""
import math
import torch
from torch import nn

class MolecularDiffusion(nn.Module):
    def __init__(self,features,condition):
        super().__init__();self.net=nn.Sequential(nn.Linear(features+condition+32,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,features))
        t=torch.arange(101)/100;v=torch.cos((t+.008)/1.008*math.pi/2)**2;v=v/v[0];beta=(1-v[1:]/v[:-1]).clamp(.0001,.999)
        self.register_buffer('beta',beta);self.register_buffer('alpha',1-beta);self.register_buffer('abar',torch.cumprod(1-beta,0))
    def forward(self,x,condition,t):
        f=torch.exp(-math.log(10000)*torch.arange(16,device=x.device)/15);a=t[:,None]*f[None]
        return self.net(torch.cat([x,condition,torch.sin(a),torch.cos(a)],1))
    @torch.inference_mode()
    def sample(self,condition,anchor,scale,seed):
        n,f=anchor.shape;cond=condition.repeat(32,1);p=anchor.repeat(32,1);gen=torch.Generator(device=p.device).manual_seed(seed)
        x=torch.randn(p.shape,device=p.device,generator=gen);upper=math.log1p(10000)
        for t in range(99,-1,-1):
            ab=self.abar[t];prev=self.abar[t-1] if t else torch.ones((),device=p.device)
            eps=self(x,cond,torch.full((len(x),),t,device=p.device));x0=(x-(1-ab).sqrt()*eps)/ab.sqrt()
            x0=torch.maximum(torch.minimum(x0,(upper-p)/scale),-p/scale)
            mu=self.beta[t]*prev.sqrt()/(1-ab)*x0+(1-prev)*self.alpha[t].sqrt()/(1-ab)*x
            x=mu+(self.beta[t]*(1-prev)/(1-ab)).sqrt()*torch.randn(x.shape,device=x.device,generator=gen) if t else mu
        return (p+x*scale).reshape(32,n,f)

@torch.inference_mode()
def project_mean(samples,anchor):
    upper=math.log1p(10000);lo=-samples.max(0).values-upper;hi=upper-samples.min(0).values+upper
    for _ in range(32):
        mid=(lo+hi)/2;mean=(samples+mid).clamp(0,upper).mean(0);mask=mean<anchor;lo=torch.where(mask,mid,lo);hi=torch.where(mask,hi,mid)
    result=(samples+(lo+hi)/2).clamp(0,upper)
    if not torch.allclose(result.mean(0),anchor,rtol=0,atol=2e-5):raise AssertionError('Mean projection failed')
    return result
