"""Neural encoder training used by the paired recovery workflow."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F
from ...models.multimodal_vae import ImprovedMultiModalVAE
from ...models.poe import ProductOfExperts
from ...paths import resource_path

CFG = json.loads(resource_path('recovery', 'training.json').read_text(encoding='utf-8'))
DEVICE = torch.device('cpu')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def tensor(x): return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)


def input_values(counts,rows=None):
    x=counts if rows is None else counts[rows]
    return tensor(x.toarray() if sparse.issparse(x) else x)


def batch_inputs(counts,rows):
    return {m:input_values(x,rows) if m!='atac' else (input_values(x,rows)>0).float() for m,x in counts.items()}


def forward(model,values,source=None):
    n=len(next(iter(values.values()))); zero=torch.zeros(n,1,device=DEVICE)
    return model(torch.log1p(values.get('rna',zero)),values.get('atac',zero),adt_x=torch.log1p(values['adt']) if 'adt' in values else None,
        rna_mask=torch.full((n,),float('rna' in values and source in (None,'rna')),device=DEVICE),
        atac_mask=torch.full((n,),float('atac' in values and source in (None,'atac')),device=DEVICE),
        adt_mask=torch.full((n,),float('adt' in values and source in (None,'adt')),device=DEVICE) if model.use_adt else None)


def reconstruction(model,values,result):
    n=len(next(iter(values.values()))); zero=torch.zeros(n,1,device=DEVICE)
    losses=model.compute_loss(torch.log1p(values.get('rna',zero)),values.get('atac',zero),result,adt_x=torch.log1p(values['adt']) if 'adt' in values else None,
        rna_loss_mask=torch.full((n,),float('rna' in values),device=DEVICE),atac_loss_mask=torch.full((n,),float('atac' in values),device=DEVICE),
        adt_loss_mask=torch.ones(n,device=DEVICE) if model.use_adt else None)
    loss=sum(losses[f'{m}_recon_loss']/values[m].shape[1] for m in values)
    return loss,losses['kl_loss']


def make_vae(counts,no_alpha=False):
    c=CFG['vae']
    return ImprovedMultiModalVAE(rna_dim=counts['rna'].shape[1],atac_dim=counts['atac'].shape[1] if 'atac' in counts else 1,
        adt_dim=counts['adt'].shape[1] if 'adt' in counts else None,use_adt='adt' in counts,
        latent_dim=c['dim_c']+c['dim_u'],dim_c=c['dim_c'],dim_u=c['dim_u'],hidden_dims=c['hidden_dims'],
        use_gated_poe=not no_alpha,use_shared_backbone=c['use_shared_backbone'],rna_distribution=c['rna_distribution'],adt_distribution=c['adt_distribution']).to(DEVICE)


def train_vae(counts,split,out,seed,epochs=None,no_alpha=False):
    seed_all(seed); model=make_vae(counts,no_alpha); path=out/'vae_final.pt'; epochs=epochs or CFG['vae']['epochs']
    if path.exists(): model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)); return model.eval()
    teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['vae']['learning_rate'],weight_decay=1e-5)
    c=CFG['vae']; history=[]; started=time.time()
    for epoch in range(1,epochs+1):
        model.train(); order=np.random.default_rng(seed+epoch).permutation(split['train']); total=0.; nb=0
        for j in range(0,len(order),c['batch_size']):
            rows=order[j:j+c['batch_size']]
            if len(rows)<2: continue
            values=batch_inputs(counts,rows); result=forward(model,values)
            rec,kl=reconstruction(model,values,result); loss=rec+c['kl_weight']*min(1.,epoch/10)*kl
            latent=[result[{'rna':'rna_mu_enc','atac':'atac_mu','adt':'adt_mu_enc'}[m]][:,:c['dim_c']] for m in values]
            loss+=c['bridge_alignment_weight']*sum(F.mse_loss(z,latent[0]) for z in latent[1:])
            geometry=0.
            for m,z in zip(values,latent):
                raw=torch.log1p(values[m][:64,:min(256,values[m].shape[1])]); zz=z[:64]
                d=torch.cdist(raw,raw); zd=torch.cdist(zz,zz)
                geometry+=F.mse_loss(zd/zd.detach().mean().clamp_min(1e-4),d/d.mean().clamp_min(1e-4))
            loss+=c['geometry_weight']*geometry/len(values)
            with torch.no_grad(): tr=forward(teacher,values)
            loss+=c['ema_weight']*F.mse_loss(result['z_mu'],tr['z_mu'])
            if not no_alpha:
                for m in values:
                    mu=result[{'rna':'rna_mu_enc','atac':'atac_mu','adt':'adt_mu_enc'}[m]]
                    lv=result[{'rna':'rna_logvar_enc','atac':'atac_logvar','adt':'adt_logvar_enc'}[m]]
                    loss+=c['gate_prior_weight']*(model._gate_weight(m,mu,lv).mean()-1.).square()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite VAE loss')
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            with torch.no_grad():
                for tp,mp in zip(teacher.parameters(),model.parameters()): tp.mul_(c['ema_decay']).add_(mp,alpha=1-c['ema_decay'])
                for tb,mb in zip(teacher.buffers(),model.buffers()): tb.copy_(mb)
            total+=float(loss.detach()); nb+=1
        row={'epoch':epoch,'train_loss':total/nb,'seconds':time.time()-started}
        if epoch%10==0 or epoch==epochs:
            model.eval(); val=[]
            with torch.no_grad():
                for j in range(0,len(split['selection']),256):
                    v=batch_inputs(counts,split['selection'][j:j+256]); rr=forward(model,v); rec,_=reconstruction(model,v,rr); val.append(float(rec))
            row['selection_reconstruction']=float(np.mean(val)); print(f'{out.name} VAE {epoch}/{epochs}: train={total/nb:.4f}, selection={np.mean(val):.4f}',flush=True)
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch},out/'vae_progress.pt')
        history.append(row); pd.DataFrame(history).to_csv(out/'vae_history.csv',index=False)
    torch.save(model.state_dict(),path); return model.eval()


@torch.inference_mode()
def encode(model,counts,mod,rows):
    mus=[]; lvs=[]; gates=[]
    for j in range(0,len(rows),256):
        values=input_values(counts[mod],rows[j:j+256]); values=(values>0).float() if mod=='atac' else torch.log1p(values)
        mu,lv=getattr(model,mod+'_encoder')(values)
        gate=model._gate_weight(mod,mu,lv) if model.use_gated_poe else torch.ones(len(mu),1,device=DEVICE)
        mu,lv=ProductOfExperts.poe_with_prior([mu],[lv],weights=[gate])
        mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy()); gates.append(gate.cpu().numpy().reshape(-1))
    return np.concatenate(mus),np.concatenate(lvs),np.concatenate(gates)


def normalized_truth(x,mod):
    raw=x.toarray() if sparse.issparse(x) else x
    return (raw>0).astype(np.float32) if mod=='atac' else np.log1p(raw*1e4/np.maximum(raw.sum(1,keepdims=True),1e-8)).astype(np.float32)
