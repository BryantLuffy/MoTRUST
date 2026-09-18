"""Matched conditional-distribution fitting and held-out evaluation."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ...paths import workdir
from ..recovery import encoder as fm
from ..recovery.encoder import sha
from .model import MolecularDiffusion, project_mean
from .scores import empirical_scores, feature_crps, check

OUT = workdir() / 'results/rna_diffusion/distribution'
DATASETS = ['TEA', 'BMMC-Multiome', 'Retina']
SEEDS = [42, 0, 1]
UPPER = np.log1p(10000)
METHODS = ['residual_diffusion', 'conditional_head']

def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def tensor(x): return fm.tensor(x)


def fit_model(ds, method, seed, folder, anchor, residual, x):
    fm.seed_all(seed)
    features = anchor.shape[1]
    net = (MolecularDiffusion(features, x.shape[1]) if method == METHODS[0] else
           nn.Sequential(nn.Linear(x.shape[1],256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,features*3))).to(fm.DEVICE)
    checkpoint = folder/'model.pt'
    reused = checkpoint.exists()
    scale = np.sqrt(np.mean(residual.astype(np.float64)**2,axis=0)).clip(.001).astype(np.float32)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=fm.DEVICE, weights_only=False)
        net.load_state_dict(state['model'] if method == METHODS[0] else state)
        if method == METHODS[0]: np.testing.assert_allclose(state['scale'], scale)
        elapsed = float(pd.read_csv(checkpoint.parent/'history.csv').iloc[-1].seconds)
    else:
        assert not (folder/'history.partial.csv').exists(), 'Preserve interrupted fit before restarting'
        optimizer = torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
        xx = tensor(x)
        yy = tensor(residual/scale if method == METHODS[0] else (anchor+residual).clip(0,UPPER))
        history = []
        torch.cuda.synchronize(); start = time.perf_counter()
        for epoch in range(1,101):
            order = np.random.default_rng(52000+seed+epoch).permutation(len(x)); losses=[]
            for j in range(0,len(x),128):
                ix = order[j:j+128]
                if method == METHODS[0]:
                    tt=torch.randint(100,(len(ix),),device=xx.device); noise=torch.randn_like(yy[ix]); ab=net.abar[tt,None]
                    loss=(net(ab.sqrt()*yy[ix]+(1-ab).sqrt()*noise,xx[ix],tt)-noise).square().mean()
                else:
                    logit,mu,ls=net(xx[ix]).chunk(3,1);mu=F.softplus(mu);ls=ls.clamp(-4,2);positive=yy[ix]>0
                    loss=(F.binary_cross_entropy_with_logits(logit,positive.float(),reduction='none')+positive*(ls+.5*((yy[ix]-mu)/ls.exp())**2)).mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
            pd.DataFrame(history).to_csv(folder/'history.partial.csv',index=False)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        state=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=100,scale=scale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()) if method == METHODS[0] else net.state_dict()
        torch.save(state,checkpoint);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        del optimizer,xx,yy,state
    net.eval()
    return net,scale,checkpoint,elapsed,reused


def evaluate(ds, method, seed, data, truth, groups):
    folder=OUT/ds/method/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():
        done=json.loads((folder/'completion.json').read_text());assert sha(done['checkpoint'])==done['checkpoint_sha256'];return
    anchor,ep,residual,coord,tr,ev=[data[k] for k in ['anchor','eval_anchor','residual','coordinate','train_indices','evaluation_indices']]
    x=np.column_stack([coord[:len(tr)],anchor/UPPER]).astype(np.float32)
    ex=np.column_stack([coord[len(tr):],ep/UPPER]).astype(np.float32)
    net,scale,checkpoint,training_seconds,reused=fit_model(ds,method,seed,folder,anchor,residual,x)
    train_means=(anchor+residual).clip(0,UPPER).mean(0)
    cuts=np.quantile(train_means,[.25,.5,.75]); bins=np.searchsorted(cuts,train_means,side='right')
    buffers=[]; cells=[]; maxdiff=0.; generation_seconds=0.
    strata_sums={key:np.zeros(4,np.float64) for key in ['all','zero','nonzero','expression_Q1','expression_Q2','expression_Q3','expression_Q4']}
    feature_sums=np.zeros((truth.shape[1],3),np.float64)
    with torch.inference_mode():
        for j in range(0,len(ev),64):
            p=tensor(ep[j:j+64]);condition=tensor(ex[j:j+64]);y=tensor(truth[j:j+64])
            torch.cuda.synchronize();start=time.perf_counter()
            if method==METHODS[0]: draws=net.sample(condition,p,tensor(scale),52042+j)
            else:
                logit,mu,ls=net(condition).chunk(3,1);mu=F.softplus(mu);sigma=ls.clamp(-4,2).exp()
                gen=torch.Generator(device=p.device).manual_seed(52042+j);shape=(32,*p.shape)
                present=torch.rand(shape,device=p.device,generator=gen)<logit.sigmoid()
                draws=(mu+sigma*torch.randn(shape,device=p.device,generator=gen)).clamp(0,UPPER)*present
            draws=project_mean(draws,p)
            torch.cuda.synchronize();generation_seconds+=time.perf_counter()-start
            maxdiff=max(maxdiff,float((draws.mean(0)-p).abs().max()))
            crps,energy=empirical_scores(draws,y);fc=feature_crps(draws,y)
            torch.testing.assert_close(fc.mean(1),crps,rtol=1e-5,atol=2e-6)
            lo,hi=torch.quantile(draws,torch.tensor([.05,.95],device=p.device),dim=0);hit=(y>=lo)&(y<=hi);width=hi-lo
            buffers.append(torch.stack([crps,energy],1).cpu().numpy())
            cells.append(torch.stack([hit.float().mean(1),width.mean(1)],1).cpu().numpy())
            masks={'all':torch.ones_like(y,dtype=torch.bool),'zero':y==0,'nonzero':y>0}
            masks.update({f'expression_Q{k+1}':torch.as_tensor(bins==k,device=y.device)[None].expand_as(y) for k in range(4)})
            for key,mask in masks.items():
                strata_sums[key]+=np.array([int(mask.sum()),float(fc[mask].sum()),float(hit[mask].sum()),float(width[mask].sum())])
            feature_sums+=torch.stack([fc.sum(0),hit.sum(0),width.sum(0)],1).cpu().numpy()
    a=np.concatenate(buffers);v=np.concatenate(cells)
    np.savez_compressed(folder/'scores.npz',scores=a,intervals=v,evaluation_indices=ev)
    ident=dict(dataset=ds,method=method,training_seed=seed)
    rows=[dict(**ident,stratum=key,n_entries=int(z[0]),crps=z[1]/z[0] if z[0] else None,coverage=z[2]/z[0] if z[0] else None,width=z[3]/z[0] if z[0] else None) for key,z in strata_sums.items()]
    pd.DataFrame(rows).to_csv(folder/'strata.csv',index=False)
    pd.DataFrame([dict(**ident,group=name,n=int(mask.sum()),crps=float(a[mask,0].mean()),energy=float(a[mask,1].mean()),coverage=float(v[mask,0].mean()),width=float(v[mask,1].mean())) for name,mask in groups.items() if mask.any()]).to_csv(folder/'groups.csv',index=False)
    pd.DataFrame(dict(feature_index=np.arange(len(bins)),training_mean=train_means,expression_quartile=bins+1,crps=feature_sums[:,0]/len(ev),coverage=feature_sums[:,1]/len(ev),width=feature_sums[:,2]/len(ev))).to_csv(folder/'features.csv',index=False)
    row=dict(**ident,n_train=len(tr),n_evaluation=len(ev),features=truth.shape[1],crps=float(a[:,0].mean()),energy=float(a[:,1].mean()),coverage=float(v[:,0].mean()),width=float(v[:,1].mean()),training_seconds=training_seconds,generation_seconds=generation_seconds,generation_ms_per_cell=1000*generation_seconds/len(ev),reused_checkpoint=reused)
    pd.DataFrame([row]).to_csv(folder/'metrics.csv',index=False)
    write(folder/'completion.json',dict(status='completed',**ident,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),maximum_point_mean_difference=maxdiff))
    print('COMPLETE',ds,method,seed,'reused=',reused,'CRPS=',row['crps'],flush=True)
    del net
    torch.cuda.empty_cache()


def summarize(identity):
    roots=[OUT/d/m/f'seed_{s}' for d in DATASETS for m in METHODS for s in SEEDS]
    for name in ['metrics','groups','strata']:
        pd.concat([pd.read_csv(p/f'{name}.csv') for p in roots],ignore_index=True).to_csv(OUT/f'all_{name}.csv',index=False)
    df=pd.read_csv(OUT/'all_metrics.csv');means=df.groupby(['dataset','method'],sort=False)[['crps','energy','coverage','width','generation_ms_per_cell']].mean()
    means.to_csv(OUT/'method_means.csv');rows=[];pairs=[]
    for ds in DATASETS:
        a=means.loc[(ds,METHODS[0])];b=means.loc[(ds,METHODS[1])]
        rows.append(dict(dataset=ds,relative_crps=a.crps/b.crps-1,relative_energy=a.energy/b.energy-1,coverage_diff=a.coverage-b.coverage,width_diff=a.width-b.width,generation_time_ratio=a.generation_ms_per_cell/b.generation_ms_per_cell))
        for aa in df[(df.dataset==ds)&(df.method==METHODS[0])].itertuples():
            for bb in df[(df.dataset==ds)&(df.method==METHODS[1])].itertuples():
                pairs.append(dict(dataset=ds,diffusion_seed=aa.training_seed,control_seed=bb.training_seed,matched_seed=aa.training_seed==bb.training_seed,relative_crps=aa.crps/bb.crps-1,relative_energy=aa.energy/bb.energy-1))
    pd.DataFrame(rows).to_csv(OUT/'comparisons.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'all_seed_comparisons.csv',index=False)
    g=pd.read_csv(OUT/'all_groups.csv');g=g[g.group.str.startswith('type:')&(g.n>=20)]
    g=g.groupby(['dataset','group','method']).agg(crps=('crps','mean'),n=('n','first')).reset_index()
    a=g[g.method==METHODS[0]].merge(g[g.method==METHODS[1]],on=['dataset','group'],suffixes=('_diffusion','_control'))
    a['relative_crps']=a.crps_diffusion/a.crps_control-1;a['worsens_over_5pct']=a.relative_crps>.05
    a.to_csv(OUT/'cell_type_comparisons.csv',index=False)
    for p,h in identity.items(): assert sha(p)==h, p
    write(OUT/'completion.json',dict(status='completed',models_evaluated=len(df),new_models=int((~df.reused_checkpoint).sum()),reused_models=int(df.reused_checkpoint.sum()),source_hashes_verified=len(identity),direction='ATAC-to-RNA',datasets=DATASETS,training_seeds=SEEDS,independent_external_confirmation=False,equal_dataset_relative_crps=float(np.mean([r['relative_crps'] for r in rows])),equal_dataset_relative_energy=float(np.mean([r['relative_energy'] for r in rows]))))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def strata(meta,tr,test):
    output={'all':np.ones(len(test),dtype=bool)}
    label=next((k for k in ['cell_type','celltype','CellType','cell_type_annot','annotation'] if k in meta),None)
    if label:
        labels=meta[label].astype(str);freq=labels.iloc[tr].value_counts();rare=set(freq[freq<=freq.quantile(.25)].index)
        output['rare_train_bottom_quartile']=labels.iloc[test].isin(rare).to_numpy()
        for v in sorted(labels.iloc[test].unique()):output['type:'+v]=(labels.iloc[test].to_numpy()==v)
    for col in ['donor','donor_id','batch']:
        if col in meta:
            for v in meta.iloc[test][col].astype(str).unique():output[col+':'+v]=(meta.iloc[test][col].astype(str).to_numpy()==v)
    return output
