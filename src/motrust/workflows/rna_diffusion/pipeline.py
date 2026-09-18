"""Three-source ATAC-to-RNA point and conditional diffusion workflow.

Preparation uses paired measured counts. Evaluation cells do not enter fitting
or checkpoint selection; model seeds repeat fits on the same fixed split.
"""
import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ...paths import resource_path, workdir
from ..recovery import encoder as fm, point as r, data
from ..recovery.encoder import sha
from ..recovery.data import write, load
from . import evaluation as e

OUT = workdir() / 'results/rna_diffusion'
INDEX = workdir() / 'data/cache/cache_index.json'
SOURCES = {'TEA': 'TEA_s1', 'BMMC-Multiome': 'BMMC_s1', 'Retina': 'Retina'}
SEEDS = [42, 0, 1]


def setup():
    global OUT, INDEX
    OUT = workdir() / 'results/rna_diffusion'
    INDEX = workdir() / 'data/cache/cache_index.json'
    data.configure(OUT, INDEX)
    r.configure(OUT)
    e.OUT = OUT / 'distribution'
    e.DATASETS = list(SOURCES)


def freeze():
    """Create a new installed-workflow freeze; old experimental freezes differ."""
    import motrust.models
    from ... import paths
    from ...data import integration
    from . import model, scores
    from ..recovery import metrics
    files = [Path(__file__), Path(fm.__file__), Path(r.__file__), Path(data.__file__),
             Path(e.__file__), Path(model.__file__), Path(scores.__file__), Path(metrics.__file__),
             Path(paths.__file__), Path(integration.__file__)]
    files += sorted(Path(motrust.models.__file__).parent.glob('*.py'))
    files += [resource_path('recovery', name) for name in ['training.json', 'export_metadata.R']]
    hashes = {str(p): sha(p) for p in files + [INDEX]}
    package = resource_path().parent
    identity = {str(p.relative_to(package)).replace('\\', '/'): sha(p) for p in files}
    identity['input/cache_index.json'] = sha(INDEX)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol_freeze.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['source_hashes'] != identity:
            raise RuntimeError('Source or input identity changed; choose a fresh MOTRUST_WORKDIR output directory.')
    else:
        write(path, dict(schema='motrust.rna_diffusion.v1', at_utc=datetime.now(timezone.utc).isoformat(), source_hashes=identity,
            datasets=list(SOURCES), direction='ATAC-to-RNA', training_seeds=SEEDS,
            split='RNG913; within each original library 60% train, 20% validation, 20% held-out evaluation; unique library/barcode IDs.',
            features='Up to 2000 RNA and 5000 ATAC features by training-only prevalence; stable ties; omit zero-training features.',
            point_model='VAE100 final; heads100 selected every10; masked adaptation50; ATAC-weight3 refinement20. Equal mean of seeds42,0,1; source posterior statistics from seed42.',
            distribution='DDPM and conditional stochastic head: 100 final epochs per seed, batch128, AdamW0.0002, weight decay0.0001. 32 draws; DDPM100 steps; common bounded mean projection.',
            target='log1p(10000*x / selected-feature library sum); held-out RNA excluded from fitting and checkpoint selection.',
            evaluation='Finite empirical CRPS and energy with pair denominator K^2. Source-wise ratios of three-seed method means; all outcomes retained.',
            scope='Cell-level holdout on existing integration sources; not held-out-donor or independent external-cohort validation.'))
    return hashes


def prepare():
    setup()
    data.prepare()

def points():
    setup()
    for ds in SOURCES:
        dest,manifest,split,counts=load(ds)
        if (OUT/'anchors'/ds/'completion.json').exists():continue
        ident=dict(protocol=e.sha(OUT/'protocol_freeze.json'),input=e.sha(dest/'manifest.json'),split=e.sha(dest/'split.npz'))
        for seed in SEEDS:
            final=OUT/'point/runs'/ds/f'seed_{seed}/balanced_atac'
            if (final/'manifest.json').exists():continue
            folder=OUT/'neural'/ds/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            model=fm.train_vae(counts,split,folder,seed)
            z={mod:fm.encode(model,counts,mod,np.arange(manifest['n_cells']))[0] for mod in counts}
            heads=torch.nn.ModuleDict()
            for target in counts:
                source=next(m for m in counts if m!=target)
                heads[target]=r.train_head(ds,seed,source,target,'frozen_molecular',counts,split,z,dict(**ident,seed=seed,stage='head'))
            fm.seed_all(seed);r.train_adaptation(ds,seed,'masked_joint',model,heads,counts,split,dict(**ident,seed=seed,stage='masked50'))
            state=torch.load(OUT/'point/runs'/ds/f'seed_{seed}/masked_joint/model.pt',map_location=fm.DEVICE,weights_only=True)
            model.load_state_dict(state['model']);heads.load_state_dict(state['heads']);model.eval();heads.eval()
            fm.seed_all(seed)
            r.train_adaptation(ds,seed,'balanced_atac',model,heads,counts,split,dict(**ident,seed=seed,stage='weighted20'),config=r.CFG['refinement'],atac_weight=r.CFG['refinement']['atac_weight'])
            # Store the exact stage initialization alongside the checkpoint.
            p=final/'manifest.json';record=json.loads(p.read_text());record['initialization']='New source-specific VAE100 + heads100 + masked50; this stage adds20 epochs. Selection means validation, never held-out evaluation.';write(p,record)
            print('POINT MEMBER COMPLETE',ds,seed,flush=True)
            del model,heads,z,state;gc.collect();torch.cuda.empty_cache()
        p=OUT/'anchors'/ds;p.mkdir(parents=True,exist_ok=True)
        tr,ev=split['train'],split['evaluation'];rows=np.r_[tr,ev];total=np.zeros((len(rows),counts['rna'].shape[1]),np.float64);hashes={}
        for seed in SEEDS:
            checkpoint=OUT/'point/runs'/ds/f'seed_{seed}/balanced_atac/model.pt';hashes[str(checkpoint)]=e.sha(checkpoint)
            model=fm.make_vae(counts);heads=torch.nn.ModuleDict({m:r.MolecularHead(np.zeros((2,34),np.float32),np.zeros((2,x.shape[1]),np.float32),m) for m,x in counts.items()})
            state=torch.load(checkpoint,map_location=fm.DEVICE,weights_only=True);model.load_state_dict(state['model']);heads.load_state_dict(state['heads']);model.eval();heads.eval()
            total+=r.predict(model,heads,counts,'atac','rna',rows).astype(np.float64)
            if seed==42:
                mu,lv,_=fm.encode(model,counts,'atac',rows);raw=np.column_stack([mu,np.clip(lv,-12,12)])
            del model,heads,state;gc.collect();torch.cuda.empty_cache()
        anchor=(total/3).astype(np.float32);assert np.isfinite(anchor).all() and (anchor>=0).all() and (anchor<=e.UPPER).all()
        mean=raw[:len(tr)].mean(0);std=raw[:len(tr)].std(0).clip(.0001)
        np.savez_compressed(p/'prepared.npz',anchor=anchor[:len(tr)],eval_anchor=anchor[len(tr):],residual=fm.normalized_truth(counts['rna'][tr],'rna')-anchor[:len(tr)],coordinate=(raw-mean)/std,train_indices=tr,evaluation_indices=ev,coordinate_mean=mean,coordinate_std=std)
        write(p/'completion.json',dict(status='completed',checkpoint_hashes=hashes,prepared_sha256=e.sha(p/'prepared.npz'),evaluation_role='Held-out evaluation excluded from fitting and checkpoint selection.'))
        print('ANCHOR FROZEN',ds,flush=True)


def evaluate(identity):
    setup();e.OUT=OUT/'distribution';e.OUT.mkdir(exist_ok=True);e.DATASETS=list(SOURCES)
    frozen=dict(identity)
    for ds in SOURCES:
        p=OUT/'anchors'/ds/'prepared.npz';frozen[str(p)]=e.sha(p)
        for seed in SEEDS:
            p=OUT/'point/runs'/ds/f'seed_{seed}/balanced_atac/model.pt';frozen[str(p)]=e.sha(p)
    write(OUT/'distribution/anchor_freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=frozen))
    e.check()
    for ds in SOURCES:
        with np.load(OUT/'anchors'/ds/'prepared.npz') as f:data={k:f[k] for k in f.files}
        dest,manifest,split,counts=load(ds);np.testing.assert_array_equal(data['evaluation_indices'],split['evaluation'])
        truth=fm.normalized_truth(counts['rna'][split['evaluation']],'rna')
        groups=e.strata(pd.read_csv(dest/'metadata.csv',dtype=str),split['train'],split['evaluation'])
        for method in e.METHODS:
            for seed in SEEDS:e.evaluate(ds,method,seed,data,truth,groups)
        del data,counts,truth;gc.collect()
    e.summarize(frozen)
    write(OUT/'completion.json',dict(status='completed',point_ensembles=3,point_members=9,distribution_models=18,datasets=list(SOURCES),evaluation='Fixed cell holdout on existing integration sources; not donor-held-out or external-cohort validation',result_file='distribution/comparisons.csv'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'points', 'evaluate'], default='all')
    args = parser.parse_args(argv)
    setup()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
        os.environ[key] = '1'
    torch.set_num_threads(1)
    if args.stage in ['all', 'points', 'evaluate']:
        if not torch.cuda.is_available():
            raise RuntimeError('The registered training and evaluation workflow requires CUDA. Preparation and --help do not.')
        fm.DEVICE = torch.device('cuda')
    identity = freeze()
    if args.stage in ['all', 'prepare']:
        prepare()
    if args.stage in ['all', 'points']:
        points()
    if args.stage in ['all', 'evaluate']:
        evaluate(identity)


if __name__ == '__main__':
    main()
