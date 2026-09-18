"""Paired-library preparation with training-only prevalence selection."""
import gc
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from ...paths import resource_path, workdir
from ...data.integration import resolve_reference
from .encoder import sha

OUT = workdir() / 'results/rna_diffusion'
INDEX = workdir() / 'data/cache/cache_index.json'
SOURCES = {'TEA': 'TEA_s1', 'BMMC-Multiome': 'BMMC_s1', 'Retina': 'Retina'}


def configure(output=None, index=None):
    global OUT, INDEX
    OUT = Path(output) if output is not None else workdir() / 'results/rna_diffusion'
    INDEX = Path(index) if index is not None else workdir() / 'data/cache/cache_index.json'

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')


def names(p):
    p=Path(p)
    return pd.read_csv(p).iloc[:,0].astype(str).to_numpy(dtype=str) if p.suffix=='.csv' else np.asarray(p.read_text(encoding='utf-8').splitlines())


def prepare():
    index_data = json.loads(INDEX.read_text(encoding='utf-8'))
    cache = index_data['scenarios']
    path_base = index_data.get('path_base', 'workdir')
    for batches in cache.values():
        for item in batches.values():
            for modality in item['modalities'].values():
                for key in ['matrix', 'features', 'barcodes']:
                    modality[key] = str(resolve_reference(modality[key], root=workdir(), index_path=INDEX, path_base=path_base))
            item['metadata'] = str(resolve_reference(item['metadata'], root=workdir(), index_path=INDEX, path_base=path_base))
    if not (OUT/'metadata_export_complete.json').exists():
        subprocess.run([os.environ.get('RSCRIPT', 'Rscript'),str(resource_path('recovery', 'export_metadata.R')),str(INDEX),str(OUT/'metadata_raw'),str(workdir())],check=True)
        write(OUT/'metadata_export_complete.json',dict(status='completed'))
    for ds,scenario in SOURCES.items():
        dest=OUT/'inputs'/ds;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'manifest.json').exists():continue
        entries=[(key,item) for key,item in cache[scenario].items() if {'rna','atac'}<=set(item['modalities'])]
        rng=np.random.default_rng(913);frames=[];splits={'train':[],'selection':[],'evaluation':[]};blocks=[];start=0
        hashes={};feature_names={};prevalence={}
        for key,item in entries:
            ids=names(item['modalities']['rna']['barcodes'])
            np.testing.assert_array_equal(ids,names(item['modalities']['atac']['barcodes']));assert len(np.unique(ids))==len(ids)
            md=pd.read_csv(OUT/'metadata_raw'/scenario/f'{key}.csv',dtype=str).set_index('original_barcode').loc[ids]
            label=next((k for k in ['cell_type__custom','celltype.l2','cell_type','celltype','CellType','cell_type_annot','annotation'] if k in md),None)
            frames.append(pd.DataFrame(dict(cell_id=[item['source_subdir']+'::'+x for x in ids],original_barcode=ids,batch=key,source_library=item['source_subdir'],cell_type=md[label].fillna('unannotated').to_numpy() if label else 'unannotated')))
            order=rng.permutation(len(ids));tr,va,ev=np.split(order,[int(.6*len(ids)),int(.8*len(ids))])
            for name,ix in zip(splits,[tr,va,ev]):splits[name].extend((ix+start).tolist())
            block=dest/'blocks'/key;block.mkdir(parents=True,exist_ok=True)
            for mod in ['rna','atac']:
                ref=item['modalities'][mod];fn=names(ref['features'])
                assert len(np.unique(fn))==len(fn)
                if mod in feature_names:np.testing.assert_array_equal(fn,feature_names[mod])
                else:feature_names[mod]=fn;prevalence[mod]=np.zeros(len(fn),dtype=np.int64)
                for field in ['matrix','features','barcodes']:hashes[ref[field]]=sha(ref[field])
                p=block/f'{mod}.npz'
                if p.exists():x=sparse.load_npz(p)
                else:
                    x=sparse.csr_matrix(mmread(ref['matrix']).T,dtype=np.float32);x.sum_duplicates();x.eliminate_zeros()
                    assert x.shape==(len(ids),len(fn)) and np.isfinite(x.data).all() and (x.data>=0).all()
                    assert np.allclose(x.data,np.rint(x.data),atol=1e-5,rtol=0)
                    sparse.save_npz(p,x)
                prevalence[mod]+=np.asarray((x[tr]>0).sum(0)).ravel()
                del x;gc.collect()
            hashes[item['metadata']]=sha(item['metadata'])
            blocks.append((key,len(ids)));start+=len(ids)
            print('INPUT BLOCK',ds,key,len(ids),flush=True)
        metadata=pd.concat(frames,ignore_index=True);assert metadata.cell_id.is_unique
        split={k:np.sort(v).astype(np.int64) for k,v in splits.items()}
        assert np.array_equal(np.sort(np.concatenate(list(split.values()))),np.arange(len(metadata)))
        metadata.to_csv(dest/'metadata.csv',index=False)
        np.savez_compressed(dest/'split.npz',**split,cell_ids=metadata.cell_id.to_numpy(dtype=str))
        features={}
        for mod,limit in [('rna',2000),('atac',5000)]:
            chosen=np.argsort(-prevalence[mod],kind='stable')[:limit];chosen=chosen[prevalence[mod][chosen]>0]
            pieces=[]
            for key,n in blocks:pieces.append(sparse.load_npz(dest/'blocks'/key/f'{mod}.npz')[:,chosen].tocsr())
            x=sparse.vstack(pieces,format='csr');sparse.save_npz(dest/f'{mod}_counts.npz',x)
            np.save(dest/f'{mod}_feature_names.npy',feature_names[mod][chosen]);np.save(dest/f'{mod}_feature_indices.npy',chosen)
            features[mod]=dict(n_features=len(chosen),inherited_vocabulary=len(feature_names[mod]),selected_training_only=True)
            del x,pieces;gc.collect()
        write(dest/'manifest.json',dict(dataset=ds,n_cells=len(metadata),features=features,source_hashes=hashes,split_counts={k:len(v) for k,v in split.items()},
              paired=True,unique_original_cells=True,split_kind='within-library cell holdout',split_sha256=sha(dest/'split.npz')))
        print('PREPARED',ds,len(metadata),{k:len(v) for k,v in split.items()},flush=True)


def load(ds):
    dest=OUT/'inputs'/ds;manifest=json.loads((dest/'manifest.json').read_text())
    return dest,manifest,dict(np.load(dest/'split.npz')), {m:sparse.load_npz(dest/f'{m}_counts.npz').tocsr() for m in ['rna','atac']}
