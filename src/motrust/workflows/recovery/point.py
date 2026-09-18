"""Point prediction heads, masked adaptation and ATAC-weighted refinement."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ...models.poe import ProductOfExperts
from ...paths import resource_path, workdir
from . import encoder as fm
from .metrics import metrics

CFG = json.loads(resource_path('recovery', 'training.json').read_text(encoding='utf-8'))
OUT = workdir() / 'results/rna_diffusion/point'
INPUTS = workdir() / 'results/rna_diffusion/inputs'


def configure(output):
    """Set user-owned artifact directories without changing model parameters."""
    global OUT, INPUTS
    OUT = Path(output) / 'point'
    INPUTS = Path(output) / 'inputs'

def molecular_loss(raw, y, target):
    """BCE for ATAC, MSE for log-normalized RNA or ADT."""
    return F.binary_cross_entropy_with_logits(raw, y) if target == 'atac' else F.mse_loss(raw, y)


@torch.inference_mode()
def predict(model, heads, counts, source, target, rows):
    """Predict using the observed source only; target counts are never read."""
    raw = counts[source][rows].copy()
    return np.concatenate([heads[target].prediction(latent(model, {source: fm.input_values(raw[j:j+256])}, [source])).cpu().numpy() for j in range(0, len(rows), 256)])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def completed(path, ident):
    if not path.exists():
        return False
    previous = json.loads(path.read_text(encoding='utf-8'))
    if previous['identity'] != ident:
        raise RuntimeError(f'Identity changed; archive and explicitly register a new run: {path}')
    for f, h in previous.get('outputs', {}).items():
        if not (path.parent / f).exists() or fm.sha(path.parent / f) != h:
            raise RuntimeError(f'Output changed: {path.parent / f}')
    return True


def finish(path, ident, **extra):
    outputs = {str(p.relative_to(path.parent)): fm.sha(p) for p in path.parent.rglob('*')
               if p.is_file() and p != path and p.suffix in ('.pt', '.csv', '.npz')}
    write_json(path, dict(status='completed', identity=ident, outputs=outputs, **extra))


class MolecularHead(nn.Module):
    def __init__(self, z, y, target):
        super().__init__()
        self.target = target
        self.register_buffer('center', fm.tensor(z.mean(0)))
        self.register_buffer('scale', fm.tensor(z.std(0).clip(1e-4)))
        self.net = nn.Sequential(nn.Linear(z.shape[1], 256), nn.LayerNorm(256), nn.Mish(),
                                 nn.Linear(256, 256), nn.Mish(), nn.Linear(256, y.shape[1])).to(fm.DEVICE)
        mean = fm.tensor(y.mean(0))
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.logit(mean.clamp(1e-4, 1 - 1e-4)) if target == 'atac' else mean)

    def forward(self, z):
        return self.net((z - self.center) / self.scale)

    def prediction(self, z):
        raw = self(z)
        return raw.sigmoid() if self.target == 'atac' else raw.clamp_min(0)


def latent(model, values, observed):
    """Only observed modality tensors are read; unobserved values cannot affect z."""
    mus, lvs, weights = [], [], []
    for m in observed:
        x = (values[m] > 0).float() if m == 'atac' else torch.log1p(values[m])
        mu, lv = getattr(model, m + '_encoder')(x)
        mus.append(mu); lvs.append(lv); weights.append(model._gate_weight(m, mu, lv))
    return ProductOfExperts.poe_with_prior(mus, lvs, weights=weights)[0]


@torch.inference_mode()
def head_prediction(head, z):
    return np.concatenate([head.prediction(fm.tensor(z[j:j + 256])).cpu().numpy() for j in range(0, len(z), 256)])


def score_and_save(out, dataset, seed, method, source, target, prediction, counts, split, diagnostic=False):
    rows = split['selection']
    truth = fm.normalized_truth(counts[target][rows], target)
    assert np.isfinite(prediction).all() and prediction.shape == truth.shape
    values, _ = metrics(truth, prediction, target)
    np.savez_compressed(out / f'{method}_{source}_to_{target}.npz', prediction=prediction,
                        selection_indices=rows, feature_names=np.load(INPUTS / dataset / f'{target}_feature_names.npy'))
    return dict(dataset=dataset, seed=seed, method=method, direction=source + '_to_' + target,
                target=target, split='selection', diagnostic=diagnostic, n_cells=len(rows), **values)


def train_head(dataset, seed, source, target, variant, counts, split, z, ident):
    out = OUT / 'runs' / dataset / f'seed_{seed}' / variant / f'{source}_to_{target}'
    out.mkdir(parents=True, exist_ok=True)
    tr, va = split['train'], split['selection']
    ytr = fm.normalized_truth(counts[target][tr], target)
    yva = fm.normalized_truth(counts[target][va], target)
    fm.seed_all(seed)
    head = MolecularHead(z[source][tr], ytr, target)
    if completed(out / 'manifest.json', ident):
        head.load_state_dict(torch.load(out / 'head.pt', map_location=fm.DEVICE, weights_only=True))
        return head.eval()
    c = CFG['head']; optimizer = torch.optim.AdamW(head.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    tz, ty = fm.tensor(z[source][tr]), fm.tensor(ytr)
    vz, vy = fm.tensor(z[source][va]), fm.tensor(yva)
    best, best_epoch, history = np.inf, None, []
    start = time.time()
    for epoch in range(1, c['epochs'] + 1):
        head.train(); order = np.random.default_rng(seed + epoch).permutation(len(tr)); total = 0.
        for j in range(0, len(order), c['batch_size']):
            ix = order[j:j + c['batch_size']]
            loss = molecular_loss(head(tz[ix]), ty[ix], target)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 2); optimizer.step()
            total += float(loss.detach()) * len(ix)
        record = dict(epoch=epoch, loss=total / len(tr), seconds=time.time() - start)
        if epoch % c['checkpoint_every'] == 0:
            head.eval()
            with torch.no_grad():
                value = sum(float(molecular_loss(head(vz[j:j + 256]), vy[j:j + 256], target)) * len(vz[j:j + 256]) for j in range(0, len(va), 256)) / len(va)
            record['selection_loss'] = value
            if value < best:
                best, best_epoch = value, epoch
                torch.save(head.state_dict(), out / 'head.pt')
        history.append(record)
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    head.load_state_dict(torch.load(out / 'head.pt', map_location=fm.DEVICE, weights_only=True)); head.eval()
    row = score_and_save(out, dataset, seed, variant, source, target, head_prediction(head, z[source][va]), counts, split)
    pd.DataFrame([row]).to_csv(out / 'metrics.csv', index=False)
    finish(out / 'manifest.json', ident, selected_epoch=best_epoch, selection_loss=best, seconds=time.time() - start)
    print(f'HEAD {dataset} {seed} {source}->{target} {variant} epoch{best_epoch}: r={row["feature_pearson"]:.4f} RMSE={row["rmse"]:.4f}', flush=True)
    return head


def train_adaptation(dataset, seed, variant, base, heads, counts, split, ident, *, config=None, atac_weight=1.):
    out = OUT / 'runs' / dataset / f'seed_{seed}' / variant
    out.mkdir(parents=True, exist_ok=True)
    if completed(out / 'manifest.json', ident):
        return
    model = copy.deepcopy(base); hs = nn.ModuleDict({k: copy.deepcopy(v) for k, v in heads.items()})
    for p in model.parameters(): p.requires_grad_(False)
    for m in counts:
        for p in getattr(model, m + '_encoder').parameters(): p.requires_grad_(True)
        for p in model.modality_gates[m].parameters(): p.requires_grad_(True)
    c = CFG['adaptation'] if config is None else config
    loss_fn = lambda raw, y, target: molecular_loss(raw, y, target) * (atac_weight if target == 'atac' else 1)
    optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': c['encoder_learning_rate']},
                                  {'params': hs.parameters(), 'lr': c['head_learning_rate']}], weight_decay=1e-4)
    tr, va = split['train'], split['selection']; mods = list(counts)
    raw = {m: fm.input_values(counts[m], tr) for m in mods}
    y = {m: fm.tensor(fm.normalized_truth(counts[m][tr], m)) for m in mods}
    vr = {m: fm.input_values(counts[m], va) for m in mods}
    vy = {m: fm.tensor(fm.normalized_truth(counts[m][va], m)) for m in mods}
    best, best_epoch, history = np.inf, None, []; start = time.time()
    for epoch in range(1, c['epochs'] + 1):
        model.train(); hs.train(); order = np.random.default_rng(seed + epoch).permutation(len(tr)); total = 0.
        for batch, j in enumerate(range(0, len(order), c['batch_size'])):
            ix = order[j:j + c['batch_size']]; v = {m: raw[m][ix] for m in mods}
            choice = (batch + epoch - 1) % 3
            observed = mods if choice == 2 else [mods[choice]]
            z = latent(model, v, observed)
            loss = sum(loss_fn(hs[m](z), y[m][ix], m) for m in mods) / len(mods)
            with torch.no_grad(): old = latent(base, v, observed)
            loss = loss + c['latent_anchor_weight'] * F.mse_loss(z, old)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 2); optimizer.step()
            total += float(loss.detach()) * len(ix)
        record = dict(epoch=epoch, loss=total / len(tr), seconds=time.time() - start)
        if epoch % c['checkpoint_every'] == 0:
            model.eval(); hs.eval(); values = []
            with torch.no_grad():
                for source in mods:
                    target = next(m for m in mods if m != source); total_val = 0.
                    for j in range(0, len(va), 256):
                        z = latent(model, {source: vr[source][j:j + 256]}, [source])
                        total_val += float(loss_fn(hs[target](z), vy[target][j:j + 256], target)) * len(z)
                    values.append(total_val / len(va))
            value = float(np.mean(values)); record['selection_loss'] = value
            if value < best:
                best, best_epoch = value, epoch
                torch.save({'model': model.state_dict(), 'heads': hs.state_dict()}, out / 'model.pt')
            print(f'ADAPT {dataset} {seed} {variant} {epoch}/{c["epochs"]}: {value:.5f}', flush=True)
        history.append(record)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    state = torch.load(out / 'model.pt', map_location=fm.DEVICE, weights_only=True)
    model.load_state_dict(state['model']); hs.load_state_dict(state['heads']); model.eval(); hs.eval(); rows = []
    with torch.no_grad():
        for source in mods:
            target = next(m for m in mods if m != source)
            prediction = np.concatenate([hs[target].prediction(latent(model, {source: vr[source][j:j + 256]}, [source])).cpu().numpy() for j in range(0, len(va), 256)])
            rows.append(score_and_save(out, dataset, seed, variant, source, target, prediction, counts, split))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    finish(out / 'manifest.json', ident, selected_epoch=best_epoch, selection_loss=best, seconds=time.time() - start,
           initialization='Encoder and selected point heads followed by masked adaptation; see training.json for stage budgets')
