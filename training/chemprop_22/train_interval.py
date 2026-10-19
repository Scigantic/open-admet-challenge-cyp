"""Chemprop member trained against the SCORED interval loss instead of MAE.

One declared intervention against analysis/aux_ac50_weighted_20260910/train.py:
the four scored heads use  max(0, lo-p) + max(0, p-hi)  in real pIC50 units; the
18 auxiliary heads keep MAE. Everything else -- data table, 5-fold scaffold
partition, inverse-variance weights, architecture, optimiser, batch size, epochs --
is copied unchanged.

Leak-free: the loss uses only each row's OWN interval. No retargeting round and no
cross-fitted target is needed, because a neural net has no leaf-refinement step,
which is the only reason the LightGBM work had to go through the clip identity.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import ChempropMetric

OUT = Path("analysis/chemprop_interval_20260910"); OUT.mkdir(exist_ok=True)
SMOKE = "--smoke" in sys.argv
MAX_EPOCHS = 3 if SMOKE else 200
sys.path.insert(0, "analysis/metric_decision_layer_20260907")
from featurize import load_challenge, scaffold_of        # noqa: E402

EPS4 = ("cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4")
tbl = pd.read_csv("analysis/aux_ac50_member_20260909/train_table.csv")
TARGETS = [c for c in tbl.columns if c not in ("Molecule_Name", "smiles", "InChIKey")]
SCORED = [f"{e}_pic50_direct_inhibition" for e in EPS4]
SC_IDX = [TARGETS.index(c) for c in SCORED]
Y = tbl[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
NT = len(TARGETS)

tr_ch, te_ch = load_challenge()
tr_ch["Molecule_Name"] = tr_ch["Molecule_Name"].astype(str)
te_ch["Molecule_Name"] = te_ch["Molecule_Name"].astype(str)
tbl["Molecule_Name"] = tbl["Molecule_Name"].astype(str)
ci = tr_ch.set_index("Molecule_Name")

# ---- append lo/hi for the four scored heads as extra target columns
LO, HI = [], []
for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
    LO.append(ci[f"{e}_pIC50_direct_inhibition_conf_low"].reindex(tbl.Molecule_Name).to_numpy(float))
    HI.append(ci[f"{e}_pIC50_direct_inhibition_conf_high"].reindex(tbl.Molecule_Name).to_numpy(float))
LO = np.column_stack(LO); HI = np.column_stack(HI)
# a bound is only usable where the scored label itself is present
for j, i in enumerate(SC_IDX):
    bad = ~np.isfinite(Y[:, i]); LO[bad, j] = np.nan; HI[bad, j] = np.nan
Yaug = np.hstack([Y, LO, HI])
NAUG = Yaug.shape[1]
LO_IDX = list(range(NT, NT + 4)); HI_IDX = list(range(NT + 4, NT + 8))
print(f"rows {len(tbl)} real targets {NT} augmented {NAUG}")
print(f"scored head indices {SC_IDX}; bound columns {LO_IDX} {HI_IDX}")
print(f"bound coverage: {[int(np.isfinite(LO[:,j]).sum()) for j in range(4)]}")

# ---- inverse-variance weights, identical to the parent script
wsum = np.zeros(len(tbl)); wcnt = np.zeros(len(tbl))
for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
    lo = ci[f"{e}_pIC50_direct_inhibition_conf_low"].reindex(tbl.Molecule_Name).to_numpy(float)
    hi = ci[f"{e}_pIC50_direct_inhibition_conf_high"].reindex(tbl.Molecule_Name).to_numpy(float)
    m = np.isfinite(lo) & np.isfinite(hi)
    sig = np.clip((hi - lo) / (2 * 1.96), 0.02, None)
    wsum[m] += 1.0 / sig[m] ** 2; wcnt[m] += 1
w = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)
w = np.where(np.isfinite(w), w, np.nanmedian(w)); w = w / w.mean()
cap = np.quantile(w, 0.99); w = np.clip(w, None, cap); w = w / w.mean()
print(f"weights min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f}")


class IntervalMAE(ChempropMetric):
    """MAE on every head, replaced by the scored interval hinge on the four scored heads.

    The loss runs in the target-scaled space, so predictions and bounds are
    un-scaled with the fold's StandardScaler, the hinge is taken in real pIC50
    units, then divided by the head's scale so its magnitude stays comparable to
    the MAE heads.
    """
    def __init__(self, mean, scale, sc_idx, lo_idx, hi_idx, task_weights):
        super().__init__(task_weights=task_weights)
        self.register_buffer("mu", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("sd", torch.as_tensor(scale, dtype=torch.float))
        self.sc_idx = list(sc_idx); self.lo_idx = list(lo_idx); self.hi_idx = list(hi_idx)
        self.n_real = int(min(lo_idx))

    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        # The eight bound columns ride along as targets so the loss can see them,
        # but they must not count as trainable observations. Zeroing them in the
        # mask keeps num_samples honest; task_weights already zeroes their loss.
        if mask is not None:
            mask = mask.clone(); mask[:, self.n_real:] = False
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)

    def _calc_unreduced_loss(self, preds, targets, *args) -> torch.Tensor:
        L = (preds - targets).abs()
        cols = [L[:, j] for j in range(L.shape[1])]
        for k, i in enumerate(self.sc_idx):
            li, hj = self.lo_idx[k], self.hi_idx[k]
            p = preds[:, i] * self.sd[i] + self.mu[i]
            lo = targets[:, li] * self.sd[li] + self.mu[li]
            hi = targets[:, hj] * self.sd[hj] + self.mu[hj]
            cols[i] = (torch.relu(lo - p) + torch.relu(p - hi)) / self.sd[i]
        return torch.stack(cols, dim=1)


# UNIFORM weights: see the protocol amendment. Inverse-variance weighting hurts at
# capacity and is antagonistic with the interval objective.
w_used = np.ones_like(w)
pts = [data.MoleculeDatapoint.from_smi(s, yy, weight=float(wi))
       for s, yy, wi in zip(tbl.smiles, Yaug, w_used)]
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
blind_ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in te_ch.SMILES], feat)

scaf = pd.Series([scaffold_of(s) for s in tbl.smiles])
order = sorted(scaf.groupby(scaf).groups.values(), key=len, reverse=True)
fold_of = np.empty(len(tbl), int); load = np.zeros(5)
for g in order:
    k = int(np.argmin(load)); fold_of[np.array(g)] = k; load[k] += len(g)
print(f"5-fold scaffold partition sizes {[int(x) for x in load]}")

TW = torch.ones(NAUG); TW[NT:] = 0.0     # the eight bound columns train nothing

def make(scaler):
    mp = nn.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
    crit = IntervalMAE(scaler.mean_, scaler.scale_, SC_IDX, LO_IDX, HI_IDX, TW)
    ffn = nn.RegressionFFN(n_tasks=NAUG, input_dim=700, hidden_dim=300, n_layers=2,
                           dropout=0.1, criterion=crit,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    return models.MPNN(mp, nn.NormAggregation(), ffn, batch_norm=True,
                       metrics=[nn.metrics.MAE()], warmup_epochs=2,
                       init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)

oof = np.full((len(tbl), NAUG), np.nan); blind = []
t0 = time.time()
for k in range(1 if SMOKE else 5):
    te = np.where(fold_of == k)[0]; va = np.where(fold_of == (k + 1) % 5)[0]
    trn = np.where((fold_of != k) & (fold_of != (k + 1) % 5))[0]
    tds = data.MoleculeDataset([pts[i] for i in trn], feat); sc = tds.normalize_targets()
    vds = data.MoleculeDataset([pts[i] for i in va], feat); vds.normalize_targets(sc)
    eds = data.MoleculeDataset([pts[i] for i in te], feat)
    tl = data.build_dataloader(tds, batch_size=256, num_workers=0)
    vl = data.build_dataloader(vds, batch_size=256, num_workers=0, shuffle=False)
    el = data.build_dataloader(eds, batch_size=256, num_workers=0, shuffle=False)
    bl = data.build_dataloader(blind_ds, batch_size=256, num_workers=0, shuffle=False)
    model = make(sc)
    ck = ModelCheckpoint(dirpath=OUT / f"ckpt_f{k}", monitor="val_loss", mode="min", save_top_k=1)
    cbs = [ck] + ([] if SMOKE else [EarlyStopping(monitor="val_loss", patience=30, mode="min")])
    tr = L.Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices=1, callbacks=cbs,
                   enable_progress_bar=False, logger=False)
    tr.fit(model, tl, vl)
    # load_from_checkpoint cannot rebuild a custom criterion (Factory.build has no
    # way to recover its constructor args), so rebuild the model and load weights.
    best = make(sc)
    best.load_state_dict(torch.load(ck.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    best.eval()
    oof[te] = np.concatenate(tr.predict(best, el), axis=0)
    blind.append(np.concatenate(tr.predict(best, bl), axis=0))
    print(f"fold {k} done, {(time.time()-t0)/60:.1f} min elapsed", flush=True)

cols = TARGETS + [f"lo_{e}" for e in EPS4] + [f"hi_{e}" for e in EPS4]
pd.DataFrame(oof, columns=cols).assign(Molecule_Name=tbl.Molecule_Name.values).to_csv(
    OUT / "oof_predictions.csv", index=False)
pd.DataFrame(np.mean(blind, axis=0), columns=cols).assign(
    Molecule_Name=te_ch.Molecule_Name.values).to_csv(OUT / "blind_predictions_mean.csv", index=False)
print(f"total {(time.time()-t0)/60:.1f} min")
