"""Chemprop member, take two: proper 5-fold scaffold partition + inverse-variance weights."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, models, nn

OUT = Path("analysis/aux_ac50_weighted_20260910")
SMOKE = "--smoke" in sys.argv
MAX_EPOCHS = 3 if SMOKE else 200
sys.path.insert(0, "analysis/metric_decision_layer_20260907")
from featurize import load_challenge, scaffold_of        # noqa: E402

tbl = pd.read_csv("analysis/aux_ac50_member_20260909/train_table.csv")
TARGETS = [c for c in tbl.columns if c not in ("Molecule_Name", "smiles", "InChIKey")]
SCORED = [f"{e}_pic50_direct_inhibition" for e in ("cyp1a2","cyp2c9","cyp2d6","cyp3a4")]
Y = tbl[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
print(f"rows {len(tbl)} targets {len(TARGETS)}")

# ---- inverse-variance weights from the scored endpoints' intervals
tr_ch, te_ch = load_challenge()
tr_ch["Molecule_Name"] = tr_ch["Molecule_Name"].astype(str)
te_ch["Molecule_Name"] = te_ch["Molecule_Name"].astype(str)
ci = tr_ch.set_index("Molecule_Name")
wsum = np.zeros(len(tbl)); wcnt = np.zeros(len(tbl))
for e in ("CYP1A2","CYP2C9","CYP2D6","CYP3A4"):
    lo = ci[f"{e}_pIC50_direct_inhibition_conf_low"].reindex(tbl.Molecule_Name).to_numpy(float)
    hi = ci[f"{e}_pIC50_direct_inhibition_conf_high"].reindex(tbl.Molecule_Name).to_numpy(float)
    m = np.isfinite(lo) & np.isfinite(hi)
    sig = np.clip((hi - lo) / (2 * 1.96), 0.02, None)
    wsum[m] += 1.0 / sig[m] ** 2; wcnt[m] += 1
w = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)
med = np.nanmedian(w); w = np.where(np.isfinite(w), w, med)
w = w / w.mean()
cap = np.quantile(w, 0.99); w = np.clip(w, None, cap); w = w / w.mean()
print(f"weights: min {w.min():.3f} median {np.median(w):.3f} max {w.max():.3f} "
      f"(cap at p99={cap:.2f}); {int((wcnt>0).sum())} rows have a scored interval")

# ---- proper 5-fold scaffold partition
scaf = pd.Series([scaffold_of(s) for s in tbl.smiles])
groups = scaf.groupby(scaf).groups
order = sorted(groups.values(), key=len, reverse=True)
fold_of = np.empty(len(tbl), int); load = np.zeros(5)
for g in order:
    k = int(np.argmin(load)); fold_of[np.array(g)] = k; load[k] += len(g)
print(f"5-fold scaffold partition sizes: {[int(x) for x in load]}  "
      f"({scaf.nunique()} distinct scaffolds)")

pts = [data.MoleculeDatapoint.from_smi(s, y, weight=float(wi))
       for s, y, wi in zip(tbl.smiles, Y, w)]
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
blind_ds = data.MoleculeDataset(
    [data.MoleculeDatapoint.from_smi(s) for s in te_ch.SMILES], feat)

def make(scaler, n):
    mp = nn.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
    ffn = nn.RegressionFFN(n_tasks=n, input_dim=700, hidden_dim=300, n_layers=2,
                           dropout=0.1, criterion=nn.metrics.MAE(),
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    return models.MPNN(mp, nn.NormAggregation(), ffn, batch_norm=True,
                       metrics=[nn.metrics.MAE()], warmup_epochs=2,
                       init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)

oof = np.full((len(tbl), len(TARGETS)), np.nan); blind = []
t0 = time.time()
for k in range(1 if SMOKE else 5):
    te = np.where(fold_of == k)[0]
    va = np.where(fold_of == (k + 1) % 5)[0]
    trn = np.where((fold_of != k) & (fold_of != (k + 1) % 5))[0]
    tds = data.MoleculeDataset([pts[i] for i in trn], feat); sc = tds.normalize_targets()
    vds = data.MoleculeDataset([pts[i] for i in va], feat); vds.normalize_targets(sc)
    eds = data.MoleculeDataset([pts[i] for i in te], feat)
    tl = data.build_dataloader(tds, batch_size=256, num_workers=0)
    vl = data.build_dataloader(vds, batch_size=256, num_workers=0, shuffle=False)
    el = data.build_dataloader(eds, batch_size=256, num_workers=0, shuffle=False)
    bl = data.build_dataloader(blind_ds, batch_size=256, num_workers=0, shuffle=False)
    model = make(sc, len(TARGETS))
    ck = ModelCheckpoint(dirpath=OUT / f"ckpt_f{k}", monitor="val_loss", mode="min", save_top_k=1)
    cbs = [ck] + ([] if SMOKE else [EarlyStopping(monitor="val_loss", patience=30, mode="min")])
    t = time.time()
    L.Trainer(accelerator="mps", devices=1, max_epochs=MAX_EPOCHS, logger=False,
              enable_progress_bar=False, callbacks=cbs).fit(model, tl, vl)
    el_t = time.time() - t
    print(f"fold {k}: train {len(trn)} val {len(va)} test {len(te)} | "
          f"{model.current_epoch} epochs in {el_t/60:.1f} min", flush=True)
    if SMOKE: break
    best = models.MPNN.load_from_checkpoint(ck.best_model_path)
    pr = L.Trainer(accelerator="mps", devices=1, logger=False, enable_progress_bar=False)
    oof[te] = np.concatenate(pr.predict(best, el))
    blind.append(np.concatenate(pr.predict(best, bl)))

if not SMOKE:
    o = pd.DataFrame(oof, columns=[f"{c}_pred" for c in TARGETS])
    o.insert(0, "Molecule_Name", tbl.Molecule_Name)
    o.to_csv(OUT / "oof_predictions.csv", index=False)
    cov = o[[c for c in o.columns if c.endswith("_pred")]].notna().any(axis=1).mean()
    b = pd.DataFrame(np.mean(blind, axis=0), columns=[f"{c}_pred" for c in TARGETS])
    b.insert(0, "Molecule_Name", te_ch.Molecule_Name.values)
    b.to_csv(OUT / "blind_predictions_mean.csv", index=False)
    print(f"\nOOF coverage {cov:.1%} (take one was 41%)  blind {b.shape}  "
          f"total {(time.time()-t0)/3600:.1f} h")
