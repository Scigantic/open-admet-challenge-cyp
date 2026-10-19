"""SageMaker runner: one ARM, one FOLD per job, so 15 jobs run in parallel.
Ships as a plain script to the training channel; the image executes it directly.
Mirrors analysis/chemeleon_multitask_20260910/train.py exactly except for paths."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, models, nn

CHANNEL = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
HP_FILE = Path("/opt/ml/input/config/hyperparameters.json")
hp = json.loads(HP_FILE.read_text()) if HP_FILE.exists() else {}
def g(k, d=None): return hp.get(k, d)

ARMS = {
    "A": dict(d_h=700,  depth=5, ffn=300,  dropout=0.1, init="random"),
    "B": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="random"),
    "C": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="chemeleon"),
    # D: the ORGANIZERS' own CYP-pretrained checkpoint, not the general foundation.
    # Verified 4/4 name+shape match; mean |delta| from the general weights 0.01406,
    # so CYP fine-tuning genuinely moved them and this is a distinct initialisation.
    "D": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="organizer"),
}
ARM = str(g("arm", "C")); FOLD = int(g("fold", 0))
MAX_EPOCHS = int(g("max_epochs", 200)); PATIENCE = int(g("patience", 30))
BATCH = int(g("batch_size", 128)); SEED = int(g("seed", 42))
OBJECTIVE = str(g("objective", "mae"))          # mae | interval
WEIGHTING = str(g("weighting", "invvar"))       # invvar | uniform  (SAMPLE weights)
TASK_WEIGHTING = str(g("task_weighting", "uniform"))  # uniform | denominator (HEAD weights)
RESULT_S3 = g("result_s3")
assert ARM in ARMS, f"bad arm {ARM}"
CFG = ARMS[ARM]
print(f"ARM {ARM} FOLD {FOLD}: {CFG} batch={BATCH} objective={OBJECTIVE} weighting={WEIGHTING}", flush=True)

tbl = pd.read_csv(CHANNEL / "train_table.csv")
blind = pd.read_csv(CHANNEL / "blind.csv")
ALL_T = [c for c in tbl.columns if c not in ("Molecule_Name", "smiles", "InChIKey")]
# ---- task grouping. d6_isoform6 (6 isoforms) beat both single-task (0.8980) and
# 22-target (0.9009) on CYP2D6, so the sweet spot may be a MIDDLE grouping.
_sc  = [c for c in ALL_T if c.endswith("_pic50_direct_inhibition")]
_qh  = [c for c in ALL_T if c.endswith("_pic50_qhts")]
_lg  = [c for c in ALL_T if c.endswith("_log2fc")]
GROUPS = {
    "G4":  _sc,                  # scored only -- closest to per-endpoint training
    "G8":  _sc + _qh,            # + same-assay-family dose-response
    "G12": _sc + _qh + _lg,      # + cytotoxicity fold-change
    "G22": ALL_T,                # everything -- what production uses
}
GROUP = str(g("group", "G22"))
assert GROUP in GROUPS, f"bad group {GROUP}"
TARGETS = GROUPS[GROUP]
print(f"GROUP {GROUP}: {len(TARGETS)} targets", flush=True)

# ---- interval objective (ported from chemprop_interval_20260910, which measured
# +0.0561 macro on the 700-wide architecture at batch 256). Bounds ride as extra
# target columns at task-weight zero so the loss can see them without training on
# them. Loss runs in scaled space: unscale, hinge in real pIC50 units, rescale.
import torch
from chemprop.nn.metrics import ChempropMetric
SC4 = [c for c in TARGETS if c.endswith("_pic50_direct_inhibition")]
bounds = pd.read_csv(CHANNEL / "bounds.csv")
bounds["Molecule_Name"] = bounds["Molecule_Name"].astype(str)
bounds = bounds.set_index("Molecule_Name").reindex(tbl.Molecule_Name.astype(str))

class IntervalMAE(ChempropMetric):
    def __init__(self, mean, scale, sc_idx, lo_idx, hi_idx, task_weights):
        super().__init__(task_weights=task_weights)
        self.register_buffer("mu", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("sd", torch.as_tensor(scale, dtype=torch.float))
        self.sc_idx = list(sc_idx); self.lo_idx = list(lo_idx); self.hi_idx = list(hi_idx)
        self.n_real = int(min(lo_idx))
    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        if mask is not None:
            mask = mask.clone(); mask[:, self.n_real:] = False
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)
    def _calc_unreduced_loss(self, preds, targets, *args):
        L = (preds - targets).abs()
        cols = [L[:, i] for i in range(L.shape[1])]
        for i, li, hj in zip(self.sc_idx, self.lo_idx, self.hi_idx):
            p  = preds[:, i] * self.sd[i] + self.mu[i]
            lo = targets[:, li] * self.sd[li] + self.mu[li]
            hi = targets[:, hj] * self.sd[hj] + self.mu[hj]
            cols[i] = (torch.relu(lo - p) + torch.relu(p - hi)) / self.sd[i]
        return torch.stack(cols, dim=1)
Y = tbl[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
NT = len(TARGETS)
if OBJECTIVE == "interval":
    SC_IDX = [TARGETS.index(c) for c in SC4]
    extra = []
    for c in SC4:
        e = c.split("_")[0].upper()
        extra.append(bounds[f"{e}_lo"].to_numpy(float))
    for c in SC4:
        e = c.split("_")[0].upper()
        extra.append(bounds[f"{e}_hi"].to_numpy(float))
    Y = np.hstack([Y, np.column_stack(extra)])
    LO_IDX = list(range(NT, NT + len(SC4)))
    HI_IDX = list(range(NT + len(SC4), NT + 2 * len(SC4)))
    NAUG = Y.shape[1]
    print(f"  interval objective: {NT} real + {NAUG-NT} bound columns", flush=True)
else:
    NAUG = NT
w = np.load(CHANNEL / "weights.npy")
fold_of = np.load(CHANNEL / "folds.npy")
assert len(w) == len(tbl) == len(fold_of), "weights/folds length mismatch"
if WEIGHTING == "uniform":
    w = np.ones_like(w)
    print("  UNIFORM sample weights", flush=True)
print(f"rows {len(tbl)} targets {len(TARGETS)} blind {len(blind)}", flush=True)

def load_pretrained(mp, which):
    import torch
    if which == "chemeleon":
        ck = torch.load(CHANNEL / "chemeleon_mp.pt", map_location="cpu", weights_only=False)
        src = ck.get("state_dict", ck)
    else:                                   # organizers' released CYP model
        ck = torch.load(CHANNEL / "organizer_cyp.pth", map_location="cpu", weights_only=False)
        src = {k[len("message_passing."):]: v for k, v in ck.items()
               if k.startswith("message_passing.")}
    tgt = mp.state_dict()
    common = [k for k in tgt if k in src]
    bad = [k for k in common if tuple(tgt[k].shape) != tuple(src[k].shape)]
    if len(common) != len(tgt) or bad:
        raise SystemExit(f"{which} load REFUSED: {len(common)}/{len(tgt)} keys, mismatches {bad}")
    mp.load_state_dict({k: src[k] for k in tgt})
    print(f"  {which} weights loaded: {len(common)}/{len(tgt)} tensors", flush=True)
    return mp

def make(scaler, n):
    mp = nn.BondMessagePassing(d_h=CFG["d_h"], depth=CFG["depth"], dropout=CFG["dropout"])
    if CFG["init"] in ("chemeleon", "organizer"): mp = load_pretrained(mp, CFG["init"])
    if OBJECTIVE == "interval":
        TW = torch.ones(NAUG); TW[NT:] = 0.0
        if TASK_WEIGHTING == "denominator":
            TW[SC_IDX] = torch.as_tensor(DEN_W, dtype=torch.float)
        crit = IntervalMAE(scaler.mean_, scaler.scale_, SC_IDX, LO_IDX, HI_IDX, TW)
    else:
        crit = nn.metrics.MAE()
    ffn = nn.RegressionFFN(n_tasks=n, input_dim=CFG["d_h"], hidden_dim=CFG["ffn"], n_layers=2,
                           dropout=CFG["dropout"], criterion=crit,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    return models.MPNN(mp, nn.NormAggregation(), ffn, batch_norm=True,
                       metrics=[nn.metrics.MAE()], warmup_epochs=2,
                       init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
pts = [data.MoleculeDatapoint.from_smi(s, y, weight=float(wi))
       for s, y, wi in zip(tbl.smiles, Y, w)]
blind_ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in blind.SMILES], feat)

k = FOLD
te = np.where(fold_of == k)[0]
va = np.where(fold_of == (k + 1) % 5)[0]
trn = np.where((fold_of != k) & (fold_of != (k + 1) % 5))[0]
# ---- head weights proportional to 1/D_e, because macro ST-RAE already is.
# macro = (1/4) sum_e [ sum_i hinge(i,e) / D_e ], so d(macro)/d(hinge at e) = 1/(4 D_e).
# Equal head weights therefore do NOT optimise the metric. D_e is computed on THIS
# fold's training rows only, never on val or test.
DEN_W = np.ones(len(SC4), dtype=float)   # SC_IDX only exists under objective=interval
if OBJECTIVE == "interval" and TASK_WEIGHTING == "denominator":
    inv = []
    for c in SC4:
        e = c.split("_")[0].upper()
        yv = pd.to_numeric(tbl[c], errors="coerce").to_numpy(float)[trn]
        lo = bounds[f"{e}_lo"].to_numpy(float)[trn]
        hi = bounds[f"{e}_hi"].to_numpy(float)[trn]
        m = np.isfinite(yv) & np.isfinite(lo) & np.isfinite(hi)
        mt = yv[m].mean()
        D = (np.clip(mt - hi[m], 0, None) + np.clip(lo[m] - mt, 0, None)).sum()
        inv.append(1.0 / max(D, 1e-9))
    inv = np.asarray(inv, dtype=float)
    DEN_W = inv / inv.mean()          # mean 1 across the four scored heads
    print(f"  DENOMINATOR head weights {dict(zip([c.split('_')[0] for c in SC4], DEN_W.round(3)))}",
          flush=True)

L.seed_everything(SEED + k, workers=True)
tds = data.MoleculeDataset([pts[i] for i in trn], feat); sc = tds.normalize_targets()
vds = data.MoleculeDataset([pts[i] for i in va], feat); vds.normalize_targets(sc)
eds = data.MoleculeDataset([pts[i] for i in te], feat)
tl = data.build_dataloader(tds, batch_size=BATCH, num_workers=0)
vl = data.build_dataloader(vds, batch_size=BATCH, num_workers=0, shuffle=False)
el = data.build_dataloader(eds, batch_size=BATCH, num_workers=0, shuffle=False)
bl = data.build_dataloader(blind_ds, batch_size=BATCH, num_workers=0, shuffle=False)
model = make(sc, NAUG)
ck = ModelCheckpoint(dirpath=str(MODEL_DIR / f"ckpt_{ARM}{GROUP}_f{k}"), monitor="val_loss",
                     mode="min", save_top_k=1)
t0 = time.time()
L.Trainer(accelerator="auto", devices=1, max_epochs=MAX_EPOCHS, logger=False,
          enable_progress_bar=False,
          callbacks=[ck, EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")]
          ).fit(model, tl, vl)
print(f"fold {k}: train {len(trn)} val {len(va)} test {len(te)} | "
      f"{model.current_epoch} epochs in {(time.time()-t0)/60:.1f} min", flush=True)
best = models.MPNN.load_from_checkpoint(ck.best_model_path)
pr = L.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)
oof_pred = np.concatenate(pr.predict(best, el))
blind_pred = np.concatenate(pr.predict(best, bl))
cols = [f"{c}_pred" for c in TARGETS]
oof_pred = oof_pred[:, :NT]; blind_pred = blind_pred[:, :NT]
o = pd.DataFrame(oof_pred, columns=cols); o.insert(0, "Molecule_Name", tbl.Molecule_Name.values[te])
b = pd.DataFrame(blind_pred, columns=cols); b.insert(0, "Molecule_Name", blind.Molecule_Name.values)
o.to_csv(MODEL_DIR / f"oof_{ARM}{GROUP}_f{k}.csv", index=False)
b.to_csv(MODEL_DIR / f"blind_{ARM}{GROUP}_f{k}.csv", index=False)
(MODEL_DIR / f"manifest_{ARM}{GROUP}_f{k}.json").write_text(json.dumps({
    "arm": ARM, "group": GROUP, "objective": OBJECTIVE, "weighting": WEIGHTING, "n_targets": len(TARGETS), "fold": k, "config": CFG, "batch": BATCH, "seed": SEED + k,
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "epochs_run": int(model.current_epoch), "best_ckpt": str(ck.best_model_path),
    "n_train": int(len(trn)), "n_val": int(len(va)), "n_test": int(len(te)),
    "minutes": round((time.time()-t0)/60, 1),
}, indent=1))
# NOTE: the container has NO aws CLI. SageMaker uploads everything written to
# /opt/ml/model to OutputDataConfig automatically as model.tar.gz, so an explicit
# copy is unnecessary AND fatal -- it crashed the 2026-09-10 verification job at
# the very last line, after 97 successful epochs. Do not reintroduce it.
print(f"artifacts written to {MODEL_DIR}: {sorted(p.name for p in MODEL_DIR.glob('*'))}", flush=True)
print("DONE", flush=True)
