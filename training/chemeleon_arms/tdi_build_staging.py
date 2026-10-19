"""Stage G26: the 22 production trunk targets + 4 NEW TDI-condition pIC50 aux heads.

## Why

The single largest effect measured in this project is the AUX-HEAD CLIFF: narrowing
the trunk from 22 targets to the 4 scored ones costs 0.127/0.312/0.155/0.324 ST-RAE
(`analysis/chemeleon_multitask_20260910/GROUPING_RESULT.md`). Aux heads are
load-bearing by a margin 5-20x anything else we have found.

The organizers released a second dose-response arm we have never trained on.
`cyp-challenge-TRAIN_TDI.csv` carries `CYP*_pIC50_TDI_condition` for all four
isoforms with credible intervals: 1,413 / 1,285 / 1,497 / 3,583 labels, i.e. it
roughly DOUBLES per-isoform potency supervision, plus 1,240 compounds absent from
the inhibition file entirely (+25% trunk rows).

Verified 2026-09-14: **zero of the 22 trunk targets is a TDI column.** Only
`is_TDI` was ever used (classification track), and the TDI-condition pIC50 was
tried as PSEUDO-LABELS for the scored target, which failed on calibration --
band-dependent bias flipping sign at high potency, resting on 25 compounds in the
>=6 band ([[openadmet-organizer-data-unread]]).

**As an AUX HEAD no calibration is required.** The model gets its own output and
learns any internal mapping; the shared trunk absorbs the representational
benefit. The blocker that killed the pseudo-label route is structurally absent.

## Design: minimal diff against the control

The runner defines `GROUPS["G22"] = ALL_T`, i.e. every target column in the table,
so adding 4 columns makes G22 become G26 with the runner UNCHANGED. That
guarantees exact config parity with the arm B control at stamp 20260910-2221
(d_h 2048, depth 4, ffn 1024, dropout 0.2, random init, MAE, inverse-variance
sample weights, batch 128) and, critically, the same `L.seed_everything(SEED + k)`
so seed = 42 + fold matches the control fold-for-fold. Tonight's descriptor
experiment was confounded precisely by a seed mismatch and a best-vs-final
checkpoint asymmetry ([[openadmet-chemprop-xd-and-ckpt-traps]]); reusing the
control's own runner avoids reintroducing either.

## Two changes at once, and why that is the right sequencing

This stages BOTH the 4 new heads AND the 1,240 new rows. They are confounded by
design: the cheap question is "does the TDI arm help at all", and decomposing a
null would be wasted money. If it WINS, the follow-up (4 heads on the original
31,669 rows only, isolating heads from data) costs another ~$5 and is worth it.

## Correctness checks enforced below

- blind leak on STRUCTURE, not names ([[openadmet-blind-leak-check-on-structure]]):
  0/1,240 overlap by name, canonical SMILES, full InChIKey and stereo-insensitive
  InChIKey skeleton (re-asserted here)
- folds are strictly scaffold-grouped in the control (0 of 18,945 scaffolds span
  folds), so the 182 new compounds sharing a known scaffold INHERIT that
  scaffold's fold, and the 1,058 novel scaffolds are assigned to balance sizes
- new rows get sample weight 1.0, the mean of the existing inverse-variance
  weights, i.e. deliberately neutral: they carry no scored label, so the
  scored-uncertainty weighting has nothing to say about them
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shutil, json
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

SRC = Path("analysis/chemeleon_multitask_20260910/staging")
HERE = Path("analysis/tdi_aux_heads_20260914"); HERE.mkdir(exist_ok=True)
OUT = HERE / "staging"; OUT.mkdir(exist_ok=True)
EPS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TDI_SRC = "tmp/organizer_data/cyp-challenge-TRAIN_TDI.csv"
INH_SRC = "tmp/organizer_data/cyp-challenge-TRAIN_inhibition.csv"

def scaf(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return s
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception: return s
def canon(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else None
def ikey(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m) if m else None

tbl = pd.read_csv(SRC / "train_table.csv"); tbl["Molecule_Name"] = tbl["Molecule_Name"].astype(str)
fold = np.load(SRC / "folds.npy"); w = np.load(SRC / "weights.npy")
blind = pd.read_csv(SRC / "blind.csv")
assert len(tbl) == len(fold) == len(w), "control staging length mismatch"
print(f"control table {tbl.shape}, folds {fold.shape}, weights {w.shape}")

tdi = pd.read_csv(TDI_SRC); inh = pd.read_csv(INH_SRC)
for d in (tdi, inh): d["Molecule_Name"] = d["Molecule_Name"].astype(str)
NEWCOLS = {ep: f"{ep.lower()}_pic50_tdi_condition" for ep in EPS}
assert not any(c in tbl.columns for c in NEWCOLS.values()), "TDI column already staged"

# ---- 1. attach TDI heads to existing rows
tmap = tdi.set_index("Molecule_Name")
for ep, c in NEWCOLS.items():
    tbl[c] = tbl["Molecule_Name"].map(tmap[f"{ep}_pIC50_TDI_condition"])
print("TDI labels attached to existing rows: " +
      ", ".join(f"{ep} {int(tbl[c].notna().sum())}" for ep, c in NEWCOLS.items()))

# ---- 2. the 1,240 compounds absent from the inhibition file
new = tdi[~tdi.Molecule_Name.isin(set(inh.Molecule_Name))].reset_index(drop=True)
new = new[~new.Molecule_Name.isin(set(tbl.Molecule_Name))].reset_index(drop=True)
print(f"new compounds to append: {len(new)}")

# BLIND LEAK CHECK on structure, not names
bl_can = {canon(s) for s in blind.SMILES}; bl_can.discard(None)
bl_ik = {ikey(s) for s in blind.SMILES}; bl_ik.discard(None)
bl_sk = {k[:14] for k in bl_ik}
nc = [canon(s) for s in new.SMILES]; ni = [ikey(s) for s in new.SMILES]
leaks = dict(
    name=len(set(new.Molecule_Name) & set(blind.Molecule_Name.astype(str))),
    canonical=sum(c in bl_can for c in nc),
    inchikey=sum(k in bl_ik for k in ni),
    skeleton=sum(bool(k) and k[:14] in bl_sk for k in ni))
print(f"BLIND LEAK CHECK: {leaks}")
assert sum(leaks.values()) == 0, f"BLIND LEAK -- refusing to stage: {leaks}"

# ---- 3. scaffold-preserving fold assignment
tbl["_scaf"] = [scaf(s) for s in tbl.smiles]
sc2fold = dict(zip(tbl["_scaf"], fold))
spans = tbl.groupby("_scaf")["_scaf"].size().shape[0]
chk = pd.DataFrame({"s": tbl["_scaf"], "f": fold}).groupby("s")["f"].nunique()
assert int((chk > 1).sum()) == 0, "control folds are NOT scaffold-grouped"
new["_scaf"] = [scaf(s) for s in new.SMILES]
counts = np.bincount(fold, minlength=5).astype(float)
newfold = np.empty(len(new), dtype=int); inherited = 0
# novel scaffolds grouped together so a scaffold never spans folds
novel = [s for s in new["_scaf"].unique() if s not in sc2fold]
rng = np.random.default_rng(20260914)
assign = {}
for s in sorted(novel):
    k = int(np.argmin(counts)); assign[s] = k
    counts[k] += int((new["_scaf"] == s).sum())
for i, s in enumerate(new["_scaf"]):
    if s in sc2fold: newfold[i] = sc2fold[s]; inherited += 1
    else: newfold[i] = assign[s]
print(f"folds: {inherited} inherited from known scaffolds, {len(new)-inherited} "
      f"assigned across {len(novel)} novel scaffolds")

# ---- 4. assemble
rows = pd.DataFrame({"Molecule_Name": new.Molecule_Name, "smiles": new.SMILES})
if "InChIKey" in tbl.columns: rows["InChIKey"] = ni
for c in tbl.columns:
    if c in ("Molecule_Name", "smiles", "InChIKey", "_scaf"): continue
    rows[c] = np.nan
for ep, c in NEWCOLS.items():
    rows[c] = new[f"{ep}_pIC50_TDI_condition"].to_numpy()
rows["_scaf"] = new["_scaf"]
full = pd.concat([tbl, rows[tbl.columns]], ignore_index=True)
full_fold = np.concatenate([fold, newfold])
full_w = np.concatenate([w, np.ones(len(new))])

chk2 = pd.DataFrame({"s": full["_scaf"], "f": full_fold}).groupby("s")["f"].nunique()
assert int((chk2 > 1).sum()) == 0, "scaffold-disjointness BROKEN by the append"
full = full.drop(columns=["_scaf"])
TARGETS = [c for c in full.columns if c not in ("Molecule_Name", "smiles", "InChIKey")]
assert len(TARGETS) == 26, f"expected 26 targets, got {len(TARGETS)}"
assert len(full) == len(full_fold) == len(full_w)

for f in ("blind.csv", "bounds.csv", "chemeleon_mp.pt", "organizer_cyp.pth",
          "sagemaker_runner.py"):
    shutil.copy(SRC / f, OUT / f)
full.to_csv(OUT / "train_table.csv", index=False)
np.save(OUT / "folds.npy", full_fold); np.save(OUT / "weights.npy", full_w)
print(f"\nstaged {OUT}")
print(f"  rows {len(full)} (control {len(tbl)}, +{len(new)})   targets {len(TARGETS)}")
print(f"  rows per fold: {np.bincount(full_fold)}")
print(f"  new TDI heads: " + ", ".join(
    f"{c} {int(full[c].notna().sum())}" for c in NEWCOLS.values()))
json.dump({"rows": int(len(full)), "rows_added": int(len(new)),
           "targets": TARGETS, "n_targets": len(TARGETS),
           "tdi_heads": {c: int(full[c].notna().sum()) for c in NEWCOLS.values()},
           "blind_leak_check": leaks, "folds_inherited": int(inherited),
           "control": "arm B stamp 20260910-2221 (same runner, G22 -> 26 targets)"},
          open(HERE / "staging_manifest.json", "w"), indent=1)
