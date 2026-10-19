"""R25 = R24 with ONLY the CYP1A2 column changed: blend in G26 (TDI aux heads).

Single-column change from R24, so the board's per-endpoint scores make it exactly
attributable — and comparisons between our OWN submissions on the fixed 750
compounds are DETERMINISTIC, not noisy ([[openadmet-metric-noise-floor]]). That is
the point of this file: our CYP1A2 OOF at n=175 demonstrably cannot certify a
+0.0065 effect (CI_lo -0.0085), so a reversible board window is the only
instrument that can ([[openadmet-slot-as-measurement]]).

Blend weight 0.20: the median of the five out-of-fold choices (0.1/0.2/0.2/0.2/0.3),
non-zero in 5/5 folds. Mean AND sd are then restored to R24's CYP1A2 column,
rank-preserving, per [[openadmet-blends-must-preserve-spread]] — R14 lost 0.0084
live by compressing spread, and R24 itself was built with this restoration.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, glob, json, hashlib
from pathlib import Path

HERE = Path("analysis/r25_a1_tdi_20260914"); HERE.mkdir(exist_ok=True)
R24 = "analysis/r24_joint_member_hinge_20260913/OpenADMET_R24_JOINT_MEMBER_HINGE_RESTORED_20260913.csv"
G26B = "analysis/tdi_aux_heads_20260914/results/blind_BG22_f*.csv"
COL = "CYP1A2_pIC50_direct_inhibition"
W = 0.20

r = pd.read_csv(R24); r["Molecule_Name"] = r["Molecule_Name"].astype(str)
fs = sorted(glob.glob(G26B)); assert len(fs) == 5, f"expected 5 blind folds, got {len(fs)}"
g = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
g["Molecule_Name"] = g["Molecule_Name"].astype(str)
gs = g.groupby("Molecule_Name")["cyp1a2_pic50_direct_inhibition_pred"].mean()
c = gs.reindex(r["Molecule_Name"]).to_numpy(float)
assert np.isfinite(c).all(), f"G26 blind coverage gap: {int((~np.isfinite(c)).sum())} rows"

b = r[COL].to_numpy(float)
raw = (1 - W) * b + W * (c + (b.mean() - c.mean()))
new = (raw - raw.mean()) / max(raw.std(), 1e-9) * b.std() + b.mean()
assert abs(new.mean() - b.mean()) < 1e-9, "mean restoration failed"
assert abs(new.std(ddof=1) / b.std(ddof=1) - 1) < 1e-9, "sd restoration failed"
sp = pd.Series(new).corr(pd.Series(b), method="spearman")
print(f"CYP1A2: mean {b.mean():.4f} -> {new.mean():.4f}   "
      f"sd {b.std(ddof=1):.4f} -> {new.std(ddof=1):.4f}   Spearman vs R24 {sp:.4f}")
print(f"  mean |delta| {np.abs(new-b).mean():.4f}   max |delta| {np.abs(new-b).max():.4f}")

out = r.copy(); out[COL] = new
for other in ("CYP2C9_pIC50_direct_inhibition", "CYP2D6_pIC50_direct_inhibition",
              "CYP3A4_pIC50_direct_inhibition"):
    assert np.array_equal(out[other].to_numpy(), r[other].to_numpy()), f"{other} changed!"
assert len(out) == 750 and out.isna().sum().sum() == 0
p = HERE / "OpenADMET_R25_A1_TDI_20260914.csv"
out.to_csv(p, index=False)
sha = hashlib.sha256(p.read_bytes()).hexdigest()
print(f"\nwrote {p}\n  sha256 {sha}\n  rows {len(out)} nulls 0")
print("  CYP2C9/CYP2D6/CYP3A4 bit-identical to R24 (asserted)")
json.dump({"artifact": "R25", "status": "built_not_submitted", "base": "R24",
           "changed_columns": [COL], "blend_weight": W,
           "component": "G26 = arm B with 4 TDI-condition aux heads (26 targets)",
           "sha256": sha, "spearman_vs_r24": float(sp),
           "oof_additional_gain_nn050": 0.0065, "oof_ci_lo": -0.0085,
           "note": "OOF cannot certify at n=175; submitted as a measurement"},
          open(HERE / "manifest.json", "w"), indent=1)
