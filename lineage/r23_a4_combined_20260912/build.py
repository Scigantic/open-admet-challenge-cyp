"""R23 = R17 with CYP3A4 improved by BOTH metric-aligned interventions.

  B: blend in ivB2048 (interval-objective, UNIFORM sample weights -- the good cell
     of the 2x2; interval+invvar is the worst cell, do not use it), weight chosen
     by hinge on OOF.
  A: then a hinge-fitted correction on the four discarded aux heads.

They are nearly orthogonal (per-compound delta correlation +0.080) and exactly
additive: +0.0161 at NN>=0.55 vs +0.0091 and +0.0071 alone.

FIRST candidate this session to clear all three predeclared gates:
  magnitude +0.0161 >= +0.005                                    PASS
  sign consistent 3/3 strata                                     PASS
  bootstrap 95% CI lower bound > 0 at ALL strata
    (+0.0003 / +0.0023 / +0.0023, P=0.978/0.993/0.986)           PASS
Aux-head component separately permutation-validated (0/40 nulls reached it).
Still an OFFLINE result; transfer is not guaranteed.
Building a file is not submitting one.
"""
from __future__ import annotations
import glob, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
R17 = Path("analysis/r17_revert_20260912/OpenADMET_R17_BEST_KNOWN_COLUMNS_20260912.csv")
E = "cyp3a4"; TARGET = "CYP3A4_pIC50_direct_inhibition"
ALL = ["CYP1A2_pIC50_direct_inhibition", "CYP2C9_pIC50_direct_inhibition",
       "CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"]
AUX = [f"{E}_pic50_qhts_pred", f"{E}_pic50_chembl_pred",
       f"{E}_max_response_pred", f"{E}_log2fc_pred"]
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; L2 = 0.1

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
mm = lambda b, c, w: (1 - w) * b + w * (c + (b.mean() - c.mean()))
hs = lambda p, lo, hi: np.sum(np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None))

# ---------- OOF: choose blend weight, then fit aux correction ----------
d = pd.read_csv("analysis/r08_local_baseline_20260908/r08_like_oof_CYP3A4.csv")
d["Molecule_Name"] = d["Molecule_Name"].astype(str)
d = d.dropna(subset=["label", "conf_low", "conf_high", "candidate"]).reset_index(drop=True)
ac = pd.read_csv("analysis/aux_ac50_weighted_20260910/oof_predictions.csv")
ac["Molecule_Name"] = ac["Molecule_Name"].astype(str); ac = ac.set_index("Molecule_Name")
mt = pd.read_csv("analysis/briford_recipe_parity_20260907/member_results/mt_aux8/oof_predictions.csv")
mt["Molecule_Name"] = mt["Molecule_Name"].astype(str)
ivo = pd.read_csv("analysis/chemeleon_multitask_20260910/oof_ivB2048.csv")
ivo["Molecule_Name"] = ivo["Molecule_Name"].astype(str)
d = d[d["Molecule_Name"].isin(ac.index)].reset_index(drop=True)
d["mt"] = d["Molecule_Name"].map(mt.groupby("Molecule_Name")[f"{E}_pic50_direct_inhibition_pred"].mean())
d["iv"] = d["Molecule_Name"].map(ivo.groupby("Molecule_Name")[f"{E}_pic50_direct_inhibition_pred"].mean())
d = d.dropna(subset=["mt", "iv"]).reset_index(drop=True)
A_ = ac.loc[d["Molecule_Name"]]
direct = A_[f"{E}_pic50_direct_inhibition_pred"].to_numpy(float)
base = mm(mm(mm(d["candidate"].to_numpy(), d["mt"].to_numpy(), 0.18), direct, 0.20),
          d["mt"].to_numpy(), 0.20)
lo, hi = d.conf_low.to_numpy(), d.conf_high.to_numpy()

best_w = min(W_GRID, key=lambda w: hs(mm(base, d["iv"].to_numpy(), w), lo, hi))
stage_b = mm(base, d["iv"].to_numpy(), best_w)
print(f"interval-model blend weight chosen on OOF: {best_w}")

Xraw = np.nan_to_num(np.column_stack(
    [(A_[c].to_numpy(float) - direct) if "pic50" in c else A_[c].to_numpy(float) for c in AUX]), nan=0.0)
mu, sd = Xraw.mean(0), Xraw.std(0) + 1e-9
X = (Xraw - mu) / sd
w = minimize(lambda v: hs(stage_b + X @ v, lo, hi) / len(base) + L2 * np.sum(v ** 2),
             np.zeros(4), method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-8}).x
print("aux coefficients:", {c: round(float(x), 4) for c, x in zip(AUX, w)})
oof_corr = X @ w

# ---------- blind ----------
r17 = pd.read_csv(R17); ids = r17["Molecule_Name"].astype(str).tolist()
ab = pd.read_csv("analysis/aux_ac50_weighted_20260910/blind_predictions_mean.csv")
ab["Molecule_Name"] = ab["Molecule_Name"].astype(str); ab = ab.set_index("Molecule_Name").loc[ids]
ivb_files = sorted(glob.glob("analysis/chemeleon_multitask_20260910/results/20260911-ivB2048/blind_B_f*.csv"))
assert len(ivb_files) == 5, f"expected 5 ivB2048 blind folds, got {len(ivb_files)}"
ivb = pd.concat([pd.read_csv(f) for f in ivb_files], ignore_index=True)
ivb["Molecule_Name"] = ivb["Molecule_Name"].astype(str)
iv_blind = ivb.groupby("Molecule_Name")[f"{E}_pic50_direct_inhibition_pred"].mean().loc[ids].to_numpy(float)
direct_b = ab[f"{E}_pic50_direct_inhibition_pred"].to_numpy(float)
base_b = pd.to_numeric(r17[TARGET], errors="raise").to_numpy(float)

stage_b_blind = mm(base_b, iv_blind, best_w)
Xb = (np.nan_to_num(np.column_stack(
    [(ab[c].to_numpy(float) - direct_b) if "pic50" in c else ab[c].to_numpy(float) for c in AUX]),
    nan=0.0) - mu) / sd
corr_b = Xb @ w
new = stage_b_blind + corr_b
print(f"blind: iv-blend moved mean {base_b.mean():.4f}->{stage_b_blind.mean():.4f}; "
      f"aux corr mean={corr_b.mean():+.4f} (OOF {oof_corr.mean():+.4f})")
print(f"CYP3A4 column: mean {base_b.mean():.4f} -> {new.mean():.4f}  "
      f"sd {base_b.std(ddof=1):.4f} -> {new.std(ddof=1):.4f}")

out = r17.copy(); out[TARGET] = new
for c in ALL:
    if c != TARGET and not np.array_equal(out[c].to_numpy(), r17[c].to_numpy()):
        raise SystemExit(f"{c} changed; must stay byte-identical to R17")
assert len(out) == 750 and np.isfinite(new).all()
path = HERE / "OpenADMET_R23_A4_COMBINED_20260912.csv"
out.to_csv(path, index=False)
json.dump({
    "artifact_type": "isolated_single_column_bet", "status": "built_not_submitted",
    "base": {"path": str(R17), "sha256": sha(R17)},
    "changed_endpoints": [TARGET], "unchanged_endpoints": [c for c in ALL if c != TARGET],
    "stage_B": {"component": "ivB2048 (interval objective, UNIFORM weights)", "weight": best_w},
    "stage_A": {"features": AUX, "coefficients": {c: float(x) for c, x in zip(AUX, w)}, "l2": L2},
    "gates": {"magnitude_NN055": 0.0161, "sign_consistent": "3/3",
               "bootstrap_ci_lower": [0.0003, 0.0023, 0.0023], "verdict": "ALL PASS"},
    "column_stats": {"mean_before": float(base_b.mean()), "mean_after": float(new.mean()),
                      "sd_before": float(base_b.std(ddof=1)), "sd_after": float(new.std(ddof=1))},
    "output": {"path": str(path), "sha256": sha(path), "rows": int(len(out))},
}, open(HERE / "manifest.json", "w"), indent=1)
print(f"\nwrote {path}\n  sha256 {sha(path)}")
