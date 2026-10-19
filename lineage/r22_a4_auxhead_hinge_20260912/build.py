"""R22 = R17 with ONLY CYP3A4 corrected by a metric-aligned (hinge) fit on the
four auxiliary heads the pipeline normally discards.

Evidence: analysis/auxhead_hinge_fit_20260912/
  - gain +0.0037/+0.0050/+0.0071 at NN>=0.45/0.50/0.55 against the VERIFIED
    3-layer live CYP3A4 baseline (n=1125/739/430, P(gain>0)=0.94/0.97/0.95)
  - permutation null: 0 of 40 shuffles reached the observed gain at any
    stratum (one-sided p=0.000, null max +0.0021 vs observed +0.0091) -- the
    signal is not a 4-parameter fitting artifact
  - fitting to the metric is what makes it work: MSE fitting the same features
    LOSES 0.031-0.058 (analysis/auxhead_residual_20260912/)
  - it does NOT clear the predeclared bootstrap gate (95% CI lower bound
    -0.0012 at NN>=0.55), so this is a disclosed bet, not a certified gain

Why CYP3A4 is a defensible place to spend a round: it is the only endpoint whose
interventions have ever transferred live (R11's entire gain was CYP3A4; R14
improved it even while losing overall; it was the only shipped component gaining
on blind-like rows).

Feature standardization uses the OOF mean/sd, never the blind set's, so the
transform matches what was fitted and no blind marginal leaks in.
Building a file is not submitting one.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
R17 = Path("analysis/r17_revert_20260912/OpenADMET_R17_BEST_KNOWN_COLUMNS_20260912.csv")
AC_OOF = Path("analysis/aux_ac50_weighted_20260910/oof_predictions.csv")
AC_BLIND = Path("analysis/aux_ac50_weighted_20260910/blind_predictions_mean.csv")
MT_OOF = Path("analysis/briford_recipe_parity_20260907/member_results/mt_aux8/oof_predictions.csv")
OOF_REF = Path("analysis/r08_local_baseline_20260908/r08_like_oof_CYP3A4.csv")
L2 = 0.1                      # modal choice of the nested inner-fold selection
TARGET = "CYP3A4_pIC50_direct_inhibition"
E = "cyp3a4"
ALL = ["CYP1A2_pIC50_direct_inhibition", "CYP2C9_pIC50_direct_inhibition",
       "CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"]
AUX = [f"{E}_pic50_qhts_pred", f"{E}_pic50_chembl_pred",
       f"{E}_max_response_pred", f"{E}_log2fc_pred"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mm(b, c, w):
    return (1 - w) * b + w * (c + (b.mean() - c.mean()))


def hinge_sum(p, lo, hi):
    return np.sum(np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None))


def build_features(frame, direct):
    cols = []
    for c in AUX:
        v = frame[c].to_numpy(float)
        cols.append(v - direct if "pic50" in c else v)
    return np.nan_to_num(np.column_stack(cols), nan=0.0)


def main():
    # ---------- fit on every labelled OOF row ----------
    d = pd.read_csv(OOF_REF)
    d["Molecule_Name"] = d["Molecule_Name"].astype(str)
    d = d.dropna(subset=["label", "conf_low", "conf_high", "candidate"]).reset_index(drop=True)
    ac = pd.read_csv(AC_OOF); ac["Molecule_Name"] = ac["Molecule_Name"].astype(str)
    ac = ac.set_index("Molecule_Name")
    mt = pd.read_csv(MT_OOF); mt["Molecule_Name"] = mt["Molecule_Name"].astype(str)
    mtv = mt.groupby("Molecule_Name")[f"{E}_pic50_direct_inhibition_pred"].mean()
    d = d[d["Molecule_Name"].isin(ac.index)].reset_index(drop=True)
    d["mt"] = d["Molecule_Name"].map(mtv)
    d = d.dropna(subset=["mt"]).reset_index(drop=True)
    A = ac.loc[d["Molecule_Name"]]

    direct_oof = A[f"{E}_pic50_direct_inhibition_pred"].to_numpy(float)
    l1 = mm(d["candidate"].to_numpy(), d["mt"].to_numpy(), 0.18)
    l2 = mm(l1, direct_oof, 0.20)
    base_oof = mm(l2, d["mt"].to_numpy(), 0.20)

    X_raw = build_features(A, direct_oof)
    mu, sd = X_raw.mean(0), X_raw.std(0) + 1e-9      # OOF scaler, reused on blind
    X = (X_raw - mu) / sd
    lo = d["conf_low"].to_numpy(); hi = d["conf_high"].to_numpy()

    w = minimize(lambda v: hinge_sum(base_oof + X @ v, lo, hi) / len(base_oof) + L2 * np.sum(v ** 2),
                 np.zeros(X.shape[1]), method="Powell",
                 options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-8}).x
    print(f"fitted on {len(d)} OOF rows, L2={L2}")
    for name, coef in zip(AUX, w):
        print(f"  {name:38} {coef:+.4f}")
    oof_corr = X @ w
    print(f"  OOF correction: mean={oof_corr.mean():+.4f} sd={oof_corr.std():.4f} "
          f"range=[{oof_corr.min():+.3f},{oof_corr.max():+.3f}]")

    # ---------- apply to the blind set ----------
    r17 = pd.read_csv(R17)
    ids = r17["Molecule_Name"].astype(str).tolist()
    ab = pd.read_csv(AC_BLIND); ab["Molecule_Name"] = ab["Molecule_Name"].astype(str)
    ab = ab.set_index("Molecule_Name")
    missing = set(ids) - set(ab.index)
    if missing:
        raise SystemExit(f"aux blind missing {len(missing)} ids")
    ab = ab.loc[ids]
    direct_blind = ab[f"{E}_pic50_direct_inhibition_pred"].to_numpy(float)
    Xb = (build_features(ab, direct_blind) - mu) / sd
    corr = Xb @ w

    base_blind = pd.to_numeric(r17[TARGET], errors="raise").to_numpy(float)
    new = base_blind + corr
    print(f"\n  BLIND correction: mean={corr.mean():+.4f} sd={corr.std():.4f} "
          f"range=[{corr.min():+.3f},{corr.max():+.3f}]")
    print(f"  blind vs OOF correction mean gap: {corr.mean()-oof_corr.mean():+.4f} "
          f"(a large gap would mean this acts as a mean SHIFT, which "
          f"openadmet-blind-shifts-are-correct forbids)")
    print(f"  CYP3A4 column: mean {base_blind.mean():.4f} -> {new.mean():.4f}, "
          f"sd {base_blind.std(ddof=1):.4f} -> {new.std(ddof=1):.4f}")

    out = r17.copy()
    out[TARGET] = new
    for col in ALL:
        if col == TARGET:
            continue
        if not np.array_equal(out[col].to_numpy(), r17[col].to_numpy()):
            raise SystemExit(f"{col} changed; must stay byte-identical to R17")
    if len(out) != 750:
        raise SystemExit(f"row count {len(out)}")
    if not np.isfinite(new).all():
        raise SystemExit("non-finite corrected predictions")

    path = HERE / "OpenADMET_R22_A4_AUXHEAD_HINGE_20260912.csv"
    out.to_csv(path, index=False)
    manifest = {
        "artifact_type": "isolated_single_column_bet",
        "status": "built_not_submitted",
        "base": {"path": str(R17), "sha256": sha(R17)},
        "changed_endpoints": [TARGET],
        "unchanged_endpoints": [c for c in ALL if c != TARGET],
        "method": "hinge(metric)-fitted linear correction on 4 discarded aux heads",
        "features": AUX,
        "coefficients": dict(zip(AUX, [float(x) for x in w])),
        "l2": L2,
        "scaler": {"mean": [float(x) for x in mu], "sd": [float(x) for x in sd],
                    "note": "fitted on OOF, applied unchanged to blind"},
        "n_oof_fit_rows": int(len(d)),
        "blind_correction": {"mean": float(corr.mean()), "sd": float(corr.std()),
                              "min": float(corr.min()), "max": float(corr.max())},
        "column_stats": {"mean_before": float(base_blind.mean()), "mean_after": float(new.mean()),
                          "sd_before": float(base_blind.std(ddof=1)), "sd_after": float(new.std(ddof=1))},
        "evidence": "analysis/auxhead_hinge_fit_20260912/ -- gain +0.0071 at NN>=0.55 (n=430), "
                     "permutation p=0.000 (0/40 nulls reached it), bootstrap CI lower bound -0.0012 (gate not cleared)",
        "output": {"path": str(path), "sha256": sha(path), "rows": int(len(out))},
    }
    json.dump(manifest, open(HERE / "manifest.json", "w"), indent=1)
    print(f"\nwrote {path}\n  sha256 {manifest['output']['sha256']}")


if __name__ == "__main__":
    main()
