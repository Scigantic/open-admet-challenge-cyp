"""Reconstruct an R08-like out-of-fold baseline for all four endpoints.

The blocker exposed by the 2026-09-08 live result is that this project has no local
reconstruction of its own incumbent. Every local gate ran against a weak comparator,
so the close-analog panel kept approving candidates the leaderboard then rejected.
The blend was gated against `old_equal`, beat it by +0.074, and lost to R08 by
0.187 live. The panel was probably right; the question was wrong.

R08's placement recipe is recorded in
`analysis/r08_workbench_a1_20260907/build_candidate.py`:

    per fold:  extra = interval + (candidate.mean() - interval.mean())
               r08_like = (1 - w) * candidate + w * extra        with w = 0.40

Both `candidate` and `interval` exist for all four endpoints in the base OOF table,
so the same operation reconstructs the incumbent everywhere, not just CYP1A2.

Validation: the CYP1A2 output must reproduce the existing `r08_like` column in
`a1_nested_oof.csv` to floating-point tolerance. That column was built by the code
this script mirrors, so any mismatch means the reconstruction is wrong and the run
fails closed.

Output is an OOF baseline per endpoint plus its score on the close-analog family
panel, which is the reference number every future candidate must beat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "metric_decision_layer_20260907"))
from featurize import ENDPOINTS, load_challenge, st_rae  # noqa: E402
from family_panel import build_families, fps  # noqa: E402

BASE_OOF = REPO / "analysis/classification_push_20260903/tdi_margin_live_incumbent_blend_oof.csv"
A1_REF = REPO / "analysis/r08_workbench_a1_20260907/a1_nested_oof.csv"
OUT = Path(__file__).resolve().parent
# R08 applies interval heads ONLY to CYP1A2 and CYP2C9, weights 0.4 and 0.3.
# CYP2D6 and CYP3A4 receive no interval blend. Source:
# analysis/winning_system_20260904/INTERVAL_A1_C9_CANDIDATE_MANIFEST.json
R08_WEIGHTS = {"CYP1A2": 0.40, "CYP2C9": 0.30, "CYP2D6": 0.0, "CYP3A4": 0.0}
R08_INTERVAL_WEIGHT = 0.40  # retained only for the CYP1A2 reproduction check
SEED = 0
N_BOOT = 10000
SHORT = {e: e.split("_")[0] for e in ENDPOINTS}


def apply_operation(frame, current, component, weight):
    """Fold-local blend, copied from build_candidate.apply_operation, method 'direct'."""
    out = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        sel = frame["fold"].eq(fold).to_numpy()
        base = current[sel]
        extra = component[sel].copy()
        extra += float(base.mean() - extra.mean())
        out[sel] = (1.0 - weight) * base + weight * extra
    return out


def main():
    base = pd.read_csv(BASE_OOF)
    tr, _ = load_challenge()
    names = tr["Molecule_Name"].astype(str).to_numpy()
    t = tr.set_index(names)
    fp_all = fps(tr["SMILES"].tolist())

    frames, report = {}, {"weight": R08_INTERVAL_WEIGHT, "endpoints": {}}

    for ep in ENDPOINTS:
        short = SHORT[ep]
        f = base.query("endpoint == @short")[
            ["Molecule_Name", "scaffold", "fold", "label", "conf_high", "conf_low",
             "candidate", "interval"]].dropna().reset_index(drop=True)
        w = R08_WEIGHTS[short]
        f["r08_like"] = (f["candidate"].to_numpy(float) if w == 0.0 else
                         apply_operation(f, f["candidate"].to_numpy(float),
                                         f["interval"].to_numpy(float), w))
        f.index = f["Molecule_Name"].astype(str)
        frames[ep] = f
        print(f"{short}: {len(f)} rows, interval weight {R08_WEIGHTS[short]}", flush=True)

    # ---- fail-closed validation against the one existing reference column
    ref = pd.read_csv(A1_REF)
    ref.index = ref["Molecule_Name"].astype(str)
    mine = frames["CYP1A2_pIC50_direct_inhibition"]
    common = sorted(set(ref.index) & set(mine.index))
    a = ref.loc[common, "r08_like"].to_numpy(float)
    b = mine.loc[common, "r08_like"].to_numpy(float)
    max_abs = float(np.max(np.abs(a - b)))
    if not np.allclose(a, b, rtol=1e-9, atol=1e-9):
        raise SystemExit(f"CYP1A2 reconstruction does not match the reference: "
                         f"max abs diff {max_abs}")
    print(f"validation: CYP1A2 matches the existing r08_like column on {len(common)} "
          f"rows, max abs diff {max_abs:.2e}")
    report["validation"] = {"rows": len(common), "max_abs_diff": max_abs, "passed": True}

    # ---- score the incumbent on the close-analog family panel
    rng = np.random.default_rng(SEED)
    macro = []
    for ep in ENDPOINTS:
        f = frames[ep]
        m = (tr[ep].notna() & tr[ep + "_conf_low"].notna()
             & tr[ep + "_conf_high"].notna()).to_numpy()
        idx = np.where(m)[0]
        parents, kids = build_families(tr[ep].to_numpy(float)[idx],
                                       [fp_all[i] for i in idx],
                                       np.random.default_rng(SEED))
        rows, fam = [], []
        for q, p in enumerate(parents):
            pn = names[idx[p]]
            if pn not in f.index:
                continue
            for c in kids[p]:
                cn = names[idx[c]]
                if cn not in f.index:
                    continue
                if f.loc[pn, "fold"] == f.loc[cn, "fold"]:
                    continue
                rows.append(cn); fam.append(q)
        if len(rows) < 15:
            report["endpoints"][ep] = {"panel_children": len(rows), "note": "too few"}
            continue
        fam = np.array(fam)
        y = t.loc[rows, ep].to_numpy(float)
        lo = t.loc[rows, ep + "_conf_low"].to_numpy(float)
        hi = t.loc[rows, ep + "_conf_high"].to_numpy(float)
        p08 = f.loc[rows, "r08_like"].to_numpy(float)
        cand = f.loc[rows, "candidate"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(p08)
        s = st_rae(y[ok], p08[ok], lo[ok], hi[ok])
        report["endpoints"][ep] = {
            "oof_rows": int(len(f)),
            "panel_children": int(ok.sum()),
            "panel_families": int(len(set(fam[ok]))),
            "r08_like_panel_st_rae": s,
            "pre_blend_candidate_panel_st_rae": st_rae(y[ok], cand[ok], lo[ok], hi[ok]),
            "mean_pred": float(np.mean(p08[ok])),
            "mean_truth": float(np.mean(y[ok])),
            "frac_below_4": float(np.mean(p08[ok] < 4.0)),
        }
        macro.append(s)
        r = report["endpoints"][ep]
        print(f"  {SHORT[ep]} panel: n={r['panel_children']} fam={r['panel_families']}  "
              f"r08_like {s:.4f}  (pre-blend candidate "
              f"{r['pre_blend_candidate_panel_st_rae']:.4f})  "
              f"mean_pred {r['mean_pred']:.3f} vs truth {r['mean_truth']:.3f}", flush=True)

        f[["Molecule_Name", "fold", "label", "conf_low", "conf_high",
           "candidate", "interval", "r08_like"]].to_csv(
            OUT / f"r08_like_oof_{SHORT[ep]}.csv", index=False)

    report["macro_r08_like_panel_st_rae"] = float(np.mean(macro)) if macro else None
    report["live_reference"] = {
        "R08_macro": 0.473426, "R08_by_endpoint":
        {"CYP1A2": 0.490918, "CYP2C9": 0.413525, "CYP2D6": 0.569353, "CYP3A4": 0.419906},
        "blend_macro": 0.6601, "A1_candidate_macro": 0.475,
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nmacro R08-like on the close-analog panel: "
          f"{report['macro_r08_like_panel_st_rae']:.4f}")
    print("this is the number every future candidate must beat locally")


if __name__ == "__main__":
    main()
