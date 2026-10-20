"""Rebuild the R25 submission from the frozen member predictions in data/.

Same code as notebooks/reproduce_r25.ipynb, generated from one source."""

import glob, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path.cwd()
while not (ROOT / "data" / "MANIFEST.json").exists():
    ROOT = ROOT.parent
D = ROOT / "data"
EPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
COL = {e: f"{e}_pIC50_direct_inhibition" for e in EPS}
PRED = {e: f"{e.lower()}_pic50_direct_inhibition_pred" for e in EPS}
MANIFEST = json.load(open(D / "MANIFEST.json"))

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hinge = lambda p, lo, hi: np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None)

def st_rae(y_true, y_pred, lo, hi):
    # port of the organizers' rae_soft_threshold_absolute_error
    y_true, y_pred, lo, hi = (np.asarray(a, float) for a in (y_true, y_pred, lo, hi))
    base = hinge(np.full_like(y_true, y_true.mean()), lo, hi)
    return float(hinge(y_pred, lo, hi).sum() / base.sum())

def mm(b, c, w):
    # mean-matched blend: shift c onto b's mean, then mix
    return (1 - w) * b + w * (c + (b.mean() - c.mean()))

def restore(raw, ref):
    # rank-preserving: keep the reordering, put the mean and sd back to ref's
    return (raw - raw.mean()) / max(raw.std(), 1e-9) * ref.std() + ref.mean()

print("repo root:", ROOT)

MEMBERS = ["armA", "armB", "armC", "armD", "ivB2048", "ivC2048", "mu2048", "ii2048",
           "mt_aux8", "union18", "aux_ac50", "cp_interval"]

def member(name, kind):
    d = pd.read_csv(D / "members" / name / f"{kind}.csv")
    d["Molecule_Name"] = d["Molecule_Name"].astype(str)
    return d

def member_pred(name, kind, ep, names):
    # per-compound mean over folds of the endpoint's prediction column
    d = member(name, kind)
    c = PRED[ep]
    if c not in d.columns:
        c = c[:-5]  # cp_interval wrote its columns without the _pred suffix
    return d.groupby("Molecule_Name")[c].mean().reindex(names)

def oof_base(ep):
    o = pd.read_csv(D / "oof_base" / f"r08_like_oof_{ep}.csv")
    o["Molecule_Name"] = o["Molecule_Name"].astype(str)
    return o.dropna(subset=["label", "conf_low", "conf_high", "r08_like"]).reset_index(drop=True)

sub = {k: pd.read_csv(D / "submissions" / f"{k}.csv") for k in ("R17", "R23", "R24", "R25")}
for k, v in sub.items():
    v["Molecule_Name"] = v["Molecule_Name"].astype(str)
    assert sha(D / "submissions" / f"{k}.csv") == MANIFEST["submissions"][k]["sha256"], k
IDS = sub["R17"]["Molecule_Name"].tolist()
assert len(IDS) == 750

rows = {e: len(oof_base(e)) for e in EPS}
print("OOF rows per endpoint:", rows)
print("blind compounds:", len(IDS))

score = {}
for e in EPS:
    o = oof_base(e)
    score[(e, "incumbent")] = st_rae(o.label, o.r08_like, o.conf_low, o.conf_high)
print(pd.Series({e: score[(e, "incumbent")] for e in EPS}, name="OOF ST-RAE, incumbent").round(4))

E = "CYP3A4"; e = E.lower()
AUX = [f"{e}_pic50_qhts_pred", f"{e}_pic50_chembl_pred", f"{e}_max_response_pred", f"{e}_log2fc_pred"]
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; L2_A4 = 0.1

o = oof_base(E).dropna(subset=["candidate"]).reset_index(drop=True)
ac = member("aux_ac50", "oof").set_index("Molecule_Name")
o = o[o.Molecule_Name.isin(ac.index)].reset_index(drop=True)
o["mt"] = o.Molecule_Name.map(member("mt_aux8", "oof").groupby("Molecule_Name")[PRED[E]].mean())
o["iv"] = o.Molecule_Name.map(member("ivB2048", "oof").groupby("Molecule_Name")[PRED[E]].mean())
o = o.dropna(subset=["mt", "iv"]).reset_index(drop=True)
A_ = ac.loc[o.Molecule_Name]
direct = A_[PRED[E]].to_numpy(float)
base = mm(mm(mm(o.candidate.to_numpy(), o.mt.to_numpy(), 0.18), direct, 0.20), o.mt.to_numpy(), 0.20)
lo, hi = o.conf_low.to_numpy(), o.conf_high.to_numpy()

best_w = min(W_GRID, key=lambda w: hinge(mm(base, o.iv.to_numpy(), w), lo, hi).sum())
stage_b = mm(base, o.iv.to_numpy(), best_w)

Xraw = np.nan_to_num(np.column_stack(
    [(A_[c].to_numpy(float) - direct) if "pic50" in c else A_[c].to_numpy(float) for c in AUX]), nan=0.0)
mu, sd = Xraw.mean(0), Xraw.std(0) + 1e-9
X = (Xraw - mu) / sd
w_a4 = minimize(lambda v: hinge(stage_b + X @ v, lo, hi).sum() / len(base) + L2_A4 * np.sum(v ** 2),
                np.zeros(4), method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-8}).x

score[(E, "R23")] = st_rae(o.label, stage_b + X @ w_a4, lo, hi)
score[(E, "R23 base")] = st_rae(o.label, base, lo, hi)
print(f"interval-member blend weight: {best_w}")
print("aux coefficients:", {c: round(float(x), 4) for c, x in zip(AUX, w_a4)})
print(f"OOF ST-RAE CYP3A4: base {score[(E,'R23 base')]:.4f} -> R23 {score[(E,'R23')]:.4f}  (n={len(o)})")

# ---- apply to the blind set
ab = member("aux_ac50", "blind").set_index("Molecule_Name").loc[IDS]
iv_blind = member_pred("ivB2048", "blind", E, IDS).to_numpy(float)
direct_b = ab[PRED[E]].to_numpy(float)
base_b = sub["R17"][COL[E]].to_numpy(float)
Xb = (np.nan_to_num(np.column_stack(
    [(ab[c].to_numpy(float) - direct_b) if "pic50" in c else ab[c].to_numpy(float) for c in AUX]), nan=0.0) - mu) / sd
a4_new = mm(base_b, iv_blind, best_w) + Xb @ w_a4

R23 = sub["R17"].copy(); R23[COL[E]] = a4_new
diff = np.abs(R23[COL[E]].to_numpy() - sub["R23"][COL[E]].to_numpy()).max()
print(f"max |rebuilt - submitted R23| on CYP3A4: {diff:.2e}")
# the original R24 build read R23 back from its CSV, so round-trip through CSV here too
BUILD = ROOT / "notebooks" / "_build"; BUILD.mkdir(exist_ok=True)
R23.to_csv(BUILD / "R23_rebuilt.csv", index=False)
print("R23 byte-identical to submitted:", sha(BUILD / "R23_rebuilt.csv") == MANIFEST["submissions"]["R23"]["sha256"])
R23 = pd.read_csv(BUILD / "R23_rebuilt.csv"); R23["Molecule_Name"] = R23["Molecule_Name"].astype(str)

L2 = {"CYP1A2": 1.0, "CYP2C9": 0.1, "CYP2D6": 0.1}   # modal nested inner-fold choices
R24 = R23.copy(); weights = {}
for E in ("CYP1A2", "CYP2C9", "CYP2D6"):
    o = oof_base(E)
    M = {m: member_pred(m, "oof", E, o.Molecule_Name) for m in MEMBERS}
    mk = [m for m in MEMBERS if M[m].notna().all()]
    keep = pd.DataFrame({m: M[m] for m in mk}).reset_index(drop=True)
    ok = keep.notna().all(axis=1).to_numpy()
    o = o[ok].reset_index(drop=True); keep = keep[ok].reset_index(drop=True)
    base = o.r08_like.to_numpy()
    X = np.column_stack([keep[m].to_numpy(float) - base for m in mk])
    lo, hi = o.conf_low.to_numpy(), o.conf_high.to_numpy()
    w = minimize(lambda v: hinge(base + X @ v, lo, hi).sum() / len(base) + L2[E] * np.sum(v ** 2),
                 np.zeros(X.shape[1]), method="Powell",
                 options={"maxiter": 60000, "xtol": 1e-6, "ftol": 1e-8}).x
    weights[E] = dict(zip(mk, w))
    score[(E, "R24")] = st_rae(o.label, base + X @ w, lo, hi)

    B = {m: member_pred(m, "blind", E, pd.Index(IDS)) for m in mk}
    assert all(B[m].notna().all() for m in mk), E
    base_b = R23[COL[E]].to_numpy(float)
    Xb = np.column_stack([B[m].to_numpy(float) - base_b for m in mk])
    R24[COL[E]] = restore(base_b + Xb @ w, base_b)
    diff = np.abs(R24[COL[E]].to_numpy() - sub["R24"][COL[E]].to_numpy()).max()
    print(f"{E}: {len(mk)} members, fit on {len(o)} rows, OOF {score[(E,'incumbent')]:.4f} -> {score[(E,'R24')]:.4f}; "
          f"max |rebuilt - submitted R24| = {diff:.2e}")

R24.to_csv(BUILD / "R24_rebuilt.csv", index=False)
print("R24 byte-identical to submitted:", sha(BUILD / "R24_rebuilt.csv") == MANIFEST["submissions"]["R24"]["sha256"])
R24 = pd.read_csv(BUILD / "R24_rebuilt.csv"); R24["Molecule_Name"] = R24["Molecule_Name"].astype(str)
print(pd.DataFrame(weights).round(3))

E = "CYP1A2"; W_G26 = 0.20
g = member_pred("g26_tdi_armB", "blind", E, IDS).to_numpy(float)
b = R24[COL[E]].to_numpy(float)
R25 = R24.copy(); R25[COL[E]] = restore(mm(b, g, W_G26), b)
for E2 in ("CYP2C9", "CYP2D6", "CYP3A4"):
    assert np.array_equal(R25[COL[E2]].to_numpy(), R24[COL[E2]].to_numpy())
diff = np.abs(R25[COL[E]].to_numpy() - sub["R25"][COL[E]].to_numpy()).max()
print(f"max |rebuilt - submitted R25| on CYP1A2: {diff:.2e}")

p = BUILD / "R25_rebuilt.csv"
R25.to_csv(p, index=False)
submitted = MANIFEST["submissions"]["R25"]["sha256"]
rebuilt = sha(p)
cols = [COL[e] for e in EPS]
maxdiff = float(np.abs(R25[cols].to_numpy() - sub["R25"][cols].to_numpy()).max())
print(f"submitted sha256 {submitted}")
print(f"rebuilt   sha256 {rebuilt}")
print(f"byte-identical: {rebuilt == submitted}")
print(f"max |rebuilt - submitted| over all four columns: {maxdiff:.2e}")
if rebuilt != submitted:
    print("The predictions still agree to the tolerance below. Powell is derivative-free,\n"
          "so where it stops depends on the order floating-point sums accumulate, which the\n"
          "BLAS numpy is linked against decides. README.md has the measured values.")
assert maxdiff < 1e-4, (
    f"the rebuild disagrees with the submitted file by {maxdiff:.1e} pIC50, which is too "
    "large to be floating-point drift")

live = {"R17": {"CYP1A2": 0.4909, "CYP2C9": 0.3862, "CYP2D6": 0.5694, "CYP3A4": 0.4014, "macro": 0.4620},
        "R24": {"CYP1A2": 0.4806, "CYP2C9": 0.3802, "CYP2D6": 0.5442, "CYP3A4": 0.3762, "macro": 0.4453},
        "R25": {"CYP1A2": 0.4739, "CYP2C9": 0.3802, "CYP2D6": 0.5442, "CYP3A4": 0.3762, "macro": 0.4436}}
tab = pd.DataFrame({
    "OOF incumbent": {e: score[(e, "incumbent")] for e in EPS},
    "OOF after step": {"CYP1A2": score[("CYP1A2", "R24")], "CYP2C9": score[("CYP2C9", "R24")],
                       "CYP2D6": score[("CYP2D6", "R24")], "CYP3A4": score[("CYP3A4", "R23")]},
    "live R17": {e: live["R17"][e] for e in EPS},
    "live R25": {e: live["R25"][e] for e in EPS},
}).round(4).astype(object)
# the OOF columns have no macro: each endpoint was fitted on a different row set
tab.loc["macro"] = ["", "", live["R17"]["macro"], live["R25"]["macro"]]
print(tab)
