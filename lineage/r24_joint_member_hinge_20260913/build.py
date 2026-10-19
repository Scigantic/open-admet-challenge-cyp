"""R24 = R23, with CYP1A2/CYP2C9/CYP2D6 replaced by a JOINT hinge-fitted linear
combination over all 12 member predictions. CYP3A4 kept from R23 (its +0.0161
beats the member fit's +0.0098 there).

Evidence: analysis/hinge_stack_all_members_20260913/
  positive at ALL THREE strata on ALL FOUR endpoints -- the only intervention
  this session to manage that. Permutation null (25 shuffles, identical nested
  procedure, features shuffled across compounds): CYP3A4 @NN>=0.55 observed
  +0.0096 vs null max +0.0022 (p=0.000); CYP1A2 @NN>=0.45 observed +0.0127 vs
  null max +0.0075 (p=0.000).

Why it beats equal weighting (which LOST): a joint hinge fit can assign NEGATIVE
weights so members cancel each other's errors, and it optimises the metric
directly instead of MSE.

RISK FLAGGED: CYP2D6 shows the largest gain (+0.046) on the LEAST reliable
validator (live MAE understated 72%), and CYP2D6 offline gains have inverted
live TWICE (R10 +0.057 panel -> -0.0141; R14 +0.0392 significant -> -0.0324).
Per-endpoint board reporting makes every column independently attributable and
revertible, so including it costs no information -- but it is the column most
likely to fail.
"""
from __future__ import annotations
import glob, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
R23 = Path("analysis/r23_a4_combined_20260912/OpenADMET_R23_A4_COMBINED_20260912.csv")
RES = "analysis/chemeleon_multitask_20260910/results"; MEM = "analysis/briford_recipe_parity_20260907/member_results"
OOF_SRC = {"armA":f"{RES}/20260910-2221/oof_A_f*.csv","armB":f"{RES}/20260910-2221/oof_B_f*.csv",
 "armC":f"{RES}/20260910-2221/oof_C_f*.csv","armD":f"{RES}/20260910-2221/oof_D_f*.csv",
 "ivB":"analysis/chemeleon_multitask_20260910/oof_ivB2048.csv",
 "ivC":"analysis/chemeleon_multitask_20260910/oof_ivC2048.csv",
 "mu2048":f"{RES}/20260911-mu2048/oof_B_f*.csv","ii2048":f"{RES}/20260911-ii2048/oof_B_f*.csv",
 "mt_aux8":f"{MEM}/mt_aux8/oof_predictions.csv","union18":f"{MEM}/union18/oof_predictions.csv",
 "aux_ac50":"analysis/aux_ac50_weighted_20260910/oof_predictions.csv",
 "cp_interval":"analysis/chemprop_interval_20260910/oof_predictions.csv"}
BLIND_SRC = {"armA":f"{RES}/20260910-2221/blind_A_f*.csv","armB":f"{RES}/20260910-2221/blind_B_f*.csv",
 "armC":f"{RES}/20260910-2221/blind_C_f*.csv","armD":f"{RES}/20260910-2221/blind_D_f*.csv",
 "ivB":f"{RES}/20260911-ivB2048/blind_B_f*.csv","ivC":f"{RES}/20260911-ivC2048/blind_C_f*.csv",
 "mu2048":f"{RES}/20260911-mu2048/blind_B_f*.csv","ii2048":f"{RES}/20260911-ii2048/blind_B_f*.csv",
 "mt_aux8":f"{MEM}/mt_aux8/blind_predictions_mean.csv","union18":f"{MEM}/union18/blind_predictions_mean.csv",
 "aux_ac50":"analysis/aux_ac50_weighted_20260910/blind_predictions_mean.csv",
 "cp_interval":"analysis/chemprop_interval_20260910/blind_predictions_mean.csv"}
L2 = {"CYP1A2":1.0, "CYP2C9":0.1, "CYP2D6":0.1}      # modal nested inner-fold choices
ALL = [f"{e}_pIC50_direct_inhibition" for e in ("CYP1A2","CYP2C9","CYP2D6","CYP3A4")]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hs = lambda p,lo,hi: np.sum(np.clip(lo-p,0,None)+np.clip(p-hi,0,None))

def grab(srcs, ep, names):
    e=ep.lower(); out={}
    for k,pat in srcs.items():
        fs=sorted(glob.glob(pat))
        if not fs: continue
        d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
        d["Molecule_Name"]=d["Molecule_Name"].astype(str)
        c=f"{e}_pic50_direct_inhibition_pred"
        if c not in d.columns: c=f"{e}_pic50_direct_inhibition"
        if c not in d.columns: continue
        out[k]=d.groupby("Molecule_Name")[c].mean().reindex(names)
    return out

r23=pd.read_csv(R23); ids=r23["Molecule_Name"].astype(str).tolist()
out=r23.copy(); manifest_cols={}
for ep in ("CYP1A2","CYP2C9","CYP2D6"):
    col=f"{ep}_pIC50_direct_inhibition"
    o=pd.read_csv(f"analysis/r08_local_baseline_20260908/r08_like_oof_{ep}.csv")
    o["Molecule_Name"]=o["Molecule_Name"].astype(str)
    o=o.dropna(subset=["label","conf_low","conf_high","r08_like"]).reset_index(drop=True)
    M=grab(OOF_SRC, ep, o["Molecule_Name"])
    mk=[k for k,v in M.items() if v.notna().all()]
    keep=pd.DataFrame({k:M[k] for k in mk}).reset_index(drop=True)
    ok=keep.notna().all(axis=1).to_numpy()
    o=o[ok].reset_index(drop=True); keep=keep[ok].reset_index(drop=True)
    base=o["r08_like"].to_numpy()
    X=np.column_stack([keep[k].to_numpy(float)-base for k in mk])
    lo=o.conf_low.to_numpy(); hi=o.conf_high.to_numpy()
    w=minimize(lambda v: hs(base+X@v,lo,hi)/len(base)+L2[ep]*np.sum(v**2),
               np.zeros(X.shape[1]), method="Powell",
               options={"maxiter":60000,"xtol":1e-6,"ftol":1e-8}).x
    B=grab(BLIND_SRC, ep, pd.Index(ids))
    missing=[k for k in mk if k not in B or B[k].isna().any()]
    assert not missing, f"{ep}: blind predictions missing for {missing}"
    base_b=pd.to_numeric(r23[col],errors="raise").to_numpy(float)
    Xb=np.column_stack([B[k].to_numpy(float)-base_b for k in mk])
    raw=base_b+Xb@w
    # RESTORE the incumbent mean and sd. Rank-preserving, so it keeps the
    # reordering (which is where the gain lives, verified on OOF) and discards
    # the shift + compression, which reproduced R15/R16's rejected CYP2D6 mean
    # shift and R14's spread compression that lost 0.0324 live.
    new=(raw-raw.mean())/max(raw.std(),1e-9)*base_b.std()+base_b.mean()
    assert np.isfinite(new).all()
    assert abs(new.mean()-base_b.mean())<1e-9 and abs(new.std(ddof=1)/base_b.std(ddof=1)-1)<1e-9, \
        "mean/sd restoration failed"
    from scipy.stats import spearmanr as _sp
    print(f"   rank corr vs raw fit: {_sp(new,raw).correlation:.6f} (must be 1.0 -- restoration is monotone)")
    print(f"{ep}: {len(mk)} members, L2={L2[ep]}, fit on {len(o)} rows")
    print(f"   weights: {dict(zip(mk,[round(float(x),3) for x in w]))}")
    print(f"   column mean {base_b.mean():.4f} -> {new.mean():.4f}   "
          f"sd {base_b.std(ddof=1):.4f} -> {new.std(ddof=1):.4f}")
    out[col]=new
    manifest_cols[ep]={"members":mk,"weights":{k:float(x) for k,x in zip(mk,w)},"l2":L2[ep],
                        "n_fit":int(len(o)),"mean_before":float(base_b.mean()),
                        "mean_after":float(new.mean()),"sd_before":float(base_b.std(ddof=1)),
                        "sd_after":float(new.std(ddof=1))}

a4=f"CYP3A4_pIC50_direct_inhibition"
assert np.array_equal(out[a4].to_numpy(), r23[a4].to_numpy()), "CYP3A4 must stay identical to R23"
assert len(out)==750 and out[ALL].notna().all().all()
path=HERE/"OpenADMET_R24_JOINT_MEMBER_HINGE_RESTORED_20260913.csv"
out.to_csv(path,index=False)
json.dump({"artifact_type":"multi_column_joint_hinge_fit_mean_sd_restored","status":"built_not_submitted",
  "base":{"path":str(R23),"sha256":sha(R23)},
  "changed_endpoints":["CYP1A2","CYP2C9","CYP2D6"],"unchanged":["CYP3A4 (kept from R23)"],
  "per_endpoint":manifest_cols,
  "evidence":"analysis/hinge_stack_all_members_20260913/ -- positive at 3/3 strata on 4/4 endpoints; "
             "permutation p=0.000 on CYP3A4@0.55 and CYP1A2@0.45",
  "risk":"CYP2D6 is the largest gain on the least reliable validator and has inverted live twice",
  "output":{"path":str(path),"sha256":sha(path),"rows":int(len(out))}},
  open(HERE/"manifest.json","w"),indent=1)
print(f"\nwrote {path}\n  sha256 {sha(path)}")
