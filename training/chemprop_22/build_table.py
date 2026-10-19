"""union18 heads plus four qHTS AC50 auxiliary heads."""
import pandas as pd, numpy as np
from pathlib import Path
OUT=Path("analysis/aux_ac50_member_20260909")
u=pd.read_csv('analysis/parity_reset_v2_20260907/training_inputs/union18_full.csv')
ik=pd.read_csv(OUT/'union_inchikeys.csv')
u=u.merge(ik,on="Molecule_Name",how="left")
c2k=pd.read_csv('analysis/qhts_ac50_20260909/cid_to_inchikey.csv'); c2k.columns=[c.upper() for c in c2k.columns]
smi=dict(zip(u.InChIKey,u.smiles))
new_rows={}
for ep,aid in (("CYP1A2",410),("CYP2C9",883),("CYP3A4",884),("CYP2D6",891)):
    a=pd.read_csv(f'analysis/qhts_ac50_20260909/{ep}_ac50.csv')
    j=a.merge(c2k[["CID","INCHIKEY","CANONICALSMILES"]] if "CANONICALSMILES" in c2k.columns
              else c2k[["CID","INCHIKEY"]], on="CID").dropna(subset=["INCHIKEY"])
    g=j.groupby("INCHIKEY").pq.median()
    col=f"{ep.lower()}_pic50_qhts"
    u[col]=u.InChIKey.map(g)
    for k,v in g.items():
        if k not in smi: new_rows.setdefault(k,{})[col]=v
    print(f"{ep}: attached to {u[col].notna().sum()} existing union rows")
print(f"compounds only in qHTS (dropped, no SMILES resolved locally): {len(new_rows)}")
targets=[c for c in u.columns if c not in ("Molecule_Name","smiles","InChIKey")]
u=u.dropna(subset=["smiles"]).reset_index(drop=True)
u.to_csv(OUT/"train_table.csv",index=False)
print(f"\ntable {u.shape}, {len(targets)} target columns")
print("target coverage:")
for c in targets:
    v=pd.to_numeric(u[c],errors='coerce')
    tag=" <-- NEW" if c.endswith("_qhts") else ""
    print(f"   {c:42} {v.notna().sum():>6}{tag}")
