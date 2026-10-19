"""Shared featurization + data loading for the ST-RAE decision-layer experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)
_MORGAN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024, countSimulation=False)
_DESC = [(n, f) for n, f in Descriptors.descList]


def load_challenge() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load_dataset("openadmet/cyp-challenge-train-test")
    return d["train"].to_pandas(), d["test"].to_pandas()


def scaffold_of(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return smiles


def featurize(smiles_list) -> np.ndarray:
    """ECFP4 (2048) + ECFP6 (1024) + RDKit 2D descriptors."""
    rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append(np.zeros(2048 + 1024 + len(_DESC), dtype=np.float32))
            continue
        fp1 = np.frombuffer(_MORGAN.GetFingerprintAsNumPy(m).tobytes(), dtype=np.uint8).astype(np.float32)
        fp2 = np.frombuffer(_MORGAN3.GetFingerprintAsNumPy(m).tobytes(), dtype=np.uint8).astype(np.float32)
        desc = np.empty(len(_DESC), dtype=np.float32)
        for i, (_, fn) in enumerate(_DESC):
            try:
                v = fn(m)
            except Exception:
                v = 0.0
            desc[i] = v if np.isfinite(v) else 0.0
        rows.append(np.concatenate([fp1, fp2, desc]))
    X = np.vstack(rows)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def st_rae(y_true, y_pred, lo, hi) -> float:
    """Exact port of the official rae_soft_threshold_absolute_error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    soft = np.clip(y_pred - hi, 0, None) + np.clip(lo - y_pred, 0, None)
    mt = np.mean(y_true)
    base = np.clip(mt - hi, 0, None) + np.clip(lo - mt, 0, None)
    return float(np.sum(soft) / np.sum(base))
