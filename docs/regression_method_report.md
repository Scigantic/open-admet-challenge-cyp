# How R25 was built

The regression entry for the OpenADMET CYP inhibition blind challenge predicts pIC50
against CYP1A2, CYP2C9, CYP2D6 and CYP3A4 for 750 blind compounds. This document is the
construction: the training tables, the twelve models behind it, the loss function that
made the difference, and the decision layer that combines them into one CSV.

Everything here runs from this repository. `scripts/build_r25.py` rebuilds the uploaded
file from frozen member predictions in about three seconds with no training and no GPU,
and checks the result against the submitted bytes.

Every code block below is from the file it is attributed to, abridged where a line does
not carry the idea. The authoritative versions are `scripts/build_r25.py`,
`training/featurize.py` and `training/chemeleon_arms/sagemaker_runner.py`.

## 1. Shape of the thing

R25 is not one model. It is the last link in a chain where each step changes specific
columns and leaves the others byte-identical:

```
R17  ->  R23            ->  R24                      ->  R25
         CYP3A4 only         CYP1A2, CYP2C9, CYP2D6       CYP1A2 only
```

Each step is fitted out of fold, applied to the blind set, and written out. Columns are
treated independently because the metric scores four independent columns and averages
them, so a change at one endpoint cannot help or hurt another.

Underneath the chain sit twelve frozen models, the "members". Every step is a small
fitted combination of member predictions on top of the previous column.

## 2. Training data

`data/training/chemeleon_22/train_table.csv` holds 31,669 compounds and 22 target
columns. Only four of those columns are scored. The other eighteen are auxiliary tasks
that exist to regularize the shared representation:

| columns | source | labels |
|---|---|---|
| `cyp*_pic50_direct_inhibition` | organizers | 6,525 |
| `cyp*_log2fc` | organizers, single-concentration screen | 17,500 |
| `cyp*_pic50_chembl`, five isoforms | ChEMBL IC50, median per connectivity key | 56,705 |
| `cyp*_max_response`, five isoforms | PubChem AID 1851 (Veith 2009 qHTS) | 80,665 |
| `cyp*_pic50_qhts` | PubChem AIDs 410, 883, 884, 891, median per InChIKey | 15,070 |

The scored labels are the smallest block in the table by an order of magnitude. That is
the point: 6,525 potency labels will not train a 2048-wide message passing network on
their own, and the auxiliary heads carry the representation.

Alongside it, `bounds.csv` holds the credible interval per scored label, `weights.npy`
the inverse-variance sample weights derived from those intervals, and `folds.npy` the
scaffold fold of each row.

### Leak control

Public rows were joined to the challenge on the **first block of the InChIKey**, so
salts, stereochemistry and charge are ignored and a held-out compound cannot re-enter
the training set under a different identifier. The scaffold used for folding is Murcko,
without chirality:

```python
def scaffold_of(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return smiles
```

One blind compound, OCNT-0496981, does appear in the union table under its PubChem
identifier, with AID 1851 maximum-response values only and no potency label. It was kept
for parity with the public recipe it reproduces and is recorded in
`training/workbench_recipe/sealed_manifest.json`.

## 3. The twelve members

Two backbones, both directed message passing networks over the molecular graph via
Chemprop, all trained as multi-task regressors over the 22-column table, five scaffold
folds each. 364 SageMaker training jobs were run in total; these twelve earned a place.

```python
ARMS = {
    "A": dict(d_h=700,  depth=5, ffn=300,  dropout=0.1, init="random"),
    "B": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="random"),
    "C": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="chemeleon"),
    "D": dict(d_h=2048, depth=4, ffn=1024, dropout=0.2, init="organizer"),
}
```

Arm C initializes message passing from the CheMeleon foundation checkpoint, arm D from
the organizers' CYP-finetuned CheMeleon. The remaining members vary three switches over
arm B and C: the objective (`mae` or `interval`), the sample weighting (`invvar` or
`uniform`), and the training table.

| member | arm | objective | weights | table |
|---|---|---|---|---|
| armA, armB, armC, armD | A, B, C, D | mae | invvar | chemeleon_22 |
| ivB2048, ivC2048 | B, C | interval | uniform | chemeleon_22 |
| mu2048 | B | mae | uniform | chemeleon_22 |
| ii2048 | B | interval | invvar | chemeleon_22 |
| g26_tdi_armB | B | mae | invvar | chemeleon_26_tdi |
| union18, mt_aux8 | Workbench recipe | | | union18_full, mt_aux8_full |
| aux_ac50, cp_interval | Chemprop | mae, interval | invvar | chemprop_22 |

`union18` and `mt_aux8` are recipe-level reproductions of a public SuperCowPowers
Workbench member, pinned against hashes in
`training/workbench_recipe/FROZEN_REPRODUCIBILITY_CONTRACT.json`. `g26_tdi_armB` adds
four TDI-condition pIC50 heads and the compounds that only appear in the organizers' TDI
file. The full command line for each member is in `training/README.md`.

## 4. The interval objective

This is the one modeling idea in the project that paid for itself, and it is worth
reading closely.

The metric does not reward hitting the label. It rewards landing anywhere inside the
label's credible interval, and charges distance to the nearer edge otherwise. Training
on absolute error against a point label therefore spends capacity on precision the
metric does not score.

The obstacle is mechanical. Chemprop's loss sees predictions and targets, not a third
and fourth array of bounds. The trick is to **ride the bounds in as extra target columns
at task weight zero**, so the loss can read them while no head is trained to predict
them:

```python
if OBJECTIVE == "interval":
    SC_IDX = [TARGETS.index(c) for c in SC4]
    extra = []
    for c in SC4:                       # all four lower bounds, then all four upper
        e = c.split("_")[0].upper()
        extra.append(bounds[f"{e}_lo"].to_numpy(float))
    for c in SC4:
        e = c.split("_")[0].upper()
        extra.append(bounds[f"{e}_hi"].to_numpy(float))
    Y = np.hstack([Y, np.column_stack(extra)])
    LO_IDX = list(range(NT, NT + len(SC4)))
    HI_IDX = list(range(NT + len(SC4), NT + 2 * len(SC4)))
```

The loss then replaces absolute error with a two-sided hinge on the scored columns only,
and leaves every auxiliary head on plain absolute error. Chemprop trains in scaled
space, so the hinge has to be computed in real pIC50 units and scaled back, otherwise
each endpoint's interval would be measured in the wrong units:

```python
class IntervalMAE(ChempropMetric):
    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        if mask is not None:                     # never train on the bound columns
            mask = mask.clone()
            mask[:, self.n_real:] = False
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)

    def _calc_unreduced_loss(self, preds, targets, *args):
        L = (preds - targets).abs()
        cols = [L[:, i] for i in range(L.shape[1])]
        for i, li, hj in zip(self.sc_idx, self.lo_idx, self.hi_idx):
            p  = preds[:, i]   * self.sd[i]  + self.mu[i]       # unscale to pIC50
            lo = targets[:, li] * self.sd[li] + self.mu[li]
            hi = targets[:, hj] * self.sd[hj] + self.mu[hj]
            cols[i] = (torch.relu(lo - p) + torch.relu(p - hi)) / self.sd[i]
        return torch.stack(cols, dim=1)
```

Masking the bound columns in `update` is load bearing. Without it the model grows eight
heads that learn to predict interval edges, which wastes capacity and changes the
gradient balance across tasks.

## 5. The decision layer

Four small functions do all of the combining. Together with the metric port they are the
whole vocabulary of the build:

```python
hinge = lambda p, lo, hi: np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None)

def st_rae(y_true, y_pred, lo, hi):
    """Port of the organizers' rae_soft_threshold_absolute_error."""
    base = hinge(np.full_like(y_true, y_true.mean()), lo, hi)
    return float(hinge(y_pred, lo, hi).sum() / base.sum())

def mm(b, c, w):
    """Mean-matched blend: shift c onto b's mean, then mix."""
    return (1 - w) * b + w * (c + (b.mean() - c.mean()))

def restore(raw, ref):
    """Rank-preserving: keep the reordering, put the mean and sd back to ref's."""
    return (raw - raw.mean()) / max(raw.std(), 1e-9) * ref.std() + ref.mean()
```

`mm` and `restore` exist for the same reason. A blend of two models pulls the column
toward the average of two means and shrinks its spread, and on this metric a compressed
column is punished: predictions collapse toward the center and stop landing inside the
intervals out in the tails. `mm` removes the location part before mixing, and `restore`
puts both moments back afterward, so a fitted step is allowed to change the **ranking**
of compounds but not the shape of the distribution.

Note what the ST-RAE denominator is: the same hinge applied to a constant predictor at
the mean of the labels. It depends only on the labels, not on the model, so it is a
constant per endpoint. Minimizing ST-RAE is therefore exactly minimizing the summed
hinge, which is what every fit below actually optimizes.

## 6. The build, step by step

### R23, the CYP3A4 column

Three nested mean-matched blends build a base, a weight for the interval-trained member
is chosen on a coarse grid, and then a four-term linear correction on auxiliary heads is
fitted with Powell under an L2 penalty:

```python
base = mm(mm(mm(o.candidate.to_numpy(), o.mt.to_numpy(), 0.18), direct, 0.20),
          o.mt.to_numpy(), 0.20)

best_w  = min(W_GRID, key=lambda w: hinge(mm(base, o.iv.to_numpy(), w), lo, hi).sum())
stage_b = mm(base, o.iv.to_numpy(), best_w)

AUX = [f"{e}_pic50_qhts_pred", f"{e}_pic50_chembl_pred",
       f"{e}_max_response_pred", f"{e}_log2fc_pred"]
Xraw = np.nan_to_num(np.column_stack(
    [(A_[c].to_numpy(float) - direct) if "pic50" in c else A_[c].to_numpy(float)
     for c in AUX]), nan=0.0)
X = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-9)

w_a4 = minimize(lambda v: hinge(stage_b + X @ v, lo, hi).sum() / len(base)
                          + L2_A4 * np.sum(v ** 2),
                np.zeros(4), method="Powell",
                options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-8}).x
```

The auxiliary potency heads enter as **differences from the direct head**, not as raw
values. A qHTS AC50 prediction is on its own scale with its own bias, so the informative
quantity is the disagreement between the two heads, which is a signal about the
compound, while the raw value is mostly a signal about the assay.

Powell is used because the objective is piecewise linear with flat regions, so it has no
useful gradient. Powell is derivative-free and does not care.

### R24, the other three columns

The same pattern, scaled up: every member that has a complete prediction for the
endpoint enters as a difference from the incumbent, one coefficient each, fitted on the
hinge with a per-endpoint L2, then moments restored.

```python
L2 = {"CYP1A2": 1.0, "CYP2C9": 0.1, "CYP2D6": 0.1}   # modal nested inner-fold choices

for E in ("CYP1A2", "CYP2C9", "CYP2D6"):
    o    = oof_base(E)
    M    = {m: member_pred(m, "oof", E, o.Molecule_Name) for m in MEMBERS}
    mk   = [m for m in MEMBERS if M[m].notna().all()]   # complete members only
    keep = pd.DataFrame({m: M[m] for m in mk}).reset_index(drop=True)
    base = o.r08_like.to_numpy()
    X    = np.column_stack([keep[m].to_numpy(float) - base for m in mk])
    lo, hi = o.conf_low.to_numpy(), o.conf_high.to_numpy()

    w = minimize(lambda v: hinge(base + X @ v, lo, hi).sum() / len(base)
                           + L2[E] * np.sum(v ** 2),
                 np.zeros(X.shape[1]), method="Powell",
                 options={"maxiter": 60000, "xtol": 1e-6, "ftol": 1e-8}).x

    # apply to the blind set, then put the first two moments back
    Xb = np.column_stack([B[m].to_numpy(float) - base_b for m in mk])
    R24[COL[E]] = restore(base_b + Xb @ w, base_b)
```

The L2 values are not tuned on the fit itself. They are the modal choice across nested
inner folds, which is why CYP1A2 carries ten times the penalty of the other two: its fit
was the least stable across folds.

### R25, one blend on CYP1A2

```python
E = "CYP1A2"; W_G26 = 0.20
g = member_pred("g26_tdi_armB", "blind", E, IDS).to_numpy(float)
b = R24[COL[E]].to_numpy(float)
R25 = R24.copy()
R25[COL[E]] = restore(mm(b, g, W_G26), b)
for E2 in ("CYP2C9", "CYP2D6", "CYP3A4"):
    assert np.array_equal(R25[COL[E2]].to_numpy(), R24[COL[E2]].to_numpy())
```

A fifth of the TDI-augmented member, mean-matched in and moments restored, with an
assertion that the other three columns did not move. That assertion is the habit worth
copying: every step in this chain ends by proving it changed only what it meant to.

## 7. Reproducing it

```
pip install -r requirements.txt
python scripts/build_r25.py
```

The script verifies the sha256 of each stored submission against
`data/MANIFEST.json` before using it, rebuilds R23, R24 and R25 in order, and prints the
largest disagreement against the submitted file at every step along with both hashes.
`notebooks/reproduce_r25.ipynb` is the same code with commentary, committed with the
outputs of a run that matched the bytes.

It round-trips each intermediate through CSV on purpose:

```python
R23.to_csv(BUILD / "R23_rebuilt.csv", index=False)
R23 = pd.read_csv(BUILD / "R23_rebuilt.csv")
```

because the original R24 build read R23 back from disk, and sixteen decimal digits
survive that trip while a float in memory does not necessarily match one that has been
printed and parsed. Skipping it changes the last bits.

### Whether the bytes match depends on the BLAS

| environment | largest disagreement | bytes |
|---|---|---|
| macOS arm64, Accelerate, numpy 2.5.2, scipy 1.18.1 | 8.9e-16 | identical |
| macOS arm64, Accelerate, numpy 2.0.2, scipy 1.14.1 | 8.9e-16 | identical |
| linux arm64, OpenBLAS, numpy 2.0.2, scipy 1.14.1 | 5.2e-06 | differ |
| linux x86_64, OpenBLAS, numpy 2.0.2, scipy 1.14.1 | 3.4e-06 | differ |

Powell is derivative-free: it stops when a line search stops paying, and that point
moves with the order floating-point sums accumulate. A weight landing on 0.0383 instead
of 0.0384 shifts 750 predictions by a millionth of a log unit, which changes every byte
after the sixth decimal. The scored quantities do not move; the fitted weights agree to
three decimals and the out-of-fold ST-RAE of every step is identical to four decimals
across all four rows. The script fails only above 1e-4, which would mean a real bug:

```python
assert maxdiff < 1e-4, (
    f"the rebuild disagrees with the submitted file by {maxdiff:.1e} pIC50, "
    "which is too large to be floating-point drift")
```

## 8. Two things we would do differently

**Fit the decision layer on something closer to the test distribution.** We measured the
nearest-neighbor Tanimoto from the blind set to the training set at 0.587, against 0.439
for a random fold and 0.425 for a scaffold fold. The blind compounds are *closer* to the
training data than our own folds are to each other, so scaffold splitting made our
harness harder than the real task and every fitted step was tuned under a pessimism that
does not apply.

**Calibrate location against the public score early.** The build above produces
well-ranked columns, and the single largest improvement we ever measured came later and
from outside it: a uniform additive offset per endpoint, worth more than any change in
sections 4 to 6. Out of fold the optimal offset is zero at every endpoint, because
cross-validation holds the train-to-test gap constant and a location shift between them
is exactly what it cannot see. The ranking work in this document is what the model is
for; the offset is a free parameter the leaderboard will fit for you if you spend one
submission asking.
