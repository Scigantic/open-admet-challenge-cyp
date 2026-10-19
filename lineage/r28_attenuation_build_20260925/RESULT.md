# R28: R27 with the attenuation correction. BUILT, NOT SUBMITTED.

Built 2026-09-25 by `build_r28.py`, a copy of `r27_fleet_build_20260919/build.py` with one
addition. **Nothing uploaded. No existing submission CSV was read, built or modified: all 95
were sha256-checked before and after and are byte-identical** (`baseline_csv_hashes.txt`).
No network, no organizer contact, no leaderboard, no GPU. Runtime 695s.

## Artifact

`OpenADMET_R28_ATTEN_C9D6A4_20260925.csv`
sha256 `4f8cf28fe47392ea24611ea1155dc673b75d6b38e5615de466f7a187daed3a9b`
750 rows, zero nulls, column order and molecule order identical to R25.

## What changed from R27, and only that

`build.py` fits the 21 combiner weights once on all OOF rows and applies `w` to the blind
rows. R28 selects one scalar `c` per endpoint **on the OOF rows only**, by 5-fold CV over the
same weight-fitting rule, and applies **`c * w`**. Nothing else differs.

**Fidelity check: the 21 fitted weights are bit-identical to R27's at 4/4 endpoints**
(max|dw| = 0.000e+00, same modal L2, same `sum_w`). The pipeline reproduced exactly, so the
only difference in the output is the correction.

| ep | c | column | vs R27 | vs R25 mean abs delta | OOF gain from c |
| --- | --- | --- | --- | --- | --- |
| CYP1A2 | 2.5 | **held at R25** | **identical to R27** | 0.0000 | not shipped |
| CYP2C9 | 2.5 | R27+c | mean 0.0334, max 0.1473 | 0.0580 | **+0.00685** |
| CYP2D6 | 1.2 | R27+c | mean 0.0285, max 0.1947 | 0.1146 | +0.00105 |
| CYP3A4 | 1.1 | R27+c | mean 0.0066, max 0.0285 | 0.0721 | +0.00017 |

**Macro from the correction on the three shipped columns: +0.00202.** Mean and sd are pinned
to R25's exactly at 4/4, as `build.py` asserts, so this is not a moment change.

CYP1A2 keeps R25's column, inherited from R27. That is the right call independently of
R27's original stated reason (which rested on a `verify.py` anchoring defect): the real
justification is the live record, where a CYP1A2 calibration replacement shipped on an
offline +0.00929 and **lost +0.0327** (`a1_monotone_shape_20260919`). So the aggressive
c=2.5 selected at CYP1A2 never reaches the file.

## The honest caveat: CYP3A4 does not reproduce between frames

The correction was measured in `weight_attenuation_20260923` by pseudo-blind fold rotation,
which is the shipped frame. Comparing what each frame chose and got:

| ep | rotation c (per fold) | rotation gain | full-OOF c | full-OOF gain | agrees |
| --- | --- | --- | --- | --- | --- |
| CYP2C9 | 2.0 in 4/5 folds | +0.00502 | 2.5 | +0.00685 | yes, one grid step further |
| CYP2D6 | 1.1 in 3/5 folds | +0.00015 | 1.2 | +0.00105 | yes |
| CYP3A4 | **1.75 in 4/5 folds** | **+0.00285** | **1.1** | **+0.00017** | **NO** |

At CYP3A4 the two frames disagree on both the optimum and the size, and the full-OOF curve
rises sharply above 1.2 (1.75 scores 0.35645 vs 1.1's 0.35062). **So the +0.00285 measured at
CYP3A4 in the rotation is not dependable, and R28 banks almost nothing there (+0.00017).**
The correction that survives both frames is CYP2C9's, which is also the largest.

The c curve shape differs by endpoint and this matters for how much to trust each:
CYP2D6 and CYP3A4 have sharp interior optima (well determined); CYP2C9 and CYP1A2 are broad
and flat from about 2.0 to 3.0, so their exact c is not identified, only its direction.

## Expected value, stated conservatively

R27 itself is worth **+0.00959** macro in the shipped frame (`xtr_label_transport_20260923`,
FLEET over INC, retention 0.540 against a nested +0.01776). The correction adds **+0.00202**
on the OOF measure and **+0.00201** if the rotation's per-column gains are used for the three
shipped columns instead. Together roughly **+0.0116**, against a board paired sd of 0.0104
and a 0.0253 gap to first. That is about 1.1 board sd and slightly under half the gap.

## Provenance
`build_r28.py` (diffable against `../r27_fleet_build_20260919/build.py`), `build.log`,
`manifest.json` (records `c_selected`, `c_oof_gain`, the full `c_grid_scores` curve and the
21 weights per endpoint), `baseline_csv_hashes.txt`, `RESULT.md`.
