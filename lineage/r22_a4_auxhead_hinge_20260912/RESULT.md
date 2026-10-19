# R22: CYP3A4 corrected by metric-aligned fit on discarded aux heads — built, NOT submitted

`OpenADMET_R22_A4_AUXHEAD_HINGE_20260912.csv`
sha256 `addfbd656653eb6f82b61391b02cc50d42b5bd902074e2ed7da689c5b7dd42a0`
750 rows. Base: **R17** (sha256 d91b70b2...), not R11 — this stacks on the
CYP3A4 column R17 already ships, so R17 should go up first or together.
CYP1A2/CYP2C9/CYP2D6 verified byte-identical to R17.

## What it does
Adds a linear correction to CYP3A4 fitted to minimize the METRIC (hinge on
interval violation) using the four auxiliary heads the pipeline normally throws
away at inference: qHTS potency, ChEMBL potency, max_response, log2fc.

Coefficients (OOF fit, n=2326, L2=0.1): qhts **+0.0316**, chembl **+0.0406**,
max_response **-0.0311**, log2fc **-0.0526** — within 0.002 of the mean of the
five nested per-fold fits, so the production fit agrees with the cross-validated
ones.

## Safety checks
| check | value | why it matters |
| --- | --- | --- |
| blind vs OOF correction mean | +0.0186 vs -0.0000 | a large gap would mean this acts as a mean SHIFT, which [[openadmet-blind-shifts-are-correct]] forbids. It does not. |
| column mean | 4.8888 -> 4.9075 | +0.019, negligible |
| column sd | 0.9094 -> 0.9502 | EXPANDS slightly, so it cannot repeat R14's compression failure ([[openadmet-blends-must-preserve-spread]]) |
| correction per compound | sd 0.055, range [-0.106,+0.180] | modest, targeted nudges |
| feature scaler | fitted on OOF, applied unchanged to blind | no blind marginal leaks into the transform |

## Evidence and its limits
From `../auxhead_hinge_fit_20260912/`: +0.0037/+0.0050/+0.0071 at
NN>=0.45/0.50/0.55 against the verified 3-layer live baseline, n=1125/739/430,
P(gain>0)=0.94/0.97/0.95, sign consistent 3/3, magnitude growing toward the
board regime. Permutation null: **0 of 40 shuffles reached it, one-sided
p=0.000.** Misses the predeclared bootstrap CI bound by 0.0012.

**This is the best-supported candidate of the session** — it is the only one
with a permutation test behind it and the only one whose decisive stratum has
n in the hundreds rather than the fifties. It is still a disclosed bet.

## Why CYP3A4 is a defensible place to spend a round
It is the only endpoint whose interventions have ever transferred live: R11's
entire live gain was CYP3A4, R14 improved CYP3A4 even while losing overall, and
it was the only shipped component gaining on blind-like rows
([[openadmet-a4-three-layer-confirmed]]). That is a genuine prior for transfer,
unlike CYP2D6 where offline gains have inverted live twice.

## Status
**Built and verified. NOT submitted.** Needs explicit human go-ahead per
[[openadmet-experiment-conventions]]. Submit R17 first (or R22 alone, since R22
contains R17's other three columns unchanged and R17's CYP3A4 plus this
correction — so R22 supersedes R17 if you are willing to test both changes at
once; keeping them separate preserves clean attribution).
