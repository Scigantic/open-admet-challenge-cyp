# R23: CYP3A4 via two additive metric-aligned interventions — GATES PASS, built, NOT submitted

`OpenADMET_R23_A4_COMBINED_20260912.csv`
sha256 `1893ed9d0349e1d6264d373a6cdeea32396a767773933856075f61782c8ebcf3`
750 rows. Base **R17**; CYP1A2/CYP2C9/CYP2D6 byte-identical to it.
**Supersedes R22** (same idea, one component short).

## Two interventions, nearly orthogonal, exactly additive
| | NN>=0.45 | NN>=0.50 | NN>=0.55 |
| --- | --- | --- | --- |
| A: hinge-fitted aux-head correction | +0.0048 | +0.0065 | +0.0091 |
| B: blend ivB2048 (interval objective, UNIFORM weights) @ w=0.2 | +0.0023 | +0.0045 | +0.0071 |
| A + B arithmetic | +0.0071 | +0.0110 | +0.0163 |
| **C: both, measured** | **+0.0075** | **+0.0110** | **+0.0161** |

Per-compound delta correlation between A and B is **+0.080** — they are almost
independent, which is why they add. A harvests information the model never
exposes at inference; B changes the loss the model trains under. Both are the
same principle (align to the metric) applied at different layers.

## All three predeclared gates pass — first time this session
| gate | value | verdict |
| --- | --- | --- |
| magnitude >= +0.005 at NN>=0.55 | +0.0161 (3x the bar) | PASS |
| sign consistent >= 2 of 3 strata | 3 of 3 | PASS |
| bootstrap 95% CI lower bound > 0 | +0.0003 / +0.0023 / +0.0023 | PASS at ALL strata |
| P(gain>0) | 0.978 / 0.993 / 0.986 | — |

The aux-head component is separately permutation-validated (0 of 40 shuffles
reached it, one-sided p=0.000,
`../auxhead_hinge_fit_20260912/permutation_null.py`). The interval-model
component is a single weight chosen out-of-fold from a 6-point grid, so it has
almost no capacity to overfit, and its standalone value was independently
measured in `../iv2048_stack_test_20260912/` before this experiment existed.

## Use the RIGHT cell of the 2x2
ivB2048 is interval objective + **UNIFORM** sample weights, verified from its
own manifest. The 2x2 in `../chemeleon_multitask_20260910/INTERVAL_2X2_RESULT.md`
shows interval + inverse-variance is the WORST of the four cells (interaction
-0.0510). Production uses inverse-variance, so the artifact choice here is
load-bearing: `ii2048` is the wrong cell and must not be substituted.

## Safety
mean 4.8888 -> 4.9080 (+0.019); sd 0.9094 -> **0.9232** (expands, so it cannot
repeat R14's compression failure); blind aux-correction mean +0.0192 against OOF
-0.0000, so it is not a disguised distribution shift; feature scaler and blend
weight both fitted on OOF and applied unchanged to blind.

## Expected leaderboard effect, stated soberly
A CYP3A4-only column gain of D moves macro by D/4. At 1:1 transfer:
macro 0.4620 (R17) -> **0.4580**, i.e. -0.0045 against R11's 0.4625. That is
about a third of the -0.0124 needed to reach 11th, and roughly 10% of the
-0.0443 gap to first. It is the largest gate-passing single-column gain staged
so far and still does not change our rank on its own.

CYP3A4's only two measured offline->live transfer ratios are 0.24x (R14) and
2.9x (R10), so the honest range is -0.001 to -0.013 macro.

## Status
**Built and verified. NOT submitted.** Needs explicit human go-ahead per
[[openadmet-experiment-conventions]]. R23 contains R17 entirely, so submitting
R23 tests both changes at once; submitting R17 first preserves clean attribution.
