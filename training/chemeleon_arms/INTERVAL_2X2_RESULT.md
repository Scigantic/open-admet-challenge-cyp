# The interval objective DOES have independent value: +0.0275. And a component passes.

2026-09-11. Four cells, arm B (d_h 2048, depth 4, ffn 1024, dropout 0.2, random init),
batch 128, 200 epochs / patience 30, 5 scaffold folds, ALL FROM ONE RUNNER BUILD in one
batch -- which is the whole point, after the 700-width version turned out to be a
diagonal across two code paths.

## The square

                      CYP1A2   CYP2C9   CYP2D6   CYP3A4    MACRO
  mae      + invvar   0.8246   0.5841   0.8739   0.5922   0.7187
  mae      + uniform  0.8604   0.6273   0.8922   0.4883   0.7170
  interval + invvar   0.8053   0.6338   0.8831   0.6466   0.7422
  interval + uniform  0.8304   0.5889   0.8966   0.4421   0.6895

Per-fold means, checked before pooling (no fold deviates by more than 0.085; the flag
threshold is 0.15, set after the G4 fold that stopped at 4 epochs on a NaN val_loss
while still producing healthy-looking predictions):
  mae+invvar    0.700 0.724 0.734 0.740 0.701
  mae+uniform   0.687 0.710 0.772 0.734 0.692
  interval+iv   0.733 0.747 0.762 0.731 0.744
  interval+uni  0.671 0.705 0.700 0.714 0.663

## Single-factor effects (positive = helps)

  objective MAE->interval, at UNIFORM     +0.0275   <- the cell nobody had ever run
  objective MAE->interval, at invvar      -0.0235
  weighting invvar->uniform, at MAE       +0.0017
  weighting invvar->uniform, at interval  +0.0527
  interaction                             -0.0510

## Three conclusions, two of which overturn this morning's

1. **The interval objective has real independent value: +0.0275 at uniform weights.**
   Measured against MAE with everything else identical. This is the first clean
   measurement of it at any width. It helps CYP1A2 (+0.0300), CYP2C9 (+0.0384) and
   CYP3A4 (+0.0462), and is neutral on CYP2D6 (-0.0044), consistent with
   [[openadmet-cyp2d6-is-orthogonal]].

2. **"The weighting switch was doing the work" does NOT replicate at 2048.** At 700
   width the weighting looked like the whole story. Here, weighting alone at MAE is
   +0.0017 -- nothing. The weighting only matters in COMBINATION with the interval
   objective (+0.0527), which is the interaction, not a main effect.

3. **The antagonism is real and large: interaction -0.0510.** The objective helps under
   uniform weights and HURTS under inverse-variance. That reconciles the 700-width
   result: interval+invvar is genuinely the worst cell in both experiments.

## And my +0.0913 was mostly fold noise
The single-fold batch-64 probe put the diagonal at +0.0913. The same diagonal here, at
5 folds and batch 128, is **+0.0292**. So that number was inflated about 3x by
single-fold variance on top of the confound already retracted. One fold is not a
measurement.

## The component gate PASSES -- first time in a while

interval+uniform, arm B, against the incumbent on the blind-matched panel with LFO:

  endpoint    base      own  errcorr  panelGain   oofGain  agree  panelP
  CYP1A2    0.7132   0.8059    0.901   +0.02537  +0.00346    yes  0.1462
  CYP2C9    0.4829   0.5803    0.889   +0.00338  +0.00683    yes  0.3478
  CYP2D6    0.8327   0.8926    0.921   +0.04853  +0.01627    yes  0.0945
  CYP3A4    0.3695   0.4397    0.891   +0.01254  +0.00517    yes  0.0125
  macro: panel +0.02246  OOF +0.00793   4/4 positive on panel, no flips   PASS

Arm C (CheMeleon init) also passes but is weaker: panel +0.01832, OOF +0.00763,
4/4 positive, 0/4 significant.

## What this is NOT
- Not a better standalone model. 0.6895 macro OOF against the incumbent stack's ~0.60.
  Consistent with [[openadmet-backbone-is-an-accumulation]]: members lose, components
  add.
- Not decorrelated. errcorr 0.889-0.921, right in the band where
  [[openadmet-architecture-ceiling]] says everything sits.
- Not significant. 1/4 endpoints at P<0.05. And
  [[openadmet-validators-are-not-predictive]] records offline gains calling 2 of 4 live.

**No submission has been made. This requires explicit human approval.**
