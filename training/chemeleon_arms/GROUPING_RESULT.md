# Task grouping: G4 is far worse than G22. The 18 auxiliary heads are load-bearing.

2026-09-11, stamp 20260911-0735grp, arm B (d_h 2048, depth 4), batch 64, MAE objective,
inverse-variance weights.

## Motivation being tested
The runner's own comment: "d6_isoform6 (6 isoforms) beat both single-task (0.8980) and
22-target (0.9009) on CYP2D6, so the sweet spot may be a MIDDLE grouping." G4 is the
scored-four-only end of that sweep.

## Per-fold ST-RAE, G4

fold   CYP1A2   CYP2C9   CYP2D6   CYP3A4     mean  epochs
0      0.9511   0.9766   0.9898   1.0922   1.0024     100
1      1.1141   0.9129   1.0595   1.1450   1.0579      37
2      1.0339   0.8309   0.9300   0.9948   0.9474      80
3      1.4529   1.5425   1.3946   1.1151   1.3763       4

Pooled 4-fold: CYP1A2 0.9518, CYP2C9 0.8932, CYP2D6 1.0277, CYP3A4 0.8460 (mean 0.9297).
Arm B at G22, same width: 0.8247 / 0.5814 / 0.8727 / 0.5220 (mean 0.7002).

**Dropping the 18 auxiliary heads costs about 0.23-0.30 macro.** Not a sweet spot; a
cliff. The multitask trunk is doing heavy lifting, consistent with
`trunk_grouping_mlp_20260910` finding that three fluorescent heads beat four
single-task models by +0.0409, and with CYP2D6 benefiting from a trunk it correlates
with at 0.005.

## Fold 3 diverged, and the manifest alone would not have told you
`epochs_run` 4 with `best_ckpt` at `epoch=0`, under patience 30 and max_epochs 200.
Patience cannot fire at epoch 4; Lightning's EarlyStopping has `check_finite=True` and
aborts when the monitored value is non-finite, so val_loss went NaN.

Its predictions are NOT visibly degenerate -- mean 5.326, std 0.393, against 5.02-5.12
and 0.297-0.340 for the healthy folds -- so a spread check would have passed it. Only
the per-fold score exposes it (1.3763 against 0.947-1.058). **Score folds individually
before pooling; a diverged fold hides inside a pooled average.**

Training instability at G4 and not at G22 is itself weak evidence for the same
conclusion: with only four heads and inverse-variance weights, the loss is less well
conditioned.

## Collector quirk worth knowing
`--folds` on collect_and_gate.py controls only which artifacts are DOWNLOADED. The gate
then globs every `oof_{arm}_f*.csv` present in the results dir, so restricting `--folds`
does not restrict the assembly, and re-running with a subset silently reproduces the
full-set number. To exclude a fold you must remove its file.

## Status
G8 had only one fold placed before capacity was taken by higher-value work. Given the
size of the G4 cliff, the remaining question is whether G12 or G8 recovers, not whether
a small grouping wins. Low priority.
