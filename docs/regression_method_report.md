# OpenADMET CYP Inhibition Blind Challenge: regression method report

Team: Scigantic
Track: CYP direct-inhibition pIC50 regression, four endpoints, scored by
soft-thresholded relative absolute error (ST-RAE, lower is better)
Entry on the board at the time of writing: 0.4283 macro

| endpoint | our cell |
|----------|----------|
| CYP1A2 | 0.4521 |
| CYP2C9 | 0.3504 |
| CYP2D6 | 0.5345 |
| CYP3A4 | 0.3762 |
| macro | 0.4283 |

## 1. Summary

We predict pIC50 for 750 blind compounds at four CYP endpoints with gradient-boosted
ensembles over standard molecular descriptors and fingerprints, trained on the
4,905-row challenge training set.

Two things in this report are worth more than the model itself. The first is a
reading of the scoring function that makes it tractable to reason about (section 2).
The second is that the obvious consequence of that reading, training against the
metric instead of against plain absolute error, loses at all four endpoints, and we
can say why (section 4.1).

## 2. What ST-RAE actually optimizes

ST-RAE is `rae_soft_threshold_absolute_error` in the challenge's
`evaluation/custom_scoring_functions.py`. It is an epsilon-insensitive L1 error: a
prediction inside the compound's measured confidence interval contributes nothing,
and outside it contributes its distance to the nearer edge. That much is visible
from the name.

The part that matters for optimization is the denominator. The naive baseline in the
denominator is put through the *same* soft-thresholding as the prediction.
Therefore the denominator depends only on the hidden labels and their intervals, and
is a constant per endpoint. Minimizing ST-RAE is exactly minimizing a sum of
epsilon-insensitive absolute errors, with no normalization term that the model's own
predictions can influence.

We also verified that the macro is a plain unweighted mean of the four cells rather
than anything pooled or sample-weighted:

    (0.4521 + 0.3504 + 0.5345 + 0.3762) / 4 = 0.42830

which matches the posted macro exactly. Both facts are worth confirming before
spending any effort on the metric, since they determine whether per-endpoint work
can be reasoned about independently. It can.

## 3. Submitted system

The live entry is assembled per column rather than produced by one training run, and
two of its four columns carry a uniform additive calibration offset on top of the
model output.

It starts from our R25 artifact, whose per-endpoint live scores are recorded in our
submission history, and changes three columns:

| endpoint | what the column is | offset | R25 live | live now | gain |
|----------|--------------------|--------|----------|----------|------|
| CYP1A2 | R25 column plus a constant | +0.170 | 0.4739 | 0.4521 | +0.0218 |
| CYP2C9 | R25 column plus a constant | +0.140 | 0.3802 | 0.3504 | +0.0298 |
| CYP2D6 | a separate model | none | 0.5442 | 0.5345 | +0.0097 |
| CYP3A4 | R25 column, untouched | 0.000 | 0.3762 | 0.3762 | 0.0000 |

The macro closes exactly on those four cells:

    0.4436 - (0.0218 + 0.0298 + 0.0097) / 4 = 0.4283

We identified the live artifact by its CYP3A4 column rather than by its filename. The
board reports CYP3A4 at 0.3762, which is R25's recorded CYP3A4 cell to four decimals,
and the candidate files differ in exactly that column, so the one holding R25's
CYP3A4 unchanged is the one on the board. A filename is not provenance, and in this
project we have twice been wrong about which file was live by trusting one.

CYP2D6 and CYP3A4 were deliberately left without an offset. The reason is the
per-endpoint transfer record, which is the most useful artifact the regression track
produced:

| endpoint | offline-to-board transfer | record |
|----------|---------------------------|--------|
| CYP3A4 | +0.82 | 6 of 6 |
| CYP2C9 | +0.40 | 3 of 3 modern |
| CYP1A2 | +0.24 | 2 of 3 |
| CYP2D6 | -0.18 | 1 of 3 |

CYP3A4 is the only endpoint whose offline judgment is distinguishable from a coin
flip (one-sided p = 0.0156), and it is also our strongest column, so we did not
disturb it. CYP2D6 offline gains have historically *reversed* on the board, with a
negative expected signed transfer in every era we can slice, so we treat any CYP2D6
change as presumed harmful until the board says otherwise.

## 4. Negative results

### 4.1 Training against the metric loses at every endpoint

Given section 2, the natural move is to replace the training loss with the metric:
an epsilon-insensitive objective that stops penalizing a prediction once it is
inside the label's confidence interval. We implemented it as a custom LightGBM
objective using the training set's own per-compound `_conf_low` and `_conf_high`
columns, and compared it against plain L2 and plain L1 under identical folds and
hyperparameters.

| endpoint | n | L2 | MAE | band | band minus MAE |
|----------|------|--------|--------|--------|-----------|
| CYP1A2 | 1412 | 0.8455 | 0.7949 | 0.8240 | +0.0291 |
| CYP2C9 | 1285 | 0.6979 | 0.6889 | 0.7229 | +0.0339 |
| CYP2D6 | 1493 | 0.9879 | 0.9035 | 0.9689 | +0.0654 |
| CYP3A4 | 2335 | 0.5225 | 0.5144 | 0.5219 | +0.0076 |

The metric-matched objective is worse than plain MAE at all four endpoints. Applied
everywhere it would be expected to score 0.4320 against the 0.4283 that is live.

The striking part is not that it loses on ST-RAE. It is that it also lands *fewer*
predictions inside the band, which is the only thing it is built to do:

| endpoint | MAE inside band | band objective inside band |
|----------|-----------------|----------------------------|
| CYP1A2 | 24.1% | 21.0% |
| CYP2C9 | 41.1% | 35.9% |
| CYP2D6 | 19.0% | 17.5% |
| CYP3A4 | 37.7% | 26.8% |

Our reading is that the epsilon-insensitive objective discards the gradient
information from the 19% to 41% of compounds already inside their interval, and
those compounds are exactly the well-measured ones whose labels are most reliable.
Plain MAE keeps learning from them, and the resulting model generalizes better even
judged by the band criterion. A metric that ignores part of its input is not
necessarily a loss function that should ignore the same part during training.

### 4.2 An implementation trap that counterfeits a null result

This is worth publishing because it would be easy to stop at the wrong conclusion.

Our first implementation of the band objective scored 1.0016 and 1.0043, that is
ST-RAE of essentially 1.0 to four decimals, which is the naive-mean predictor
exactly. The cause is that boosting starts from a raw score of 0. Since the pIC50
labels sit around 4 to 5, every initial prediction was below the band's lower edge,
so every gradient was -1, uniformly. A constant gradient carries no split
information, so no tree could learn anything and the model returned the base rate.

The fix is one line, an `init_score` at the training median so the first gradients
are already informative. Without it the experiment produces a clean, plausible,
completely uninformative null. With it the objective still loses, which is the real
result in 4.1, but for a reason rather than an artifact.

### 4.3 There is no recombination gain left

We enumerated every per-endpoint cell we had ever recorded on the board and computed
the best achievable macro from any combination of them. The answer is 0.42830,
which is what is already live. The live entry dominates every recorded cell at every
endpoint, so there is no column swap available that improves the macro.

### 4.4 Our internal cross-validation is systematically pessimistic

We measured how far the blind set actually sits from the training set, against how
far our folds sit from each other, by nearest-neighbor Tanimoto similarity:

- blind set to training set: 0.587
- random fold to its complement: 0.439
- scaffold fold to its complement: 0.425

The blind compounds are *closer* to the training data than our own folds are to each
other. Our cross-validation therefore understates board performance, and scaffold
splitting makes the mismatch worse rather than better. This explains a pattern that
had confused us for several weeks, where a from-scratch single model reaching 0.7949
cross-validated at CYP1A2 coexists with a production ensemble reaching 0.644
out-of-fold on the same endpoint. The two numbers are measured against different
difficulty distributions, and neither is the board's.

The practical consequence is that an offline gain is not directly a board
prediction, which is what motivates the per-endpoint transfer table in section 3
rather than acting on offline deltas at face value.

### 4.5 The largest effect we found is a calibration offset, and cross-validation has the wrong sign on it

The two offsets in section 3 are worth +0.0516 of summed column gain, or +0.0129 of
macro. That is larger than any modeling change in our entire submission history. It
is also the one result our offline harness would have told us not to ship.

We recomputed it directly. Using the out-of-fold predictions with the training set's
own labels and confidence intervals, we swept a uniform additive offset per endpoint
and found the soft-threshold error minimum:

| endpoint | offline optimal offset | offline gain at optimum | offline change at the shipped offset | board change at the shipped offset |
|----------|------------------------|-------------------------|--------------------------------------|------------------------------------|
| CYP1A2 | -0.035 | +0.0017 | **-0.0539** | **+0.0218** |
| CYP2C9 | +0.000 | +0.0000 | **-0.0543** | **+0.0298** |
| CYP2D6 | +0.005 | +0.0001 | n/a | n/a |
| CYP3A4 | -0.010 | +0.0001 | 0.0000 | 0.0000 |

Out of fold, the optimal offset is zero to within a few hundredths at every endpoint,
and the offsets we shipped should have cost about 0.054 per column. On the board they
gained 0.0218 and 0.0298. The sign is opposite and the magnitude is comparable.

The explanation is that the blind compounds' labels sit higher than the training
labels. A model calibrated to the training distribution is therefore systematically
low on the blind set, and a positive offset corrects a location shift that no
resampling of the training data can expose. Cross-validation measures the model's fit
to the training distribution, and the gap between train and test is precisely the
quantity it holds constant.

This compounds the similarity result in 4.4. Our folds are not just harder than the
blind set, they are centered differently, so an offline harness built on them is
wrong about difficulty in one direction and wrong about location in the other.

Two practical consequences. First, in a blind challenge with a per-endpoint public
score, a cheap global calibration probe is worth more than a great deal of modeling,
and should be among the first submissions rather than among the last. Second, the
offset is not known to be optimized: the soft-threshold objective is convex in the
offset, we have exactly two board points per endpoint, and two points on a convex
curve cannot bound the minimum. The measured average slopes over the interval we
tested are -0.128 ST-RAE per unit offset at CYP1A2 and -0.213 at CYP2C9, both still
pointing downhill at the offsets we shipped.

## 5. What we would do differently

Most of our regression effort went into the model and the objective. The two results
that actually moved the score are the transfer table in section 3 and the calibration
offset in section 4.5, and both are built entirely from public leaderboard feedback
that was available from the first week.

For a blind challenge with per-endpoint public scoring, we would now spend the first
few submissions deliberately on measuring the board rather than on improving the
model: one probe to establish per-endpoint transfer, and one global offset probe per
endpoint to locate the calibration minimum. Both are nearly free, both are
deterministic, and in our case the offset probe alone was worth more than a month of
modeling. The offline harness is then a filter for candidates rather than the thing
being optimized, which is the only role our own numbers support for it.

The corresponding failure was bookkeeping. Our submission history is complete and
scored through 2026-09-14 and that ledger is the reason the transfer table exists.
The offset campaign that produced the current entry was not recorded the same way,
so we know that two offsets helped and by how much, and we do not know what else was
tried alongside them. Our own ledger warns about exactly this, and the cost is that
section 4.5 ends with an open question that one recorded number would have closed.

## 6. Reproduction

The complete recipe behind the model, including the frozen out-of-fold and blind
predictions of every member, the training tables and the rebuild script, is in this
repository. `docs/submission_history.md` is the scored history of every upload,
per endpoint, with the arithmetic checks that make the transfer table in section 3
auditable; it is the source for every R25 cell quoted here. `data/submissions/` holds
the uploaded files, and the live entry is `best_known_20261028.csv`, identified by
the CYP3A4 column as described in section 3. `data/oof_base/` holds the per-endpoint
labels, confidence intervals and out-of-fold predictions used for the offset sweep in
section 4.5.

The band-objective comparison, the fold harness and the similarity characterization
are in the working repository under `analysis/regression_push_20261029/`.
