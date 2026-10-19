# RESULT — the per-endpoint track record of offline prediction vs live board outcome

Protocol predeclared in `PROTOCOL.md` before any number was produced.
Machine-readable: `history.json`, `history.csv`. Builder: `build_history.py`.
Every number below carries the path it was quoted from. No submission was built or
modified; no network, no paid job.

---

## VERDICT

**"Offline gains called 2 of 4" is a one-day snapshot, not the track record.** It comes
from `analysis/candidate_projection_20260911/RESULT.md:27-31`, which tabulated exactly
four observations — all from 2026-09-10 (R10's three columns plus the A1GBM window).

The full reconstructable record is **13 of 20 genuine offline directional calls correct**
(65%, Jeffreys 95% CI 0.432-0.828, one-sided p vs a coin flip 0.13 — so pooled, still
not distinguishable from a coin flip). But it is **radically non-uniform across
endpoints**, and that is the finding:

| endpoint | all history | 2026-09-09 onward | mean signed transfer ratio (modern) |
|---|---|---|---|
| **CYP3A4** | **6/6** | **4/4** | **+0.82** |
| **CYP2C9** | **3/4** | **3/3** | **+0.40** |
| CYP1A2 | 2/5 | 2/3 | +0.24 |
| **CYP2D6** | **2/5** | **1/3** | **−0.18** |

CYP3A4's 6/6 is the only per-endpoint result that clears a coin flip on its own
(one-sided p = 0.0156). **CYP2D6's expected signed transfer is negative in every era we
can slice** — the historical expectation of acting on a CYP2D6 offline gain is to make
the column worse. CYP1A2 is 0/3 before 2026-09-13 and 2/2 after.

---

## 1. MASTER TABLE — every scored submission

ST-RAE, lower better. `—` = the cell is recorded nowhere in this repo.
Macro identity (`macro == mean of four columns`, tolerance 2e-4 for 4-dp display
rounding) checked on all 21 rows: **15 of 15 testable rows PASS, 0 FAIL**, 6 not
testable (macro-only era, or a missing cell). Full check log in `history.json` ->
`arithmetic_checks`.

| # | sub | date (UTC) | what changed | cols changed | macro | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | id ✓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | R00 | 08-31 21:52 | tutorial LightGBM + A4 gate (wrong artifact) | all | 0.935796 | 0.861384 | 0.703272 | 1.424846 | 0.753680 | PASS |
| 2 | R01 | 09-01 09:59 | endpoint placement + multitask base | all | 0.522798 | 0.554719 | 0.477426 | 0.604407 | 0.454641 | PASS |
| 3 | R02 | 09-01 22:04 | guarded frontier + CompTox full-CV | all | 0.531319 | 0.507297 | 0.441607 | 0.694932 | 0.481440 | PASS |
| 4 | R03 | 09-02 12:01 | interval/rank D6 + direct interval A4 | D6, A4 | 0.486542 | 0.507297 | 0.442222 | 0.571990 | 0.424660 | PASS |
| 5 | R04 | 09-03 00:40 | CheMeleon family rank edits, new base | all | 0.4933 | — | — | — | — | n/t |
| 6 | R05 | 09-03 | MMELON C9 on R03 lineage | ? | — (rank 13) | — | — | — | — | n/t |
| 7 | R06 | 09-04 18:09 | CYPlebrity A4 rerank, separate base | ? | 0.4789 | — | — | — | — | n/t |
| 8 | R07 | 09-05 11:08 | D6 pairwise-only on R06 | D6 | 0.4898 | — | — | — | — | n/t |
| 9 | R08 | 09-06 00:40 | A1+C9 interval heads on R06 | A1, C9 | 0.473426 | 0.490918 | 0.413525 | 0.569353 | 0.419906 | PASS |
| 10 | R09 | 09-06 12:44 | A1 shape + C9 exact-family + A4 proxy | A1, C9, A4 | 0.488340 | 0.523618 | 0.441093 | 0.569353 | 0.419295 | PASS |
| 11 | A1WB | 09-07 | stock-Workbench A1 blend w=0.4 | A1 | ~0.475 | ~0.4972 | 0.413525 | 0.569353 | 0.419906 | PASS* |
| 12 | BLEND | 09-08 | architecture-diverse blend, raw | all | 0.6601 | — | — | — | — | n/t |
| 13 | C9R | 09-09 ~00:12 | mt_aux8 → CYP2C9 w=0.43 | C9 | 0.4668 | 0.490918 | 0.3870 | 0.569353 | 0.419906 | PASS |
| 14 | A4R | 09-09 12:29 | mt_aux8 → CYP3A4 w=0.18 | A4 | 0.4644 | 0.4909 | 0.3869 | 0.5694 | 0.4103 | PASS |
| 15 | A1GBM | 09-10 00:38 | GBM fingerprint → CYP1A2 w=0.17 | A1 | 0.4676 | 0.5037 | 0.3869 | 0.5694 | 0.4103 | PASS |
| 16 | R10 | 09-10 18:03 | inv-variance member: C9 .25 / D6 .60 / A4 .20 | C9, D6, A4 | 0.4693 | 0.5037 | 0.3862 | 0.5835 | 0.4037 | PASS |
| 17 | R11 | 09-11 06:03 | recombine best-scored columns (arithmetic) | A1, D6 | 0.4625 | 0.4909 | 0.3862 | 0.5694 | 0.4037 | PASS |
| 18 | R14 | 09-11 | mt_aux8 → ALL FOUR, weights nested NN≥0.45 | all | **—** | **—** | **—** | 0.6018† | 0.4014 | n/t |
| 19 | R17 | 09-12 | R11 + R14's CYP3A4 column (arithmetic) | A4 | 0.4620 | 0.4909 | 0.3862 | 0.5694 | 0.4014 | PASS |
| 20 | R24 | 09-13 | joint 12-member hinge on A1/C9/D6 + R23's A4 | all | 0.4453 | 0.4806 | 0.3802 | 0.5442 | 0.3762 | PASS |
| 21 | R25 | 09-14 | G26 (armB+TDI aux heads) → CYP1A2 w=0.2 | A1 | 0.4436 | **0.4739** | 0.3802 | 0.5442 | 0.3762 | PASS |

`*` A1WB's macro and CYP1A2 survive only as "0.475" and "~0.4972" in
`analysis/blend_candidate_20260908/SUBMITTED_PREDECLARED_READING.md:14`; the identity
closes to 4e-06 anyway.
`†` R14's CYP2D6 cell is **DERIVED, not recorded** (0.5694 + 0.0324), per
`analysis/d6_dispersion_20260915/RESULT.md:207-210`.

**Per-column independence tripwire** (exactly one column changed ⇒ 4 × macro delta =
column delta): **3 of 3 testable transitions PASS exactly** —
A4R→A1GBM (A1 −0.0128, 4×macro −0.0128), R11→R17 (A4 −0.0023, 4×macro_exact −0.0023),
R24→R25 (A1 −0.0067, 4×macro_exact −0.0067). The board scores four independent columns
and re-scores an untouched column to the last digit
(`analysis/d6_dispersion_20260915/RESULT.md:246`).

### Built, verified, NEVER SUBMITTED (no live outcome; excluded from every rate)

R12, R13, R15, R16, R18, R19, R20, R21, R22, **R23**, R26 — eleven files.
Source for the never-submitted list: `analysis/d6_dispersion_20260915/live_history.py:57-76`
plus each directory's own RESULT.md status line.

---

## 2. THE PER-ENDPOINT PREDICTION TABLE (the core deliverable)

`gain` is positive-is-better (ST-RAE before − after). `ratio` = live gain / offline gain.
"alt" is the second offline statistic the same document reported, where one exists.

### CYP1A2 — 2 of 5
| sub | date | mechanism | offline gain | live gain | signed err | ratio | sign |
|---|---|---|---|---|---|---|---|
| R09 | 09-06 | shape specialist | +0.0115 | **−0.0327** | −0.0442 | −2.83 | ✗ |
| A1WB | 09-07 | Workbench union18+aux8 avg w=0.4 (nested OOF +0.0234, CI[0.0121,0.0349], P=0.9998) | +0.0234 | **−0.0063** | −0.0297 | −0.27 | ✗ |
| A1GBM | 09-10 | GBM fingerprint w=0.17 (6/6 constructions +, 6/6 folds non-zero) | +0.0113 (alt OOF +0.0024) | **−0.0128** | −0.0241 | −1.13 | ✗ |
| R24 | 09-13 | joint 12-member hinge | +0.0127 (central +0.0083) | **+0.0103** | −0.0024 | **+0.81** | ✓ |
| R25 | 09-14 | G26 blend w=0.2 | +0.0065 | **+0.0067** | +0.0002 | **+1.03** | ✓ |

### CYP2C9 — 3 of 4
| sub | date | mechanism | offline gain | live gain | signed err | ratio | sign |
|---|---|---|---|---|---|---|---|
| R09 | 09-06 | exact-family foundation specialist | +0.4016 (alt ordinary −0.0005) | **−0.0276** | −0.4292 | −0.07 | ✗ (sign-sensitive, see §5.6) |
| C9R | 09-09 | mt_aux8 w=0.43 (panel +0.0579, 12 seeds 0 neg, P=0.9511) | +0.0579 | **+0.0265** | −0.0314 | +0.46 | ✓ |
| R10 | 09-10 | inv-variance member w=0.25 | +0.0092 (alt OOF +0.0087) | **+0.0007** | −0.0085 | +0.08 | ✓ |
| R24 | 09-13 | joint 12-member hinge | +0.0092 (central +0.0060) | **+0.0060** | −0.0032 | **+0.65** | ✓ |

### CYP2D6 — 2 of 5
| sub | date | mechanism | sd/R11 | offline gain | live gain | ratio | sign |
|---|---|---|---|---|---|---|---|
| R03 | 09-02 | rank-only interval blend | 1.040 | +0.1767 | **+0.1229** | +0.70 | ✓ |
| R07 | 09-05 | CheMeleon pairwise family rerank | — | +0.0346 | **−0.0436** | −1.26 | ✗ |
| R10 | 09-10 | inv-variance member w=0.60 | 0.718 | +0.0567 (alt OOF +0.0166) | **−0.0141** | −0.25 | ✗ |
| R14 | 09-11 | mt_aux8 raw blend w=0.45 (offline CI [+0.0047,+0.0742], **SIGNIFICANT**) | 0.758 | +0.0392 | **−0.0324** | −0.83 | ✗ |
| R24 | 09-13 | joint hinge, mean+sd restored exactly | 1.000 | +0.0461 (central +0.0300) | **+0.0252** | **+0.55** | ✓ |

### CYP3A4 — 6 of 6
| sub | date | mechanism | offline gain | live gain | signed err | ratio | sign |
|---|---|---|---|---|---|---|---|
| R03 | 09-02 | direct interval blend | +0.1241 | **+0.0568** | −0.0673 | +0.46 | ✓ |
| R09 | 09-06 | primary-proxy global calibration | +0.0027 (alt +0.0007) | **+0.0006** | −0.0021 | +0.23 | ✓ |
| A4R | 09-09 | mt_aux8 w=0.18 (panel rel. +2.6%) | +0.0109 | **+0.0096** | −0.0013 | +0.88 | ✓ |
| R10 | 09-10 | inv-variance member w=0.20 | +0.0111 (alt OOF +0.0023) | **+0.0066** | −0.0045 | +0.59 (alt 2.91) | ✓ |
| R14 | 09-11 | mt_aux8 raw blend w=0.20 | +0.0094 | **+0.0023** | −0.0071 | +0.24 | ✓ |
| R24 | 09-13 | R23 chain (aux-head hinge + ivB2048 w=0.2) | +0.0161 (central +0.0105) | **+0.0252** | +0.0091 | **+1.57** | ✓ |

### Excluded rows, and why (all in `history.csv`)
- **arithmetic (4)** — R11's four columns and R17's CYP3A4: the expectation was
  recombination of already-scored columns, explicitly "ARITHMETIC ON OBSERVED BOARD
  NUMBERS ... not a model prediction"
  (`analysis/live_verdict_20260910/MANIFEST.md:29-30`). Both hit their predicted macro
  exactly (0.4625, 0.4620) — tripwires, not judgement.
- **unchanged (10)** — byte-identical columns; all returned unchanged as designed.
- **confounded (3)** — R02 macro, R04 macro (causal grade C), and R08's A1+C9 aggregate
  (grade B+, not separable against a non-consecutive R06 baseline). Grades carried from
  `analysis/leaderboard_transfer_meta_20260906/transfer_observations.csv`.
- **nopred (3)** — R14's CYP1A2 and CYP2C9 (the submission itself declared them
  "uncertain in either direction",
  `analysis/scratch_interpolation_20260911/SUBMITTED.md:17`, **and their live cells are
  recorded nowhere**); and the 09-08 BLEND, submitted "as an information purchase at
  neutral expected score" with no directional claim.

---

## 3. PER-ENDPOINT HIT RATES — exact counts and binomial uncertainty

Jeffreys (Beta(0.5,0.5)) equal-tailed 95% interval; `p` is the one-sided binomial
p-value against the coin-flip null p = 0.5.

### All history (2026-08-31 → 2026-09-14)
| endpoint | k/n | rate | Jeffreys 95% | p vs coin | mean signed ratio | sd | range |
|---|---|---|---|---|---|---|---|
| CYP1A2 | **2/5** | 0.40 | 0.094 – 0.791 | 0.81 | **−0.479** | 1.578 | −2.83 … +1.03 |
| CYP2C9 | **3/4** | 0.75 | 0.284 – 0.972 | 0.31 | +0.279 | 0.333 | −0.07 … +0.65 |
| CYP2D6 | **2/5** | 0.40 | 0.094 – 0.791 | 0.81 | **−0.218** | 0.848 | −1.26 … +0.70 |
| CYP3A4 | **6/6** | 1.00 | 0.670 – 1.000 | **0.0156** | **+0.662** | 0.504 | +0.23 … +1.57 |
| pooled | 13/20 | 0.65 | 0.432 – 0.828 | 0.13 | | | |

### Blind-matched-panel era onward (2026-09-09 →), i.e. everything since the validator was rebuilt
| endpoint | k/n | rate | Jeffreys 95% | p vs coin | mean signed ratio | sd |
|---|---|---|---|---|---|---|
| CYP1A2 | **2/3** | 0.67 | 0.177 – 0.961 | 0.50 | +0.236 | 1.191 |
| CYP2C9 | **3/3** | 1.00 | 0.464 – 1.000 | 0.125 | +0.395 | 0.293 |
| CYP2D6 | **1/3** | 0.33 | 0.039 – 0.823 | 0.875 | **−0.176** | 0.689 |
| CYP3A4 | **4/4** | 1.00 | 0.555 – 1.000 | 0.0625 | **+0.822** | 0.560 |
| pooled | 10/13 | 0.77 | 0.503 – 0.930 | **0.046** | | |

### NN-stratified offline only (R14, R24, R25 — the instruments actually in use today)
| endpoint | k/n | mean signed ratio | note |
|---|---|---|---|
| CYP1A2 | 2/2 | +0.921 | **below my predeclared n≥3 bar — untested** |
| CYP2C9 | 1/1 | +0.652 | **n=1 — untested** |
| CYP2D6 | 1/2 | −0.140 | **n=2 — untested** |
| CYP3A4 | 2/2 | +0.905 | **n=2 — untested** |

**Honest statement of power.** At n = 3-6 per endpoint no hit rate except CYP3A4's 6/6
separates from a coin flip. A 2-of-3 is not a 67% rate; it is two successes and one
failure, and its 95% interval spans 0.18 to 0.96. Read the **signed transfer ratios**,
not the hit rates — they use the magnitudes and are far better determined.

### Per-endpoint records that do NOT exist
- **We have never once tested CYP2C9 judgement against live on the current
  (NN-stratified, hinge-fitted) validator except inside R24.** n = 1.
- **CYP1A2's entire positive record is the last two windows.** It is 0/3 up to
  2026-09-10 and 2/2 from 2026-09-13.
- **R14's CYP1A2 and CYP2C9 live cells do not exist.** Two offline gains (+0.0125,
  +0.0117) were shipped and their outcomes were never recorded. Those are two lost
  observations — the only two lost in the whole record.

---

## 4. RECONCILIATION WITH THE TWO STANDING TRANSFER RULES

Claim under test: *hinge-fitted components transfer at ~0.65×, model blends at ~1.0×*
(`openadmet-r24-live-and-transfer-ratios`).

**Both halves are confirmed exactly where they were measured, and both rest on a single
window.**

| rule | supporting observations | value |
|---|---|---|
| hinge-fitted ≈ 0.65× | R24's three neural-fit columns **only** | (0.811 + 0.652 + 0.547)/3 = **0.670** ✓ reproduces |
| model blends ≈ 1.0× | R25's CYP1A2 **only** | **1.031** ✓ reproduces |

**The blend rule does not survive contact with the rest of the record.** Model blends
measured on the *same* NN-stratified validator, in the same era:

| sub | endpoint | mechanism | ratio |
|---|---|---|---|
| R25 | CYP1A2 | armB+TDI blend w=0.2 | **+1.03** |
| R14 | CYP3A4 | mt_aux8 raw blend w=0.20 | **+0.24** |
| R14 | CYP2D6 | mt_aux8 raw blend w=0.45 | **−0.83** |

Mean +0.15, sd 0.94. Widen to every blend in the record (panel-era denominators
included) and the mean is **+0.28**, range −0.83 to +1.03. **"Model blends transfer at
1.0×" is one observation, and the two other blends on the same validator are 0.24 and
−0.83.** Do not use 1.0× as a planning constant.

### Does the history support a PER-ENDPOINT refinement?

**Yes for CYP3A4. No for the other three — but the CYP2D6 record supports a
MECHANISM refinement that is stronger than either rule.**

- **CYP3A4**: 6/6 correct, ratios +0.23 … +1.57, mean **+0.66** (modern **+0.82**), and
  it is the only endpoint where any interval excludes a coin flip. A per-endpoint
  CYP3A4 rule of **0.8× with 6/6 sign reliability** is the single best-supported
  transfer statement this project has. This is consistent with — and quantifies —
  `openadmet-validator-representativeness-by-endpoint` ("trustworthy for CYP3A4 ONLY")
  and `openadmet-cyp3a4-is-power-not-magic` (6-7× more high-similarity rows, so its
  validator and the board overlap in regime).
- **CYP2D6**: no size-based rule exists; the mean signed ratio is negative in all three
  eras. **Split instead on whether the shipped column preserved its first two moments:**

  | group | k/n | mean ratio | Jeffreys 95% | submissions |
  |---|---|---|---|---|
  | spread preserved or expanded (sd/R11 ≥ 1.000) | **2/2** | **+0.62** | 0.333 – 1.000 | R03, R24 |
  | spread **compressed** (sd/R11 = 0.718, 0.758) | **0/2** | **−0.54** | 0.000 – 0.667 | R10, R14 |

  Both groups are n = 2, i.e. **below my predeclared n ≥ 3 bar, so this is a hypothesis
  with a perfect but underpowered record**, not an established rule. It is nonetheless
  the only variable that separates CYP2D6's two wins from its two losses, and it is
  independently established as a mechanism in
  `analysis/d6_dispersion_20260915/RESULT.md:236-250` and
  `openadmet-blends-must-preserve-spread`.
- **CYP1A2 / CYP2C9**: n is 2 and 1 on the current validator. No per-endpoint
  refinement is licensed. Use the hinge rule (0.65×) for hinge-fitted work and treat a
  blend as 0.65× too until a second blend observation exists.

**Verdict: the global two-rule split survives only as a description of R24 and R25. The
one per-endpoint refinement the history actually licenses is CYP3A4 at ~0.8× with 6/6
sign reliability; everything else should be planned at the 0.65× hinge discount with an
explicit sign-inversion probability from §3.**

---

## 5. THE ACTIONABLE OUTPUT — four thresholds

### The bars
- Macro noise floor **0.0168** (`openadmet-metric-noise-floor`, CORRECTED; the board
  publishes its own std 0.0158-0.0179, quoted at
  `analysis/aux_head_screen_20260915/RESULT.md:463`).
- A column gain `D` moves macro by `D/4`, so **1 macro σ on one column = +0.0672** —
  the project's standing G5 bar
  (`analysis/d6_loss_anatomy_20260915/PROTOCOL.md:162`). **No single-column offline gain
  in this project's history has ever been that large** (the biggest, R24's CYP2D6 at
  +0.0461, delivered +0.0252 live = 0.37 σ). So G5 is unreachable on one column and
  cannot be the operating bar — this is `openadmet-gate-shape-blocks-single-column`,
  now quantified.
- **Operating bar (B1): expected live column gain ≥ 0.0168** = +0.0042 macro = a quarter
  of a macro σ. Below this the project's own precedent (R26 at +0.0007 macro = 0.04 σ,
  `analysis/r26_a1_auxhead_20260915/RESULT.md:1`) is that the window is not worth
  spending.
- Threshold = `B1 / E[signed transfer ratio]`, using the modern-era (2026-09-09 →)
  mean signed ratio, which already folds sign inversions into the expectation.

### The four one-liners

> **CYP3A4** — an offline per-column gain is worth **0.82× of itself live** (6/6 correct,
> ratios +0.23…+1.57). **Act at offline ≥ +0.021.** This is the only endpoint whose
> offline judgement is established as better than a coin flip.

> **CYP2C9** — an offline per-column gain is worth **0.40× of itself live** (3/4, 3/3
> modern; the only two large-sample ratios are +0.46 and +0.65, the third is +0.08).
> **Act at offline ≥ +0.043.** Nothing smaller has ever produced a measurable live move:
> R10's +0.0092 offline returned +0.0007.

> **CYP1A2** — an offline per-column gain is worth **0.24× of itself live across the
> record and 0.92× across the last two windows only** (0/3 before 2026-09-13, 2/2 after).
> **Act at offline ≥ +0.071 on the whole record, or ≥ +0.018 if you accept that the
> instrument genuinely changed on 2026-09-13.** State which you are using. The cautious
> reading is +0.071, because CYP1A2 is the endpoint that has produced this project's two
> worst inversions (−1.13× and −2.83×) and both were "consistent in sign across
> independent local checks."

> **CYP2D6** — **there is no offline gain size worth acting on.** Expected signed
> transfer is **−0.18** (modern), **−0.14** (NN-era), **−0.22** (all history): the
> historical expectation of shipping a CYP2D6 offline gain is a *worse* column. The one
> exploitable structure is mechanism, not size: **act only on a change that preserves
> the column's mean AND sd exactly, and then at offline ≥ +0.027** (2/2, mean ratio
> +0.62, n = 2 — underpowered). A CYP2D6 candidate that compresses spread should be
> rejected at any offline gain, including a statistically significant one: R14's +0.0392
> had a 95% CI of [+0.0047, +0.0742] and lost 0.0324 live.

### How to apply this to the other two agents' findings
1. Read the offline gain **per column**, at the endpoint's representative stratum.
2. Multiply by the endpoint's E[ratio] above → expected live column gain.
3. Divide by 4 → expected live macro gain. Compare to +0.0042 (worth a window) and
   +0.0168 (one σ; not achievable on one column).
4. For CYP2D6 additionally check `sd_candidate / sd_incumbent == 1.000` and
   `mean_candidate − mean_incumbent == 0.000` before step 2. If not, stop.
5. A gain that clears the bar on two endpoints at once is worth twice as much and does
   not need a bigger per-column effect — the board scores four independent columns
   (`openadmet-score-per-endpoint-mixing`).

---

## 6. INCONSISTENCIES FOUND

### 6.1 R23 never scored — "macro 0.4580" is a PREDICTION, not a live board score
The brief listed `R23: macro 0.4580 (analysis/r23_a4_combined_20260912/RESULT.md)` as a
live anchor. It is not one. That file's own line 51 reads:

> "At 1:1 transfer: macro 0.4620 (R17) → **0.4580**"

and its title (line 1) and line 58 both say **"built, NOT submitted."**
`analysis/d6_dispersion_20260915/live_history.py:73-74` lists it as
`"not submitted (its A4 went live inside R24)"`. **R23's CYP3A4 column reached the board
only inside R24**, where the same column scored **0.3762**, not the 0.3855 that 0.4580
implies. Nothing is wrong with the underlying work; the number has simply been carried
as live when it is an expectation. **Correct: R23 has no live score. Delete 0.4580 from
any live table.**

### 6.2 R25's CYP1A2 recorded two ways; the identity resolves it
| file | R25 CYP1A2 | mean of its four columns | matches its own stated macro 0.4436? |
|---|---|---|---|
| `analysis/d6_loss_anatomy_20260915/PROTOCOL.md:10` | **0.4806** | 0.44530 | **NO** (off by 0.0017) |
| `analysis/analog_holdout_revival_20260914/fidelity.json` | **0.4739** | 0.443625 | **YES** |

**0.4739 is right**; 0.4806 is R24's value carried forward one window. `d6_loss_anatomy`
is a frozen protocol so the text should stand with an erratum, but any table built from
it inherits the error — and it understates CYP1A2's live gain by exactly the amount R25
bought. (This is the error the brief already suspected; it is confirmed and localised.)

### 6.3 "2.6 σ gap to first" is stale — already corrected today, but still in circulation
`analysis/field_structure_20260915/result.json` `_gap`: `leader_est 0.4182`,
`ours 0.4436`, **`gap_macro 0.0254`, `gap_sigma 1.51`**, with the note
*"2.6 sigma was stale: it used the R11-era gap 0.0443 with the corrected 0.0168 floor."*
Six files still quote 0.0443 as the current gap:
`argmin_gate_20260915/RESULT.md:29`, `combiner_alternatives_20260914/RESULT.md:63,375`,
`family_decomposition_20260915/RESULT.md:86`, `interval_as_target_20260915/RESULT.md:193`,
`r26_a1_auxhead_20260915/RESULT.md:72`, `tdi_effect_recompute_20260915/RESULT.md:63`.
**Correct: 0.0254 macro, 1.51 σ.** Per-endpoint gaps to the field best are CYP1A2 0.0328
(our rank 6), CYP2C9 0.0372 (rank 9), CYP2D6 0.0536 (rank 4), CYP3A4 0.0055 (rank 2).

### 6.4 The A4 round's transfer ratio is recorded as both 0.880 and 0.939
- `analysis/r08_a4_stack_20260909/RESULT.md:33-36` computes it directly:
  *"a transfer ratio of **0.880**, with a 4 dp rounding band of 0.875 to 0.884"*
  (panel relative gain 2.6% ⇒ 0.010918; realised 0.419906 − 0.4103 = 0.009606).
- `analysis/a1_gbm_candidate_20260910/PREDECLARED_READING.md:49-50,81` and its
  `RESULT.md:36` quote the same third observation as **0.939** ("the series is now 0.63,
  0.675, **0.939**, −1.93").

**0.880 is right** — it is the arithmetic in the round's own RESULT.md and reproduces
from the board cells. 0.939 appears to be a pre-result estimate that was never updated.
The whole downstream series (`a1_shape_panel_gate_20260909:69`,
`a1_panel_geometry_placement_20260909:60`) inherits it. Immaterial to any decision, but
it is a third instance of the documented carried-number pattern.

### 6.5 Two files assert R14 held CYP1A2/CYP2C9 at delta 0.0000 "as designed" — the CSVs say otherwise
- `analysis/r17_revert_20260912/RESULT.md:34-35`: *"Per-column independence is now
  confirmed twice (R08→R11 tripwire passed exactly; R11→R14 held CYP1A2/CYP2C9 at delta
  0.0000 as designed)"*
- `analysis/r19_a1_spread_safe_20260912/RESULT.md:25`: *"Per-column independence has been
  confirmed twice live (R08→R11, R11→R14): untouched columns move by exactly 0.0000."*

Diffing the two submitted CSVs directly (`live_verdict_20260910/…R11….csv` vs
`scratch_interpolation_20260911/…R14….csv`):

| column | max abs Δ | sd R11 → R14 |
|---|---|---|
| CYP1A2 | **0.2848** | 0.9600 → 0.9015 |
| CYP2C9 | **0.1972** | 0.7104 → 0.7110 |
| CYP2D6 | 0.7308 | 0.8650 → 0.6556 |
| CYP3A4 | 0.1334 | 0.8985 → 0.9094 |

R14 changed **all four** columns (`SUBMITTED.md:9-16` gives the four weights
0.25/0.30/0.45/0.20 and `manifest_r14.json` confirms them). So R11→R14 is **not** an
independence confirmation, and the claim of "twice confirmed" rests on one case (R08→R11)
plus two later ones (A4R→A1GBM, R24→R25) that do hold. **Practical consequence: R14's
CYP1A2 and CYP2C9 live cells are unknown, not zero, and the two offline gains shipped in
them (+0.0125, +0.0117) have no recorded outcome.**

### 6.6 R14's macro is recorded nowhere, and the two surviving numbers do not close
- `analysis/d6_dispersion_20260915/RESULT.md:207-210` states plainly that **no file in
  this repo holds R14's CYP2D6 board cell**; five files record only its delta, +0.0324.
- No file holds R14's macro either. The memory note
  `openadmet-blends-must-preserve-spread` says −0.0084 macro.
- Arithmetic: if only CYP2D6 (+0.0324) and CYP3A4 (−0.0023) had moved, the macro delta
  would be **+0.00752** (macro 0.4700), not 0.0084. Closing to 0.0084 requires CYP1A2 +
  CYP2C9 to have summed **+0.0035** worse — which is consistent with §6.5 (they did
  change) and inconsistent with §6.5's "delta 0.0000" claim.
- Conversely `0.0324 / 4 = 0.00810`, so if +0.0324 was itself back-derived from a macro
  delta, that macro delta was **0.0081**, not 0.0084.

**R14 is the only submission whose macro identity cannot be closed from artifacts. Its
CYP2D6 ratio (−0.83) is therefore the one entry in §2 resting on a derived number.**
Removing it changes CYP2D6 to 2/4 all-history and 1/2 modern, and the mechanism split to
2/2 vs 0/1 — it does not change any threshold sign or any conclusion.

### 6.7 R24's predeclared expectation cites two live inversions that never happened
`analysis/r24_joint_member_hinge_20260913/PREDECLARED_LIVE_EXPECTATION.md:44-45` calls
CYP2D6 "the risk column … two prior live inversions (R14, **R15/R16**)". R15 and R16 were
never submitted — both directories' RESULT.md say "do not submit" and
`d6_dispersion_20260915/live_history.py:64-69` lists them as rejected. The real prior
inversions were **R10 and R14**. Same count, wrong identities; no decision consequence.

### 6.8 Sensitivity: the one sign call that depends on which offline statistic you use
R09's CYP2C9 is scored ✗ above because the intervention was promoted on the 14-family
panel gain **+0.4016** and the column got worse. The same document's ordinary-scaffold
gain was **−0.000481**, i.e. it predicted a (tiny) *worsening*, which the board
delivered. Counting that reading instead makes **CYP2C9 4/4 all-history (p = 0.0625)**.
No other row in the table is sign-sensitive to the choice of offline statistic. I report
the panel reading as primary because it is the number the round was decided on.

---

## 7. WHAT THIS AUDIT CHANGES

1. **Replace "offline gains called 2 of 4"** with the per-endpoint table in §3. The
   global statement is 13/20, and it is uninformative because the endpoints differ by
   more than the pooled uncertainty.
2. **CYP3A4 is the only endpoint whose offline judgement is measurably informative**
   (6/6, p = 0.0156, ratio 0.82×) — and it is also the endpoint with the smallest
   remaining gap to the field (0.0055, rank 2). That is an unfortunate pairing: our
   trustworthy instrument points at our smallest opportunity.
3. **CYP2D6 offline magnitude has negative expected value.** Any CYP2D6 finding must
   come with a moment-preservation proof before its size is even discussed.
4. **CYP1A2's two successes are both post-2026-09-13.** If the other agent's CYP1A2
   finding is hinge-fitted or a restored blend, the 0.92× ratio is defensible; if it is
   any other construction, the applicable record is 0/3.
5. **CYP2C9 has n = 1 on the current validator.** Whatever the other agent finds, it
   will be the second observation. Predeclare it as such.
6. **Two observations were lost** (R14's CYP1A2 and CYP2C9). Record every column's live
   cell after every window, even for columns nobody predicted — the cost of not doing so
   is permanent.

---

## AUTHORSHIP
Built 2026-09-15 by an audit session at the coordinating session's request, from repo
artifacts only. `build_history.py` regenerates `history.csv` / `history.json` and
re-runs all arithmetic checks. Read-only outside this directory.
