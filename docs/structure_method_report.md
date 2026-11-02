# OpenADMET CYP Inhibition Blind Challenge: structure prediction method report

Team: Scigantic
Track: CYP3A4 ligand-complex structure prediction, scored by lDDT-PLI (higher is better)
Entry on the board at the time of writing: 0.4565

## 1. Summary

We predict each CYP3A4 ligand complex with Boltz-2 co-folding, using a ligand-free
CYP3A4 crystal template, the heme as an explicit second ligand chain, and a soft
pocket constraint on four active-site residues. The pose submitted for each compound
is the highest-confidence diffusion sample.

The method is deliberately plain. The substance of this report is the set of
negative results around it, because across roughly fifteen GPU-hours of controlled
ablation we were unable to move the input side or the sampler side of this pipeline
in a direction that survives its own noise. We report the ablations, the effect
sizes with confidence intervals, and the one measurement that explains why the
others came back empty.

## 2. System setup

Protein: the 486-residue CYP3A4 construct supplied with the challenge, as chain A.
Construct residue *i* corresponds to canonical CYP3A4 residue *i* + 21.

Heme: an explicit chain B, declared as CCD ligand `HEM`. Modeling the heme as part
of the complex rather than omitting it matters because lDDT-PLI is computed over
protein-ligand contact pairs and the heme face defines one wall of the pocket.

Ligand: chain C, from the organizers' SMILES. We re-pulled the organizers' ligand
file after they added stereochemistry on 2026-10-24 and verified the shipped
coordinates against it (section 5).

Template: 1TQN, ligand-free CYP3A4 at 2.05 Angstrom.

Pocket constraint: binder C within 8 Angstrom of construct residues 98, 283, 284
and 288 (canonical 119, 304, 305, 309).

Alignment: the ColabFold MMseqs2 server, `uniref.a3m` concatenated with
`bfd.mgnify30.metaeuk30.smag30.a3m`, which is the order and the pair Boltz itself
uses when `use_env` is set. 14,364 sequences.

One implementation detail is worth recording because it is not documented and it
bounds what an MSA experiment on this pipeline can test. Every alignment Boltz
obtains through its MSA-server path is MMseqs2-filtered: `mmseqs2.py` defaults
`use_filter=True` and `main.py` calls `run_mmseqs2(..., use_env=True,
use_pairing=False)` without overriding it, so no command-line flag reaches the
unfiltered alignment. Testing the filter requires fetching the alignment out of
band and feeding it in as a file, which is what we did in section 4.3.

Generation: boltz 2.2.1, 3 diffusion samples, 3 recycling steps, 200 sampling
steps, `--use_potentials`.

Selection: highest `confidence_score`. Boltz writes its samples pre-sorted by
confidence, so this is rank-1 of the sampled set.

Post-processing: `CONECT` and `TER` records stripped, chain B HETATM records
renamed `HEM`, ligand left as `LIG` on chain C.

## 3. The benchmark's resolving power is the binding constraint

We evaluated candidate changes on 22 held-out CYP-ligand crystal complexes with
OpenStructure, at the scorer's defaults (`lddt_pli_radius=6.0`,
`add_mdl_contacts=True`, `bs_radius=4.0`, `lddt_lp_radius=15.0`).

In a ten-arm sampler ablation we included a replica arm: an arm byte-identical to
the control, same seed, same inputs. It scored **0.4250 against the control's
0.4107, a difference of +0.0143 lDDT-PLI**. That is the pipeline's run-to-run
nondeterminism on this benchmark, and it is larger than every apparent gain we
measured anywhere in the campaign. The matching detection floor is 0.0338
lDDT-PLI. Our gap to first place is 0.0330.

In other words, the internal benchmark is blind at exactly the scale that decides
rank. Converted through the measured slope of 0.109 lDDT-PLI per Angstrom of
BiSyRMSD, the floor corresponds to 0.31 Angstrom of pose accuracy, which is a
larger improvement than any single input change plausibly delivers.

We consider this the most useful result of the campaign and we state it before the
ablations because it is the correct lens for reading them. A null result from a
22-case benchmark is not evidence that a lever is dead. It is mostly evidence that
the benchmark cannot see the lever. Where we claim a lever is closed below, the
claim rests on a confidence interval that excludes zero or on direct board
feedback, not on a null.

A corollary we acted on late and should have acted on early: the public leaderboard
measures the actual 20 compounds exactly, and it is free. For effect sizes in the
0.01 to 0.03 band it is a strictly better instrument than any internal benchmark we
could afford to build.

## 4. Ablations

### 4.1 Narrowing the alignment by pocket identity makes it worse

We built four alternative alignments by filtering the 14,078-row shipped MSA on
identity at the 48-residue contact core (the columns with at least 10 contacts
across 112 CYP crystals, which recover exactly the six Gotoh substrate recognition
sites), and ran all of them against the shipped alignment as the control.

| arm | sequences | Neff/sqrt(L) | lDDT-PLI | contact recall | lDDT-LP |
|-----|-----------|--------------|----------|----------------|---------|
| M0 shipped | 8192 | 270.7 | 0.4079 | 0.7596 | 0.8002 |
| M1 pocket-strict | 1138 | 13.0 | 0.3834 | 0.7281 | 0.7320 |
| M2 pocket-moderate | 1664 | 23.7 | 0.3756 | 0.7283 | 0.7509 |
| M3 pocket-very-strict | 226 | 2.2 | 0.4135 | 0.7647 | 0.7920 |
| M4 hybrid | 1796 | 19.4 | 0.3975 | 0.7471 | 0.7590 |

M1 and M2 lose contact recall with intervals that exclude zero (M1 -0.0315, 95% CI
[-0.0527, -0.0103]; M2 -0.0313, CI [-0.0543, -0.0080]). This is a real negative,
not a null.

The obvious explanation is an Neff cliff, and it is wrong. M3, the shallowest arm
at 226 sequences and Neff/sqrt(L) of 2.2, is the *best* of the narrowed arms and
beats M1 by +0.0366 contact recall, CI [+0.0154, +0.0581]. Depth is therefore not
what distinguishes the narrowed arms from each other, and our original design
confounded composition with depth because every narrowed arm also cut depth.

### 4.2 More depth does not help either

We raised the cap to `--max_msa_seqs 16384` for +46.3% Neff. Null to slightly
negative.

### 4.3 A 31-times deeper unfiltered alignment is a clean null

To test the MMseqs2 filter we fetched both `env` and `env-nofilter` alignments for
the construct directly from the ColabFold API. Filtered: 6,522 uniref plus 7,842
env, 14,364 total. Unfiltered: 345,164 uniref plus 105,706 env, **450,870
sequences**, 428,544 unique, 180,602 covering at least 90% of the query.

Naive truncation to the 8,192-row budget is worse than the shipped alignment, which
we predeclared and confirmed (-75.9% Neff), because the file order of an unfiltered
search is dominated by near-duplicates. Random sampling of 8,192 from the
well-covered pool instead gives Neff 6,287.9 against the shipped alignment's
5,433.3, a +15.7% global gain, and pocket-column Neff of 6,006.6 against 542.8,
which is **+1007%**.

That arm is a clean null on every endpoint: contact recall +0.0024, CI [-0.0284,
+0.0324]; lDDT-PLI -0.0046, CI [-0.0373, +0.0299]; lDDT-LP +0.0038. All intervals
include zero, and the control reproduced the independent control from the sampler
ablation to within 0.0013, so the comparison itself is sound.

An eleven-fold increase in pocket-column diversity changing nothing is the
strongest statement we can make about this axis. The alignment is not the
bottleneck for this target.

Two related sub-results. Close-homolog coverage is already saturated: 4 sequences
at 99% identity or better, 21 at 90% or better, 82 at 80% or better, so there is
nothing to add at the top. And the Neff-optimal subset we went looking for does not
exist: the well-covered pool holds 7,495 sequences, which is smaller than the 8,192
budget, so there is no reordering gain to capture. An apparent +42.8% headroom we
first measured was fragments, drawn from a tail that is 89.2% partial sequences.

### 4.4 Sampler knobs

`--step_scale` is the per-step interpolation gain toward the denoised estimate. At
the boltz2 default of 1.5 the sampler overshoots the mode by 50% per step. Lowering
it is mechanically live and gives a clean dose-response in pose spread (depth
standard deviation 0.202, 0.349, 0.437, 0.533 as it falls), and it is monotonically
harmful: fold quality collapses below 1.25, and at 0.5 all 66 ligands in the arm
stop being chemically valid molecules. The broken-ligand ladder across the arms is
0, 0, 1, 9, 66.

`--subsample_msa` was the best arm in that experiment at +0.0120, which is inside
the noise floor and therefore not a result. It is worth noting that this flag is
declared `is_flag=True` with no `default=`, so click passes `False` when it is
absent while its own help string says "Default is True". It had been off in every
run we had ever done.

### 4.5 The pocket constraint is not what keeps poses shallow

Our predicted ligands sit about 6.8 Angstrom from the heme iron, while the two
crystal far-site cases sit at 13.59 and 12.71 Angstrom. The natural suspect is the
8 Angstrom pocket constraint. It is not the cause. We scored a genuinely
unconstrained run and its poses sit at 5.8 to 6.9 and 4.0 to 7.3 Angstrom for those
two cases, which is no deeper than the constrained ones. The constraint is
exonerated on direct evidence.

### 4.6 Pose selection has no usable signal, and consensus selection actively loses

We tested 17 selection signals (12 confidence-derived, 5 geometric) on two
independent sets. Nothing holds above noise, and top-1 selection is exactly chance.
An earlier claim of +0.0183 for `confidence_score` did not replicate: on 88
held-out cells it was -0.0011.

The board then gave us a sharper result than our benchmark could. A
coordinate-medoid pick over 10 diffusion samples, generated with an otherwise
identical recipe, **scored 0.4034 against 0.4565 for the rank-1 pick**. Consensus
selection costs 0.053 lDDT-PLI here. This is consistent with the error structure we
see throughout: pose error on this target is a displaced mode, not scatter about the
right answer, so moving a pick toward the center of the sampled distribution moves
it toward the center of the wrong mode. Averaging and consensus are the wrong family
of operation for this failure mode, and we would caution anyone reading an
encouraging medoid result on a different target against transferring it here.

## 5. Stereochemistry audit

The organizers added stereocenters to the test ligands on 2026-10-24, after our
poses were generated. Four of the twenty compounds carry stereochemistry. We
re-verified the shipped coordinates against the current reference SMILES by
assigning bond orders from the reference template and reading CIP codes from the 3D
structure. Three are ordinary R/S centers and all three match.

The fourth, OCNT-2313038, is a 1,4-disubstituted cyclohexane whose relationship is
cis/trans across the ring rather than R/S, so RDKit assigns it no CIP code and a
CIP-based check passes it without testing anything. We tested it geometrically
instead, by fitting the ring plane and comparing which face each exocyclic
substituent sits on, calibrated against embedded conformers of the reference SMILES
(12 of 12 trans). The shipped pose is trans and correct.

We record this because a CIP-code comparison is the obvious way to run this audit
and it silently skips exactly the case most likely to be wrong.

## 6. Known remaining error

The systematic error we did not fix is pose depth. Predictions cluster around 6.8
Angstrom from the heme iron with a standard deviation under 0.4 Angstrom, while the
crystal distribution is much wider and includes genuine far-site binders at 12 to 14
Angstrom. Per protomer, 4.1% of our reference corpus binds far from the heme. We
confirmed this is real and not an artifact: an earlier 19.6% figure was a
cross-protomer measurement error, visible as a 75.86 Angstrom maximum, and the
per-protomer re-check of the two far-site cases confirms a single chain with a
single ligand copy at 13.59 and 12.71 Angstrom.

Neither the input side nor the sampler side moved this. The pocket constraint is not
responsible (4.5), the alignment is not responsible (4.1 to 4.3), and the sampler
cannot be loosened enough to reach those depths without destroying the ligands
(4.4). We believe closing it requires a model whose prior over CYP3A4 binding
geometry is different, not a better-configured version of this one.

## 7. Reproduction

Each ablation directory under `analysis/` holds a `PROTOCOL.md` written and
timestamped before any output existed, the launcher with its preflight assertions,
the scoring script, the per-pose sample table and a `RESULT.md` recording the
outcome against the predeclared gates. Where we overrode one of our own gates, the
override and its direction are recorded in that file, including one case where the
override was in the permissive direction and we flagged it as the kind of move to
distrust. Where a claim was withdrawn, the withdrawal is an addendum in the same
file rather than an edit to the original text.
