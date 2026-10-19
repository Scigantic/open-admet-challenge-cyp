# Training the members

The twelve members behind R24/R25 come from four scripts. Each file here is the one that
produced the predictions in `data/members/`, with only the SageMaker account, role and
bucket identifiers replaced by placeholders. All members are 5-fold scaffold ensembles;
the fold assignment is fixed in `data/training/chemeleon_22/folds.npy` for the CheMeleon
arms and derived inside the script for the others (seed 42).

| member | script | table | notes |
|---|---|---|---|
| armA | `chemeleon_arms/sagemaker_runner.py --arm A` | `chemeleon_22` | d_h 700, depth 5, FFN 300, dropout 0.1, random init, MAE, inverse-variance sample weights |
| armB | `--arm B` | `chemeleon_22` | d_h 2048, depth 4, FFN 1024, dropout 0.2, random init |
| armC | `--arm C` | `chemeleon_22` | as B, message passing initialised from the CheMeleon checkpoint |
| armD | `--arm D` | `chemeleon_22` | as B, initialised from the organizers' CYP-finetuned CheMeleon checkpoint |
| ivB2048 | `--arm B --objective interval --weighting uniform` | `chemeleon_22` | scored heads trained on the interval hinge, uniform weights |
| ivC2048 | `--arm C --objective interval --weighting uniform` | `chemeleon_22` | |
| mu2048 | `--arm B --objective mae --weighting uniform` | `chemeleon_22` | |
| ii2048 | `--arm B --objective interval --weighting invvar` | `chemeleon_22` | |
| g26_tdi_armB | `--arm B` on `chemeleon_26_tdi` | `chemeleon_26_tdi` | arm B plus four TDI-condition pIC50 heads and the compounds only in that file |
| union18 | `workbench_recipe/train_briford_workbench.py --member-spec union18_member_spec.json` | `workbench_recipe/union18_full.csv` | recipe-level reproduction of the public SuperCowPowers Workbench member (18 heads) |
| mt_aux8 | `... --member-spec mt_aux8_member_spec.json` | `workbench_recipe/mt_aux8_full.csv` | same recipe, organizer data only (8 heads) |
| aux_ac50 | `chemprop_22/train_aux_ac50.py` | `chemprop_22/train_table.csv` | union18 heads plus four qHTS AC50 heads, inverse-variance weights |
| cp_interval | `chemprop_22/train_interval.py` | `chemprop_22/train_table.csv` | as aux_ac50 with the scored heads on the interval hinge |

Hyperparameters not listed are in the scripts. The CheMeleon arms ran one fold per
SageMaker job (`chemeleon_arms/launch.py`, ml.g6.2xlarge); the Chemprop members ran
locally or on one GPU. Wall clock per fold was 5 to 40 minutes.

## Tables

`data/training/chemeleon_22/train_table.csv` has 31,669 compounds and 22 target columns:

| columns | source | labels |
|---|---|---|
| `cyp*_pic50_direct_inhibition` | organizers, `cyp-challenge-train-test` | 6,525 |
| `cyp*_log2fc` | organizers, single-concentration screen | 17,500 |
| `cyp*_pic50_chembl` (five isoforms incl. CYP2C19) | ChEMBL IC50 records per isoform, median per connectivity key | 56,705 |
| `cyp*_max_response` (five isoforms) | PubChem AID 1851 (Veith et al. 2009 qHTS), maximum response | 80,665 |
| `cyp*_pic50_qhts` | PubChem AIDs 410, 883, 884, 891 (qHTS AC50), median per InChIKey | 15,070 |

`chemeleon_26_tdi/train_table.csv` adds `cyp*_pic50_tdi_condition` from the organizers'
TDI file and the compounds that only appear there. `bounds.csv` holds the credible
interval per scored label; `weights.npy` the inverse-variance sample weights derived
from them; `folds.npy` the scaffold fold of each row. `workbench_recipe/union18_full.csv`
is the 18-head table the Workbench recipe trains on (no qHTS AC50 heads);
`mt_aux8_full.csv` is the organizer-only subset.

Public rows were joined to the challenge on InChIKey connectivity (first block), with
salts, stereo and charge ignored, so that a held-out compound could not enter under a
different identifier. One blind compound (OCNT-0496981) does appear in the union table
under its PubChem identifier with AID 1851 maximum-response values only, no potency
label; it was kept for parity with the public recipe and is recorded in
`workbench_recipe/sealed_manifest.json`.

## Checkpoints

Two files were shipped with the CheMeleon jobs and are not in this repo:

- `chemeleon_mp.pt`: the CheMeleon foundation checkpoint, downloaded by chemprop from
  https://zenodo.org/records/15460715 (`chemprop train --from-foundation CheMeleon`
  fetches it to `~/.chemprop/`).
- `organizer_cyp.pth`: the organizers' CYP-finetuned CheMeleon,
  https://huggingface.co/openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1.

## Other files

- `featurize.py`: challenge loading, scaffold function and the ST-RAE port used by every script.
- `build_r08_baseline.py`: how the incumbent out-of-fold baseline in `data/oof_base/` was reconstructed.
- `chemeleon_arms/INTERVAL_2X2_RESULT.md`, `GROUPING_RESULT.md`: the two experiments that
  fixed the objective, weighting and head grouping of the CheMeleon arms.
- `workbench_recipe/FROZEN_REPRODUCIBILITY_CONTRACT.json`: the pinned public recipe and
  hashes the union18 and mt_aux8 reproductions were run against.
