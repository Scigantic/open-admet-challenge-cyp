#!/usr/bin/env python3
"""Standalone score-era Briford/Workbench Chemprop member runner.

This is a deliberately small extraction of the public Workbench training path.  It
does not import Workbench or depend on a Workbench FeatureSet/Endpoint.  One invocation
trains one *member specification* with the score-era five-model scaffold-ensemble recipe,
writes pooled out-of-fold predictions, predicts a supplied blind CSV with every fold
model, and averages the five raw predictions.  This establishes recipe-level parity; it
does not claim exact or bitwise reproduction of the historical scored artifact.

Public provenance (SuperCowPowers/workbench):

* ``971f6a7f`` introduced ``cyp_chemprop_union.py`` (the 18-head union-p30 member).
* ``34689b18`` is the scored-ensemble era and contains the public ensemble recipe.
* The model construction, split, checkpoint, and prediction logic below is minimally
  adapted from, respectively:

  - ``src/workbench/training/chemprop_core.py``
  - ``src/workbench/training/splits.py``
  - ``src/workbench/model_scripts/chemprop/chemprop.template``
  - ``src/workbench/endpoints/chemprop_utils.py``

The copied semantics are fixed in ``SCORE_ERA_RECIPE``.  Input-specific choices are
limited to member name, CSVs, column names, targets, and positional task weights.

Dry run (imports RDKit/pandas, but never Chemprop/Lightning/Torch):

    python train_briford_workbench.py --member-spec spec.json --dry-run

Training (intended for a pinned SageMaker GPU image):

    python train_briford_workbench.py --member-spec spec.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SOURCE_REPOSITORY = "https://github.com/SuperCowPowers/workbench"
MODEL_RECIPE_COMMIT = "971f6a7f"
SCORE_ERA_COMMIT = "34689b183c943ffa1ef7d589c45962ba8b713697"
COMPOSITE_UNION_PIN = "4cf07a8cb4366eb29111fb42396336b20b69ada0"
COMPOSITE_FINAL_RECIPE_PIN = "8c72c7559471febaf0962fd551b4eaab8db3a136"
SCORE_ERA_DEFAULT_INSTANCE = "ml.g6.2xlarge"
SCORE_ERA_MUTABLE_IMAGE = "py312-base-training:latest"
SCORE_ERA_PYTHON_MAJOR_MINOR = (3, 12)

# Blob hashes are over the exact public files as materialized by ``git show COMMIT:path``.
PROVENANCE_BLOBS = {
    "union_member_declaration": {
        "commit": MODEL_RECIPE_COMMIT,
        "path": "ml_pipelines/OpenADMET/cyp/cyp_chemprop_union.py",
        "sha256": "0011c94b3da0faaa40d3695ed727dda57eb9c1c3c2720e7d78cf862ef71447ea",
    },
    "composite_old_18_head_union": {
        "commit": COMPOSITE_UNION_PIN,
        "path": "ml_pipelines/OpenADMET/cyp/cyp_chemprop_union.py",
        "sha256": "f1e131ff10423564f7cf5164336e3eb36997482c550e77850a889323b3cc2ac6",
    },
    "composite_final_ensemble_recipe": {
        "commit": COMPOSITE_FINAL_RECIPE_PIN,
        "path": "ml_pipelines/OpenADMET/cyp/scripts/cyp_ensemble_submit.py",
        "sha256": "f94d9b57cf7c4f14be811b0598a06b2f555ecb720c36b58abb76e467b56be2fd",
    },
    "score_era_training_core": {
        "commit": SCORE_ERA_COMMIT,
        "path": "src/workbench/training/chemprop_core.py",
        "sha256": "089ab0316573f478b8d1c5a25c558363d545090cad82e43fbb40ccfb2897c539",
    },
    "scaffold_split": {
        "commit": SCORE_ERA_COMMIT,
        "path": "src/workbench/training/splits.py",
        "sha256": "b80b9d31db0b7a14fe5f69ef058c7631f8e6fe60dbb6295f3db59b1b902af9d8",
    },
    "score_era_template": {
        "commit": SCORE_ERA_COMMIT,
        "path": "src/workbench/model_scripts/chemprop/chemprop.template",
        "sha256": "32bc43e240ef66f36bab4db13fefc0e240d36b2a64e6611393789c18a1db4d8f",
    },
    "score_era_ensemble": {
        "commit": SCORE_ERA_COMMIT,
        "path": "ml_pipelines/OpenADMET/cyp/scripts/cyp_ensemble_submit.py",
        "sha256": "f94d9b57cf7c4f14be811b0598a06b2f555ecb720c36b58abb76e467b56be2fd",
    },
    "score_era_environment_lock": {
        "commit": SCORE_ERA_COMMIT,
        "path": "sagemaker_images/pytorch_chem/training/requirements.lock",
        "sha256": "baf86702f5cd90b028d38bdc94d6713efff8cd8baabb722e89fa1fe607791366",
    },
}

# These are intentionally not tuning knobs.  A member spec may repeat them, but a
# differing value is rejected before training so a run cannot silently cease to be a
# score-era parity run.
SCORE_ERA_RECIPE: dict[str, Any] = {
    "n_folds": 5,
    "split_strategy": "scaffold",
    "split_seed": 42,
    "fold_seed_rule": "42 + zero_based_fold_index",
    "initialization": "random",
    "message_passing": "BondMessagePassing",
    "hidden_dim": 700,
    "depth": 5,
    "dropout": 0.1,
    "aggregation": "NormAggregation",
    "aggregation_norm": 100,
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
    "batch_norm": True,
    "criterion": "mae",
    "batch_size": 64,
    "max_epochs": 400,
    "patience": 40,
    "warmup_epochs": 2,
    "init_lr": 1e-4,
    "max_lr": 1e-3,
    "final_lr": 1e-4,
    "lr_schedule": "Chemprop NoamLR",
    "checkpoint_monitor": "val_loss",
    "checkpoint_mode": "min",
    "precision": "16-mixed",
    "features": "SMILES only",
}

# Exact versions in the 34689b18 pytorch_chem training requirements.lock.  The local
# dry-run reports drift without importing the training stack.  A real training run
# refuses drift unless --allow-version-drift is explicit and recorded in the manifest.
SCORE_ERA_PACKAGES = {
    "chemprop": "2.3.0",
    "lightning": "2.6.5",
    "numpy": "2.4.6",
    "pandas": "2.3.3",
    "rdkit": "2026.3.4",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.1",
    "torch": "2.13.0+cu130",
}

KNOWN_FIDELITY_GAPS = [
    (
        "Recipe-level parity only: pinned public caches and literal Workbench joins reproduce "
        "the published pre-expansion union exactly at 31,670 rows and 161,400 labels, but the "
        "historical serialized FeatureSet bytes and checksum were not published."
    ),
    (
        "The corrected inputs preserve the source row order and source SMILES selected by the "
        "public Workbench operations. The historical stored FeatureSet row artifact is absent, "
        "so exact historical scaffold membership cannot be proven independently."
    ),
    (
        "The literal public recipe retains one AID1851 cross-assay auxiliary structure whose "
        "connectivity matches a blind compound. It contains no hidden challenge direct label "
        "and is retained deliberately for recipe parity."
    ),
    (
        "The historical checkpoints, target scalers, inference vectors, training logs, and "
        "submitted CSV are unavailable; this run trains all 20 neural folds anew."
    ),
    (
        "The score-era Workbench image was referenced by a mutable tag and its digest was not "
        "published. The pinned replacement image is newly built from the public score-era "
        "Dockerfile and exact package lock, so environment semantics match without claiming "
        "historical image identity."
    ),
    (
        "A package lock, image digest, seeds, and source bytes seal the recipe, but GPU/kernel "
        "nondeterminism means independent training is not promised to be bitwise identical."
    ),
]

RUNNER_SOURCE = Path(__file__).resolve()
EXECUTED_RUNNER_ARCHIVE = "executed_train_briford_workbench.py"

OUTPUT_NAMES = {
    "validation": "input_validation.json",
    "splits": "scaffold_fold_assignments.csv",
    "manifest": "run_manifest.json",
    "metrics": "metrics.json",
    "oof": "oof_predictions.csv",
    "blind_raw": "blind_predictions_by_fold.csv",
    "blind_mean": "blind_predictions_mean.csv",
}


@dataclass(frozen=True)
class RunConfig:
    member_name: str
    member_spec_path: Path | None
    data_csv: Path
    blind_csv: Path
    output_dir: Path
    smiles_column: str
    id_column: str
    targets: tuple[str, ...]
    task_weights: tuple[float, ...]
    scored_targets: tuple[str, ...]
    interval_columns: dict[str, dict[str, str]]
    expected_data_sha256: str | None
    expected_blind_sha256: str | None
    expected_rows: int | None
    sealed_input_manifest_path: Path | None
    sealed_input_manifest_sha256: str | None
    sealed_input_member_key: str | None
    num_workers: int
    training_image_uri: str | None
    training_image_digest: str | None
    sagemaker_instance_type: str | None
    expected_runner_sha256: str | None
    allow_version_drift: bool
    dry_run: bool


@dataclass
class ValidatedInputs:
    train: pd.DataFrame
    blind: pd.DataFrame
    train_valid_source_rows: np.ndarray
    blind_valid_positions: np.ndarray
    scaffolds: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    report: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attest_and_archive_runner(config: RunConfig) -> dict[str, Any]:
    """Bind the loaded run to the exact source bytes supplied to the container."""
    source_bytes = RUNNER_SOURCE.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    expected = config.expected_runner_sha256
    if expected is not None and observed != expected:
        raise RuntimeError(
            "executed runner SHA-256 does not match --expected-runner-sha256: "
            f"expected={expected}, observed={observed}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    archived_path = config.output_dir / EXECUTED_RUNNER_ARCHIVE
    archived_path.write_bytes(source_bytes)
    archived = sha256(archived_path)
    if archived != observed:
        raise RuntimeError(
            "archived runner SHA-256 differs from executed source bytes: "
            f"source={observed}, archived={archived}"
        )
    return {
        "source_path": str(RUNNER_SOURCE),
        "observed_sha256": observed,
        "expected_sha256": expected,
        "expected_sha256_provided": expected is not None,
        "expected_sha256_matches": expected == observed if expected is not None else None,
        "archived_path": str(archived_path.resolve()),
        "archived_sha256": archived,
        "archived_bytes": archived_path.stat().st_size,
        "archive_matches_source": True,
    }


def sequence_sha256(values: Sequence[Any]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")


def package_versions() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for distribution, expected in SCORE_ERA_PACKAGES.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        report[distribution] = {
            "expected": expected,
            "observed": observed,
            "exact_match": observed == expected,
        }
    return report


def parse_csv_strings(raw: str | Sequence[str] | None, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    else:
        values = [str(item).strip() for item in raw]
    if not values or any(not item for item in values):
        raise ValueError(f"{field} must contain non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicates: {values}")
    return tuple(values)


def parse_weights(raw: str | Sequence[Any] | None) -> tuple[float, ...]:
    if raw is None:
        return ()
    values = raw.split(",") if isinstance(raw, str) else raw
    try:
        parsed = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("task_weights must be numeric") from error
    if not parsed or any(not math.isfinite(value) or value < 0 for value in parsed):
        raise ValueError("task_weights must be finite and non-negative")
    if not any(value > 0 for value in parsed):
        raise ValueError("at least one task weight must be positive")
    return parsed


def _path_from_spec(value: str | None, spec_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (spec_dir / path).resolve()


def _image_digest(uri: str | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if uri and "@sha256:" in uri:
        return "sha256:" + uri.rsplit("@sha256:", 1)[1]
    return None


def _validate_locked_recipe(spec: dict[str, Any]) -> None:
    supplied: dict[str, Any] = {}
    nested = spec.get("recipe", {})
    if nested is not None and not isinstance(nested, dict):
        raise ValueError("member spec 'recipe' must be an object")
    supplied.update(nested or {})
    for key in SCORE_ERA_RECIPE:
        if key in spec:
            supplied[key] = spec[key]
    unknown = sorted(set(supplied) - set(SCORE_ERA_RECIPE))
    if unknown:
        raise ValueError(f"unknown locked recipe field(s): {unknown}")
    mismatches = {
        key: {"expected": SCORE_ERA_RECIPE[key], "observed": value}
        for key, value in supplied.items()
        if value != SCORE_ERA_RECIPE[key]
    }
    if mismatches:
        raise ValueError(f"score-era recipe mismatch: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-spec", type=Path, help="JSON member specification")
    parser.add_argument("--data-csv", type=Path, help="Training/member dataframe CSV")
    parser.add_argument("--blind-csv", type=Path, help="Blind/test molecule CSV")
    parser.add_argument("--output-dir", type=Path, help="Artifact directory; defaults to SM_MODEL_DIR")
    parser.add_argument("--targets", help="Comma-separated target columns; overrides the spec")
    parser.add_argument("--task-weights", help="Comma-separated positional task weights; overrides the spec")
    parser.add_argument("--member-name", help="Member identifier; overrides the spec")
    parser.add_argument("--smiles-column", help="SMILES column; overrides the spec")
    parser.add_argument("--id-column", help="Stable ID column; overrides the spec")
    parser.add_argument("--num-workers", type=int, help="Dataloader workers (runtime-only, not a model knob)")
    parser.add_argument("--training-image-uri", help="ECR image URI for provenance logging")
    parser.add_argument("--training-image-digest", help="ECR sha256 digest for provenance logging")
    parser.add_argument("--sagemaker-instance-type", help="SageMaker instance type for provenance logging")
    parser.add_argument(
        "--expected-runner-sha256",
        help="Expected 64-hex SHA-256 of this runner; a mismatch aborts before training",
    )
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Permit training outside the exact 34689b18 package lock; recorded as a fidelity gap",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, reproduce scaffold folds, and write manifests without importing/training Chemprop",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> RunConfig:
    channel = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    default_spec = channel / "member_spec.json"
    spec_path = args.member_spec
    if spec_path is None and default_spec.exists():
        spec_path = default_spec

    spec: dict[str, Any] = {}
    spec_dir = Path.cwd()
    if spec_path is not None:
        spec_path = spec_path.expanduser().resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(f"member spec not found: {spec_path}")
        spec = json.loads(spec_path.read_text())
        if not isinstance(spec, dict):
            raise ValueError("member spec must be a JSON object")
        spec_dir = spec_path.parent
    _validate_locked_recipe(spec)

    data_csv = args.data_csv or _path_from_spec(spec.get("data_csv"), spec_dir)
    blind_csv = args.blind_csv or _path_from_spec(spec.get("blind_csv"), spec_dir)
    if data_csv is None and (channel / "data.csv").exists():
        data_csv = channel / "data.csv"
    if blind_csv is None and (channel / "blind.csv").exists():
        blind_csv = channel / "blind.csv"
    if data_csv is None or blind_csv is None:
        raise ValueError("data_csv and blind_csv are required (CLI, member spec, or SageMaker channel)")
    data_csv, blind_csv = data_csv.expanduser().resolve(), blind_csv.expanduser().resolve()

    output_dir = args.output_dir or _path_from_spec(spec.get("output_dir"), spec_dir)
    if output_dir is None:
        output_dir = Path(os.environ.get("SM_MODEL_DIR", "./briford_model"))
    output_dir = output_dir.expanduser().resolve()

    targets = parse_csv_strings(args.targets if args.targets is not None else spec.get("targets"), "targets")
    weights = parse_weights(
        args.task_weights if args.task_weights is not None else spec.get("task_weights")
    )
    if not targets:
        raise ValueError("targets are required")
    if len(targets) != len(weights):
        raise ValueError(
            f"targets/task_weights length mismatch: {len(targets)} targets, {len(weights)} weights"
        )

    scored = parse_csv_strings(spec.get("scored_targets", list(targets)), "scored_targets")
    missing_scored = sorted(set(scored) - set(targets))
    if missing_scored:
        raise ValueError(f"scored_targets are not in targets: {missing_scored}")

    interval_columns = spec.get("interval_columns", {}) or {}
    if not isinstance(interval_columns, dict):
        raise ValueError("interval_columns must be an object keyed by target")
    for target, columns in interval_columns.items():
        if target not in targets or not isinstance(columns, dict) or set(columns) != {"low", "high"}:
            raise ValueError(
                "each interval_columns entry must target a modeled column and contain exactly low/high"
            )

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(spec.get("num_workers", min(os.cpu_count() or 4, 8)))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    sealed = spec.get("sealed_input_manifest")
    sealed_path = None
    sealed_sha = None
    sealed_member = None
    if sealed is not None:
        if not isinstance(sealed, dict) or set(sealed) != {"path", "sha256", "member_key"}:
            raise ValueError("sealed_input_manifest must contain exactly path, sha256, member_key")
        sealed_path = _path_from_spec(str(sealed["path"]), spec_dir)
        sealed_sha = str(sealed["sha256"])
        sealed_member = str(sealed["member_key"])

    image_uri = args.training_image_uri or spec.get("training_image_uri") or os.environ.get(
        "BRIFORD_TRAINING_IMAGE_URI"
    )
    digest = _image_digest(
        image_uri,
        args.training_image_digest
        or spec.get("training_image_digest")
        or os.environ.get("BRIFORD_TRAINING_IMAGE_DIGEST"),
    )
    if digest is not None and not (
        digest.startswith("sha256:") and len(digest.removeprefix("sha256:")) == 64
    ):
        raise ValueError("training_image_digest must be sha256:<64 lowercase/uppercase hex characters>")
    if digest is not None:
        try:
            int(digest.removeprefix("sha256:"), 16)
        except ValueError as error:
            raise ValueError("training_image_digest is not hexadecimal") from error

    expected_runner_sha256 = args.expected_runner_sha256
    if expected_runner_sha256 is not None:
        if re.fullmatch(r"[0-9a-fA-F]{64}", expected_runner_sha256) is None:
            raise ValueError("expected_runner_sha256 must be exactly 64 hexadecimal characters")
        expected_runner_sha256 = expected_runner_sha256.lower()

    return RunConfig(
        member_name=args.member_name or spec.get("member_name", "unnamed_briford_member"),
        member_spec_path=spec_path,
        data_csv=data_csv,
        blind_csv=blind_csv,
        output_dir=output_dir,
        smiles_column=args.smiles_column or spec.get("smiles_column", "smiles"),
        id_column=args.id_column or spec.get("id_column", "Molecule_Name"),
        targets=targets,
        task_weights=weights,
        scored_targets=scored,
        interval_columns=interval_columns,
        expected_data_sha256=spec.get("expected_data_sha256"),
        expected_blind_sha256=spec.get("expected_blind_sha256"),
        expected_rows=int(spec["expected_rows"]) if spec.get("expected_rows") is not None else None,
        sealed_input_manifest_path=sealed_path,
        sealed_input_manifest_sha256=sealed_sha,
        sealed_input_member_key=sealed_member,
        num_workers=num_workers,
        training_image_uri=image_uri,
        training_image_digest=digest,
        sagemaker_instance_type=args.sagemaker_instance_type
        or spec.get("sagemaker_instance_type")
        or os.environ.get("BRIFORD_SAGEMAKER_INSTANCE_TYPE"),
        expected_runner_sha256=expected_runner_sha256,
        allow_version_drift=bool(args.allow_version_drift or spec.get("allow_version_drift", False)),
        dry_run=bool(args.dry_run),
    )


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"empty CSV: {path}") from error


def _require_unique_header(path: Path) -> list[str]:
    header = _csv_header(path)
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate CSV column(s) in {path}: {duplicates}")
    return header


def _numeric_targets(frame: pd.DataFrame, targets: Sequence[str], path: Path) -> pd.DataFrame:
    converted = frame.copy()
    for target in targets:
        original = converted[target]
        numeric = pd.to_numeric(original, errors="coerce")
        nonblank = original.notna() & original.astype(str).str.strip().ne("")
        bad = nonblank & numeric.isna()
        if bad.any():
            examples = original.loc[bad].astype(str).head(5).tolist()
            raise ValueError(f"non-numeric values in {path} column {target}: {examples}")
        finite = numeric.dropna().map(math.isfinite)
        if not finite.all():
            raise ValueError(f"non-finite values in {path} column {target}")
        converted[target] = numeric
    return converted


def get_scaffolds(smiles: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Workbench get_scaffold_groups semantics plus explicit invalid-row reporting."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.error")
    valid: list[int] = []
    scaffold_values: list[str] = []
    invalid: list[str] = []
    for index, value in enumerate(smiles):
        if not isinstance(value, str) or not value.strip():
            invalid.append(str(index))
            continue
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            invalid.append(str(index))
            continue
        try:
            scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule))
        except Exception:
            scaffold = ""
        valid.append(index)
        scaffold_values.append(scaffold)

    scaffold_to_group: dict[str, int] = {}
    groups: list[int] = []
    for scaffold in scaffold_values:
        if scaffold not in scaffold_to_group:
            scaffold_to_group[scaffold] = len(scaffold_to_group)
        groups.append(scaffold_to_group[scaffold])
    return np.asarray(valid, dtype=int), np.asarray(groups, dtype=int), scaffold_values


def group_folds(groups: np.ndarray, n_splits: int = 5, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact Workbench random-group-order/greedy-lightest fold partition."""
    unique_groups, counts = np.unique(groups, return_counts=True)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"cannot make {n_splits} scaffold folds from only {len(unique_groups)} unique scaffolds"
        )
    load = np.zeros(n_splits, dtype=np.int64)
    group_to_fold: dict[int, int] = {}
    for index in np.random.default_rng(random_state).permutation(len(unique_groups)):
        fold = int(np.argmin(load))
        group_to_fold[int(unique_groups[index])] = fold
        load[fold] += counts[index]
    fold_of_row = np.asarray([group_to_fold[int(group)] for group in groups], dtype=int)
    indices = np.arange(len(groups))
    return [(indices[fold_of_row != fold], indices[fold_of_row == fold]) for fold in range(n_splits)]


def _fold_assignment(folds: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int) -> np.ndarray:
    assignment = np.full(n_rows, -1, dtype=int)
    for fold, (_, validation) in enumerate(folds):
        if (assignment[validation] != -1).any():
            raise AssertionError("a row was assigned to multiple validation folds")
        assignment[validation] = fold
    if (assignment < 0).any():
        raise AssertionError("not every row was assigned to one validation fold")
    return assignment


def validate_inputs(config: RunConfig) -> ValidatedInputs:
    for path in (config.data_csv, config.blind_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_unique_header(path)

    data_hash, blind_hash = sha256(config.data_csv), sha256(config.blind_csv)
    if config.expected_data_sha256 and data_hash != config.expected_data_sha256:
        raise ValueError(
            f"data SHA mismatch: expected {config.expected_data_sha256}, observed {data_hash}"
        )
    if config.expected_blind_sha256 and blind_hash != config.expected_blind_sha256:
        raise ValueError(
            f"blind SHA mismatch: expected {config.expected_blind_sha256}, observed {blind_hash}"
        )

    sealed_manifest_record = None
    sealed_member_record = None
    if config.sealed_input_manifest_path is not None:
        if not config.sealed_input_manifest_path.is_file():
            raise FileNotFoundError(config.sealed_input_manifest_path)
        observed_manifest_sha = sha256(config.sealed_input_manifest_path)
        if observed_manifest_sha != config.sealed_input_manifest_sha256:
            raise ValueError(
                "sealed input manifest SHA mismatch: "
                f"expected {config.sealed_input_manifest_sha256}, observed {observed_manifest_sha}"
            )
        sealed_document = json.loads(config.sealed_input_manifest_path.read_text())
        try:
            sealed_member_record = sealed_document["members"][config.sealed_input_member_key]
        except KeyError as error:
            raise ValueError(
                f"member {config.sealed_input_member_key!r} absent from sealed input manifest"
            ) from error
        if sealed_document.get("schema") != "openadmet_briford_corrected_full_data_v2":
            raise ValueError("sealed input manifest is not the corrected Workbench v2 schema")
        if sealed_document.get("status") != "SEALED":
            raise ValueError("sealed input manifest is not frozen")
        if sealed_document.get("unresolved_discrepancies") != []:
            raise ValueError("sealed input manifest contains unresolved discrepancies")
        sealed_manifest_record = {
            "path": str(config.sealed_input_manifest_path),
            "sha256": observed_manifest_sha,
            "member_key": config.sealed_input_member_key,
            "schema": sealed_document.get("schema"),
            "status": sealed_document.get("status"),
            "scope": sealed_document.get("scope"),
            "workbench_code_pins": sealed_document.get("workbench_code_pins"),
        }
        sealed_output = sealed_member_record["output"]
        if sealed_output["sha256"] != data_hash:
            raise ValueError("training CSV does not match the sealed member output SHA")
        if tuple(sealed_member_record["target_columns_in_exact_published_order"]) != config.targets:
            raise ValueError("target order does not match the sealed member manifest")
        if tuple(float(value) for value in sealed_member_record["task_weights_in_exact_target_order"]) != config.task_weights:
            raise ValueError("task weights do not match the sealed member manifest")

    # Match the score-era template's explicit pandas Python CSV engine.
    raw_train = pd.read_csv(config.data_csv, engine="python")
    raw_blind = pd.read_csv(config.blind_csv, engine="python")
    if config.expected_rows is not None and len(raw_train) != config.expected_rows:
        raise ValueError(f"expected {config.expected_rows} training rows, observed {len(raw_train)}")
    if sealed_member_record is not None and len(raw_train) != int(sealed_member_record["output"]["rows"]):
        raise ValueError("training row count does not match the sealed member manifest")

    required_train = [config.id_column, config.smiles_column, *config.targets]
    interval_names = [
        name for columns in config.interval_columns.values() for name in columns.values()
    ]
    required_train.extend(interval_names)
    required_blind = [config.id_column, config.smiles_column]
    missing_train = [name for name in required_train if name not in raw_train.columns]
    missing_blind = [name for name in required_blind if name not in raw_blind.columns]
    if missing_train or missing_blind:
        raise ValueError(
            f"missing columns: training={missing_train or 'none'}, blind={missing_blind or 'none'}"
        )

    for label, frame in (("training", raw_train), ("blind", raw_blind)):
        if frame[config.id_column].isna().any():
            raise ValueError(f"{label} IDs contain missing values")
        if frame[config.id_column].astype(str).duplicated().any():
            duplicated = frame.loc[
                frame[config.id_column].astype(str).duplicated(keep=False), config.id_column
            ].astype(str).head(5).tolist()
            raise ValueError(f"{label} IDs are not unique: {duplicated}")

    train = _numeric_targets(raw_train, config.targets, config.data_csv)
    for interval in interval_names:
        train = _numeric_targets(train, [interval], config.data_csv)
    source_rows = np.arange(len(train), dtype=int)

    missing_smiles = train[config.smiles_column].isna() | train[config.smiles_column].astype(str).str.strip().eq("")
    no_targets = train[list(config.targets)].isna().all(axis=1)
    keep = ~(missing_smiles | no_targets)
    filtered = train.loc[keep].reset_index(drop=True)
    filtered_source_rows = source_rows[keep.to_numpy()]

    valid_positions, groups, scaffold_values = get_scaffolds(filtered[config.smiles_column].tolist())
    invalid_after_basic = np.setdiff1d(np.arange(len(filtered)), valid_positions)
    clean = filtered.iloc[valid_positions].reset_index(drop=True)
    clean_source_rows = filtered_source_rows[valid_positions]
    if clean.empty:
        raise ValueError("no valid labeled training molecules remain")

    # Workbench filters invalid molecule datapoints before making one joint scaffold split.
    folds = group_folds(groups, n_splits=5, random_state=42)
    assignment = _fold_assignment(folds, len(clean))
    for fold, (train_idx, val_idx) in enumerate(folds):
        if len(val_idx) < 2:
            raise ValueError(f"fold {fold} has only {len(val_idx)} validation row(s); batch norm needs at least 2")
        empty_train_targets = [
            target for target in config.targets if clean.iloc[train_idx][target].notna().sum() == 0
        ]
        if empty_train_targets:
            raise ValueError(f"fold {fold} has no training labels for {empty_train_targets}")

    blind_valid_positions, _, blind_scaffolds = get_scaffolds(raw_blind[config.smiles_column].tolist())
    if len(blind_valid_positions) == 0:
        raise ValueError("blind CSV contains no valid molecules")

    train_ids = set(clean[config.id_column].astype(str))
    blind_ids = set(raw_blind[config.id_column].astype(str))
    id_overlap = sorted(train_ids & blind_ids)
    train_smiles = set(clean[config.smiles_column].astype(str))
    valid_blind_smiles = set(
        raw_blind.iloc[blind_valid_positions][config.smiles_column].astype(str)
    )
    smiles_overlap = sorted(train_smiles & valid_blind_smiles)

    target_counts = {target: int(clean[target].notna().sum()) for target in config.targets}
    if sealed_member_record is not None and target_counts != sealed_member_record["label_counts"]:
        raise ValueError(
            "post-validation target counts do not match the sealed member manifest: "
            f"observed={target_counts}, expected={sealed_member_record['label_counts']}"
        )
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        fold_rows.append(
            {
                "fold": fold,
                "training_seed": 42 + fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "validation_scaffolds": int(len(set(groups[val_idx].tolist()))),
                "validation_target_counts": {
                    target: int(clean.iloc[val_idx][target].notna().sum())
                    for target in config.targets
                },
            }
        )

    versions = package_versions()
    warnings = []
    if invalid_after_basic.size:
        warnings.append(
            f"Workbench-compatible filtering drops {len(invalid_after_basic)} invalid training SMILES"
        )
    if len(blind_valid_positions) != len(raw_blind):
        warnings.append(
            f"{len(raw_blind) - len(blind_valid_positions)} invalid blind SMILES will receive NaN predictions"
        )
    if id_overlap:
        warnings.append(f"training/blind ID overlap: {len(id_overlap)}")
    if smiles_overlap:
        warnings.append(f"exact training/blind SMILES overlap: {len(smiles_overlap)}")
    drifted = [name for name, item in versions.items() if not item["exact_match"]]
    if drifted:
        warnings.append(f"local package drift from the score-era lock: {drifted}")
    if not config.training_image_digest:
        warnings.append("training image digest is not supplied; a real parity run must log/pin it")

    report = {
        "status": "validated",
        "validated_at_utc": utc_now(),
        "member_name": config.member_name,
        "inputs": {
            "data_csv": str(config.data_csv),
            "data_sha256": data_hash,
            "blind_csv": str(config.blind_csv),
            "blind_sha256": blind_hash,
            "member_spec": str(config.member_spec_path) if config.member_spec_path else None,
            "member_spec_sha256": sha256(config.member_spec_path) if config.member_spec_path else None,
            "sealed_input_manifest": sealed_manifest_record,
        },
        "columns": {
            "id": config.id_column,
            "smiles": config.smiles_column,
            "targets_in_order": list(config.targets),
            "task_weights_in_order": list(config.task_weights),
            "scored_targets": list(config.scored_targets),
        },
        "row_filtering": {
            "input_training_rows": int(len(raw_train)),
            "missing_smiles_rows": int(missing_smiles.sum()),
            "all_targets_missing_rows": int(no_targets.sum()),
            "invalid_smiles_after_basic_filter_rows": int(len(invalid_after_basic)),
            "modeled_training_rows": int(len(clean)),
            "blind_rows": int(len(raw_blind)),
            "valid_blind_rows": int(len(blind_valid_positions)),
        },
        "target_label_counts": target_counts,
        "joint_scaffold_split": {
            "algorithm": "get_scaffold_groups + seeded random group order + greedy lightest fold",
            "seed": 42,
            "folds": 5,
            "unique_scaffolds": int(len(set(scaffold_values))),
            "fold_assignment_sha256": sequence_sha256(assignment.tolist()),
            "modeled_id_order_sha256": sequence_sha256(clean[config.id_column].astype(str).tolist()),
            "rows": fold_rows,
            "every_row_exactly_one_validation_fold": True,
        },
        "leakage_observations_not_used_for_splitting": {
            "training_blind_id_overlap_count": len(id_overlap),
            "training_blind_id_overlap_examples": id_overlap[:10],
            "training_blind_exact_smiles_overlap_count": len(smiles_overlap),
            "training_blind_exact_smiles_overlap_examples": smiles_overlap[:10],
        },
        "score_era_package_contract": versions,
        "warnings": warnings,
    }

    # Attach these internal alignment fields only after all checks; they never become targets.
    clean = clean.copy()
    clean["_briford_source_row"] = clean_source_rows
    clean["_briford_scaffold"] = scaffold_values
    clean["_briford_fold"] = assignment
    return ValidatedInputs(
        train=clean,
        blind=raw_blind.copy(),
        train_valid_source_rows=clean_source_rows,
        blind_valid_positions=blind_valid_positions,
        scaffolds=groups,
        folds=folds,
        report=report,
    )


def write_validation_artifacts(config: RunConfig, inputs: ValidatedInputs) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / OUTPUT_NAMES["validation"], inputs.report)
    split_columns = [
        "_briford_source_row",
        config.id_column,
        config.smiles_column,
        "_briford_scaffold",
        "_briford_fold",
    ]
    inputs.train[split_columns].to_csv(config.output_dir / OUTPUT_NAMES["splits"], index=False)


def initial_manifest(
    config: RunConfig,
    inputs: ValidatedInputs,
    runner_attestation: dict[str, Any],
) -> dict[str, Any]:
    versions = inputs.report["score_era_package_contract"]
    return {
        "artifact_type": "standalone_briford_workbench_chemprop_member",
        "parity_scope": "recipe_level_only_not_exact_or_bitwise",
        "status": "dry_run_validated" if config.dry_run else "training",
        "started_at_utc": utc_now(),
        "member_name": config.member_name,
        "public_provenance": {
            "repository": SOURCE_REPOSITORY,
            "model_recipe_commit": MODEL_RECIPE_COMMIT,
            "score_era_commit": SCORE_ERA_COMMIT,
            "composite_union_pin": COMPOSITE_UNION_PIN,
            "composite_final_recipe_pin": COMPOSITE_FINAL_RECIPE_PIN,
            "source_blobs": PROVENANCE_BLOBS,
        },
        "executed_runner": runner_attestation,
        "recipe": SCORE_ERA_RECIPE,
        "member": {
            "targets_in_order": list(config.targets),
            "task_weights_in_order": list(config.task_weights),
            "scored_targets": list(config.scored_targets),
            "smiles_column": config.smiles_column,
            "id_column": config.id_column,
        },
        "inputs": inputs.report["inputs"],
        "joint_scaffold_split": inputs.report["joint_scaffold_split"],
        "environment": {
            "python": platform.python_version(),
            "score_era_python_major_minor": list(SCORE_ERA_PYTHON_MAJOR_MINOR),
            "python_major_minor_matches_score_era": sys.version_info[:2]
            == SCORE_ERA_PYTHON_MAJOR_MINOR,
            "platform": platform.platform(),
            "packages": versions,
            "exact_score_era_package_match": all(item["exact_match"] for item in versions.values()),
            "version_drift_override": config.allow_version_drift,
            "sagemaker_training_job_name": os.environ.get("TRAINING_JOB_NAME"),
            "aws_region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
            "training_image_uri": config.training_image_uri,
            "training_image_digest": config.training_image_digest,
            "sagemaker_instance_type": config.sagemaker_instance_type,
            "score_era_default_instance_type": SCORE_ERA_DEFAULT_INSTANCE,
            "instance_type_matches_score_era_default": config.sagemaker_instance_type
            == SCORE_ERA_DEFAULT_INSTANCE,
            "score_era_image_reference": SCORE_ERA_MUTABLE_IMAGE,
            "score_era_image_digest_published": False,
        },
        "validation_artifact": str(config.output_dir / OUTPUT_NAMES["validation"]),
        "split_assignment_artifact": str(config.output_dir / OUTPUT_NAMES["splits"]),
        "warnings": list(inputs.report["warnings"]),
        "known_fidelity_gaps": KNOWN_FIDELITY_GAPS,
        "submission_action": "none",
    }


def enforce_training_environment(config: RunConfig, manifest: dict[str, Any]) -> None:
    drift = {
        name: details
        for name, details in manifest["environment"]["packages"].items()
        if not details["exact_match"]
    }
    python_drift = not manifest["environment"]["python_major_minor_matches_score_era"]
    if (drift or python_drift) and not config.allow_version_drift:
        raise RuntimeError(
            "training Python/packages differ from the public score-era image contract; "
            "use the pinned image or explicitly record --allow-version-drift: "
            f"python_drift={python_drift}, packages={drift}"
        )
    if not config.training_image_digest:
        manifest["warnings"].append(
            "training proceeded without an ECR digest; environment identity is not fully sealed"
        )


def enforce_cuda_for_instance(instance_type: str | None, cuda_available: bool) -> None:
    """Refuse a silent CPU fallback on SageMaker GPU instance families."""
    if instance_type and instance_type.lower().startswith("ml.g") and not cuda_available:
        raise RuntimeError(
            f"CUDA is unavailable on declared GPU instance type {instance_type}; "
            "refusing CPU fallback"
        )


def checkpoint_best_epoch(checkpoint_payload: dict[str, Any]) -> int:
    """Return a validated zero-based best epoch from a Lightning checkpoint."""
    if "epoch" not in checkpoint_payload:
        raise RuntimeError("best-val checkpoint does not record an epoch")
    try:
        epoch = int(checkpoint_payload["epoch"])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("best-val checkpoint epoch is not an integer") from error
    if not 0 <= epoch < SCORE_ERA_RECIPE["max_epochs"]:
        raise RuntimeError(
            "best-val checkpoint epoch is outside the configured training range: "
            f"epoch={epoch}, max_epochs={SCORE_ERA_RECIPE['max_epochs']}"
        )
    return epoch


def load_training_stack():
    """Deferred imports: dry-run and input-validation tests never load training libraries."""
    import torch
    from chemprop import data, models, nn
    from chemprop.nn.metrics import MAE
    from lightning import pytorch as pl

    return torch, data, models, nn, MAE, pl


def _molecule_datapoints(data_module, smiles: Sequence[str], targets: np.ndarray | None = None):
    datapoints = []
    for index, smi in enumerate(smiles):
        y = targets[index].tolist() if targets is not None else None
        datapoints.append(data_module.MoleculeDatapoint.from_smi(smi, y=y))
    return datapoints


def _safe_batch_size(dataset_len: int, batch_size: int) -> int:
    if dataset_len <= 0:
        raise ValueError("cannot batch an empty dataset")
    resolved = min(dataset_len, batch_size)
    return resolved + 1 if dataset_len % resolved == 1 else resolved


def _loader_kwargs(num_workers: int, pin_memory: bool, persistent: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs.update(persistent_workers=persistent, prefetch_factor=2)
    return kwargs


def _predict_model(model, frame: pd.DataFrame, config: RunConfig, stack) -> np.ndarray:
    torch, data_module, _, _, _, _ = stack
    datapoints = _molecule_datapoints(data_module, frame[config.smiles_column].astype(str).tolist())
    dataset = data_module.MoleculeDataset(datapoints)
    dataset.cache = True
    device = next(model.parameters()).device
    loader = data_module.build_dataloader(
        dataset,
        batch_size=_safe_batch_size(len(dataset), SCORE_ERA_RECIPE["batch_size"]),
        shuffle=False,
        **_loader_kwargs(config.num_workers, device.type == "cuda", persistent=False),
    )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            bmg, v_d, x_d, *_ = batch
            bmg.to(device)
            v_d = None if v_d is None else v_d.to(device)
            x_d = None if x_d is None else x_d.to(device)
            chunks.append(model(bmg, v_d, x_d).detach().cpu().numpy())
    predictions = np.concatenate(chunks, axis=0)
    if predictions.ndim == 3 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(axis=1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return predictions


def _build_model(n_targets: int, weights: np.ndarray, output_transform, stack):
    _, _, models, nn_module, MAE, _ = stack
    message_passing = nn_module.BondMessagePassing(
        d_h=SCORE_ERA_RECIPE["hidden_dim"],
        depth=SCORE_ERA_RECIPE["depth"],
        dropout=SCORE_ERA_RECIPE["dropout"],
    )
    aggregation = nn_module.NormAggregation(norm=SCORE_ERA_RECIPE["aggregation_norm"])
    predictor = nn_module.RegressionFFN(
        input_dim=SCORE_ERA_RECIPE["hidden_dim"],
        hidden_dim=SCORE_ERA_RECIPE["ffn_hidden_dim"],
        n_layers=SCORE_ERA_RECIPE["ffn_num_layers"],
        dropout=SCORE_ERA_RECIPE["dropout"],
        n_tasks=n_targets,
        output_transform=output_transform,
        criterion=MAE(task_weights=weights),
    )
    return models.MPNN(
        message_passing=message_passing,
        agg=aggregation,
        predictor=predictor,
        batch_norm=True,
        metrics=None,
        warmup_epochs=SCORE_ERA_RECIPE["warmup_epochs"],
        init_lr=SCORE_ERA_RECIPE["init_lr"],
        max_lr=SCORE_ERA_RECIPE["max_lr"],
        final_lr=SCORE_ERA_RECIPE["final_lr"],
    )


def _metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    keep = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[keep], prediction[keep]
    if not len(y):
        return {"n": 0}
    error = prediction - y
    centered = y - y.mean()
    denominator = float(np.square(centered).sum())
    r2 = None if denominator == 0 else 1.0 - float(np.square(error).sum()) / denominator
    pearson = None
    spearman = None
    if len(y) >= 2 and np.std(y) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(y, prediction)[0, 1])
        spearman = float(pd.Series(y).rank(method="average").corr(pd.Series(prediction).rank(method="average")))
    return {
        "n": int(len(y)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def _soft_threshold_rae(
    frame: pd.DataFrame, prediction: np.ndarray, target: str, columns: dict[str, str]
) -> float | None:
    keep = frame[target].notna().to_numpy() & np.isfinite(prediction)
    if not keep.any():
        return None
    y = frame.loc[keep, target].to_numpy(float)
    low = frame.loc[keep, columns["low"]].to_numpy(float)
    high = frame.loc[keep, columns["high"]].to_numpy(float)
    center = float(y.mean())
    numerator = np.clip(prediction[keep] - high, 0, None) + np.clip(low - prediction[keep], 0, None)
    denominator = np.clip(center - high, 0, None) + np.clip(low - center, 0, None)
    total = float(denominator.sum())
    return None if total == 0 else float(numerator.sum() / total)


def _metrics_for_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    config: RunConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, target in enumerate(config.targets):
        values = _metric_values(frame[target].to_numpy(float), predictions[:, index])
        if target in config.interval_columns:
            values["soft_threshold_rae"] = _soft_threshold_rae(
                frame, predictions[:, index], target, config.interval_columns[target]
            )
        result[target] = values
    scored_mae = [result[target].get("mae") for target in config.scored_targets]
    scored_mae = [value for value in scored_mae if value is not None]
    scored_st = [result[target].get("soft_threshold_rae") for target in config.scored_targets]
    scored_st = [value for value in scored_st if value is not None]
    return {
        "targets": result,
        "macro_mae_scored_targets": float(np.mean(scored_mae)) if scored_mae else None,
        "macro_soft_threshold_rae_scored_targets": float(np.mean(scored_st)) if scored_st else None,
    }


def train(config: RunConfig, inputs: ValidatedInputs, manifest: dict[str, Any]) -> None:
    enforce_training_environment(config, manifest)
    stack = load_training_stack()
    torch, data_module, models, nn_module, _, pl = stack
    torch.set_float32_matmul_precision("medium")
    cuda_available = bool(torch.cuda.is_available())
    manifest["environment"]["training_runtime"] = {
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count() if cuda_available else 0)
        ],
    }
    enforce_cuda_for_instance(config.sagemaker_instance_type, cuda_available)

    train_frame = inputs.train
    blind_valid = inputs.blind.iloc[inputs.blind_valid_positions].reset_index(drop=True)
    all_targets = train_frame[list(config.targets)].to_numpy(dtype=np.float32)
    n_rows, n_targets = all_targets.shape
    oof = np.full((n_rows, n_targets), np.nan, dtype=np.float64)
    raw_blind_by_fold: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    checkpoints_dir = config.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(inputs.folds):
        fold_started = time.time()
        seed = SCORE_ERA_RECIPE["split_seed"] + fold

        train_dps = _molecule_datapoints(
            data_module,
            train_frame.iloc[train_idx][config.smiles_column].astype(str).tolist(),
            all_targets[train_idx],
        )
        val_dps = _molecule_datapoints(
            data_module,
            train_frame.iloc[val_idx][config.smiles_column].astype(str).tolist(),
            all_targets[val_idx],
        )
        train_dataset = data_module.MoleculeDataset(train_dps)
        val_dataset = data_module.MoleculeDataset(val_dps)
        empty = np.flatnonzero(np.isnan(train_dataset.Y).all(axis=0))
        if empty.size:
            raise ValueError(f"fold {fold} has no training labels for {[config.targets[i] for i in empty]}")
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
        output_transform = nn_module.UnscaleTransform.from_standard_scaler(scaler)
        train_dataset.cache = True
        val_dataset.cache = True

        # Workbench set pin_memory=True for both train and validation loaders.
        loader_kwargs = _loader_kwargs(config.num_workers, True, persistent=True)
        train_loader = data_module.build_dataloader(
            train_dataset,
            batch_size=SCORE_ERA_RECIPE["batch_size"],
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = data_module.build_dataloader(
            val_dataset,
            batch_size=_safe_batch_size(len(val_dataset), SCORE_ERA_RECIPE["batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )

        # Match Workbench's placement: dataloaders are constructed first, then the fold
        # seed is set immediately before random model initialization.
        pl.seed_everything(seed)
        model = _build_model(
            n_targets,
            np.asarray(config.task_weights, dtype=np.float32),
            output_transform,
            stack,
        )
        fold_dir = checkpoints_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=str(fold_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=SCORE_ERA_RECIPE["patience"],
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=SCORE_ERA_RECIPE["max_epochs"],
            precision=SCORE_ERA_RECIPE["precision"],
            logger=False,
            enable_checkpointing=True,
            callbacks=[early_stop, checkpoint],
        )
        trainer.fit(model, train_loader, val_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError(f"fold {fold} produced no best-val checkpoint")
        saved = torch.load(checkpoint.best_model_path, weights_only=False, map_location="cpu")
        best_epoch = checkpoint_best_epoch(saved)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        # Lightning may leave the module on GPU; use that device for the exact serving-style
        # fp32 forward pass, then save a CPU-portable Chemprop MPNN file.
        val_predictions = _predict_model(model, train_frame.iloc[val_idx].reset_index(drop=True), config, stack)
        oof[val_idx] = val_predictions
        blind_predictions = _predict_model(model, blind_valid, config, stack)
        raw_blind_by_fold.append(blind_predictions)
        model_path = fold_dir / "chemprop_model.pt"
        models.save_model(str(model_path), model.cpu())

        best_score = checkpoint.best_model_score
        fold_record = {
            "fold": fold,
            "seed": seed,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "best_checkpoint": str(Path(checkpoint.best_model_path).resolve()),
            "best_checkpoint_sha256": sha256(Path(checkpoint.best_model_path)),
            "portable_model": str(model_path.resolve()),
            "portable_model_sha256": sha256(model_path),
            "best_val_loss": float(best_score.detach().cpu()) if best_score is not None else None,
            "best_epoch": best_epoch,
            "epochs_completed": int(trainer.current_epoch + 1),
            "elapsed_seconds": time.time() - fold_started,
            "metrics": _metrics_for_frame(
                train_frame.iloc[val_idx].reset_index(drop=True), val_predictions, config
            ),
        }
        fold_metrics.append(fold_record)
        checkpoint_records.append(
            {key: fold_record[key] for key in ("fold", "seed", "best_checkpoint", "best_checkpoint_sha256", "portable_model", "portable_model_sha256")}
        )

    if np.isnan(oof).any():
        raise AssertionError("pooled OOF matrix contains missing predictions")
    blind_stack = np.stack(raw_blind_by_fold, axis=0)
    if blind_stack.shape != (5, len(blind_valid), n_targets):
        raise AssertionError(f"unexpected blind prediction stack shape {blind_stack.shape}")

    oof_output = train_frame[
        ["_briford_source_row", config.id_column, config.smiles_column, "_briford_scaffold", "_briford_fold", *config.targets]
    ].copy()
    for target_index, target in enumerate(config.targets):
        oof_output[f"{target}_pred"] = oof[:, target_index]
    oof_path = config.output_dir / OUTPUT_NAMES["oof"]
    oof_output.to_csv(oof_path, index=False)

    raw_frames = []
    for fold in range(5):
        part = blind_valid[[config.id_column, config.smiles_column]].copy()
        part.insert(0, "_briford_blind_row", inputs.blind_valid_positions)
        part.insert(0, "fold", fold)
        for target_index, target in enumerate(config.targets):
            part[f"{target}_pred"] = blind_stack[fold, :, target_index]
        raw_frames.append(part)
    blind_raw_path = config.output_dir / OUTPUT_NAMES["blind_raw"]
    pd.concat(raw_frames, ignore_index=True).to_csv(blind_raw_path, index=False)

    blind_mean = inputs.blind[[config.id_column, config.smiles_column]].copy()
    for target_index, target in enumerate(config.targets):
        mean = np.full(len(inputs.blind), np.nan, dtype=float)
        std = np.full(len(inputs.blind), np.nan, dtype=float)
        mean[inputs.blind_valid_positions] = blind_stack[:, :, target_index].mean(axis=0)
        std[inputs.blind_valid_positions] = blind_stack[:, :, target_index].std(axis=0)
        blind_mean[f"{target}_pred"] = mean
        blind_mean[f"{target}_pred_std"] = std
    blind_mean_path = config.output_dir / OUTPUT_NAMES["blind_mean"]
    blind_mean.to_csv(blind_mean_path, index=False)

    metrics = {
        "member_name": config.member_name,
        "estimator": "single-held-out-fold prediction for every OOF row",
        "pooled_oof": _metrics_for_frame(train_frame, oof, config),
        "folds": fold_metrics,
    }
    metrics_path = config.output_dir / OUTPUT_NAMES["metrics"]
    write_json(metrics_path, metrics)

    outputs = [
        config.output_dir / OUTPUT_NAMES["validation"],
        config.output_dir / OUTPUT_NAMES["splits"],
        oof_path,
        blind_raw_path,
        blind_mean_path,
        metrics_path,
    ]
    manifest.update(
        status="complete",
        completed_at_utc=utc_now(),
        checkpoints=checkpoint_records,
        outputs=[
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        prediction_semantics={
            "oof": "exactly one held-out scaffold-fold model per training row",
            "blind_raw_shape": list(blind_stack.shape),
            "blind_reduction": "unweighted arithmetic mean of five raw fold predictions",
            "blind_std_ddof": 0,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    runner_attestation = attest_and_archive_runner(config)
    inputs = validate_inputs(config)
    write_validation_artifacts(config, inputs)
    manifest = initial_manifest(config, inputs, runner_attestation)
    manifest_path = config.output_dir / OUTPUT_NAMES["manifest"]
    write_json(manifest_path, manifest)
    if config.dry_run:
        print(json.dumps({
            "status": manifest["status"],
            "member": config.member_name,
            "modeled_rows": len(inputs.train),
            "blind_rows": len(inputs.blind),
            "fold_sizes": [len(validation) for _, validation in inputs.folds],
            "fold_assignment_sha256": inputs.report["joint_scaffold_split"]["fold_assignment_sha256"],
            "manifest": str(manifest_path),
            "training_started": False,
        }, indent=2))
        return 0

    try:
        train(config, inputs, manifest)
    except Exception as error:
        manifest.update(
            status="failed",
            failed_at_utc=utc_now(),
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "member": config.member_name,
        "manifest": str(manifest_path),
        "oof_predictions": str(config.output_dir / OUTPUT_NAMES["oof"]),
        "blind_predictions": str(config.output_dir / OUTPUT_NAMES["blind_mean"]),
        "submission_produced": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
