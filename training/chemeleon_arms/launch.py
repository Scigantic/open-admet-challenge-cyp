"""Launch the CheMeleon multitask arms on SageMaker.
DRY RUN BY DEFAULT. Requires --submit to create anything.
Pattern copied from d6_pairwise_chemeleon_20260904/launch_group_cv.py:
the image runs whatever script we upload to the training channel."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

HERE = Path("analysis/chemeleon_multitask_20260910")
STAGE = HERE / "staging"
REGION = "us-east-1"
ROLE = "arn:aws:iam::<aws-account-id>:role/<sagemaker-execution-role>"
IMAGE = "<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/scigantic-finetune-job:latest"
BUCKET = "<bucket>"
RUNNER = "sagemaker_runner.py"
FILES = ["sagemaker_runner.py", "train_table.csv", "blind.csv",
         "weights.npy", "folds.npy", "chemeleon_mp.pt", "organizer_cyp.pth", "bounds.csv"]

def aws(*a: str) -> str:
    r = subprocess.run(["aws", *a], capture_output=True, text=True)
    if r.returncode: raise SystemExit(f"aws {' '.join(a)}\n{r.stderr}")
    return r.stdout.strip()

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--objective", default="mae", help="mae|interval")
    ap.add_argument("--weighting", default="invvar", help="invvar|uniform")
    ap.add_argument("--group", default="G22", help="G4|G8|G12|G22 task grouping")
    ap.add_argument("--task-weighting", default="uniform",
                    help="uniform|denominator (HEAD weights; runner already supports it)")
    ap.add_argument("--stage", default=None,
                    help="alternate staging dir (default: this directory's staging/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instance-type", default="ml.g5.xlarge")
    ap.add_argument("--max-runtime", type=int, default=14400)
    ap.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d-%H%M"))
    ap.add_argument("--submit", action="store_true", help="REQUIRED to actually create jobs")
    a = ap.parse_args()

    global STAGE
    if a.stage:
        STAGE = Path(a.stage)
        print(f"STAGING OVERRIDE: {STAGE}")
    for f in FILES:
        p = STAGE / f
        if not p.exists(): raise SystemExit(f"missing staged input: {p}")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    plan = []
    for arm in arms:
        for fold in folds:
            tag = (arm.lower() + ("" if a.group == "G22" else a.group.lower())
                   + ("" if a.task_weighting == "uniform" else "den"))
            job = f"oadmet-chemeleon-{tag}-f{fold}-{a.stamp}"
            prefix = f"openadmet-cyp-challenge/{job}"
            plan.append({"job_name": job, "arm": arm, "fold": fold,
                         "input_s3": f"s3://{BUCKET}/{prefix}/input/",
                         "result_s3": f"s3://{BUCKET}/{prefix}/results/"})
    est_hr = 1.41
    print(f"PLAN: {len(plan)} jobs on {a.instance_type}")
    print(f"  arms {arms} x folds {folds}")
    print(f"  max_epochs {a.max_epochs} patience {a.patience} batch {a.batch_size}")
    print(f"  staged inputs:")
    for f in FILES:
        p = STAGE / f
        print(f"    {f:24} {p.stat().st_size/1e6:7.2f} MB  sha {sha(p)[:12]}")
    print(f"  ESTIMATED COST: {len(plan)} jobs x ~0.5-1.5 h x ${est_hr}/h "
          f"= ${len(plan)*0.5*est_hr:.0f}-${len(plan)*1.5*est_hr:.0f}")
    for r in plan: print(f"    {r['job_name']}")
    if not a.submit:
        print("\nDRY RUN. Nothing created. Re-run with --submit to launch.")
        (HERE / f"plan_{a.stamp}.json").write_text(json.dumps(plan, indent=1))
        return
    for r in plan:
        for f in FILES:
            aws("s3", "cp", str(STAGE / f), f"{r['input_s3']}{f}", "--only-show-errors")
        algorithm = {"TrainingImage": IMAGE, "TrainingInputMode": "File",
                     "ContainerEntrypoint": ["/opt/conda/bin/python"],
                     "ContainerArguments": [f"/opt/ml/input/data/training/{RUNNER}"]}
        hp = {"arm": r["arm"], "fold": str(r["fold"]), "max_epochs": str(a.max_epochs),
              "patience": str(a.patience), "batch_size": str(a.batch_size),
              "seed": str(a.seed), "group": a.group, "objective": a.objective,
              "weighting": a.weighting, "task_weighting": a.task_weighting,
              "result_s3": r["result_s3"]}
        inputs = [{"ChannelName": "training",
                   "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                   "S3Uri": r["input_s3"],
                                                   "S3DataDistributionType": "FullyReplicated"}}}]
        arn = aws("sagemaker", "create-training-job", "--region", REGION,
                  "--training-job-name", r["job_name"], "--role-arn", ROLE,
                  "--algorithm-specification", json.dumps(algorithm, separators=(",", ":")),
                  "--hyper-parameters", json.dumps(hp, separators=(",", ":")),
                  "--input-data-config", json.dumps(inputs, separators=(",", ":")),
                  "--output-data-config", f"S3OutputPath=s3://{BUCKET}/openadmet-cyp-challenge/output/",
                  "--resource-config",
                  f"InstanceType={a.instance_type},InstanceCount=1,VolumeSizeInGB=50",
                  "--stopping-condition", f"MaxRuntimeInSeconds={a.max_runtime}")
        r["arn"] = arn; r["status"] = "submitted"
        print(f"  submitted {r['job_name']}")
    (HERE / f"submitted_{a.stamp}.json").write_text(json.dumps(plan, indent=1))
    print(f"\n{len(plan)} jobs submitted. Track with:")
    print(f"  aws sagemaker list-training-jobs --name-contains oadmet-chemeleon-{a.stamp}")


if __name__ == "__main__":
    main()
