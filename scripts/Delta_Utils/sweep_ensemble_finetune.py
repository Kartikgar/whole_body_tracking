#!/usr/bin/env python3
"""Plan, smoke-test, or execute a bounded ensemble-gated finetune/Genesis sweep.

Planning is the default. Every worker has a separate process group and a timeout.
Completed stages may be reused only with the same resolved configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def resolve_config(path, smoke=False, overrides=None):
    config = json.loads(Path(path).read_text())
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for field in ("base_checkpoint", "motion_file", "isaac_python", "genesis_python"):
        value = Path(config[field]).expanduser()
        config[field] = str((ROOT / value).resolve() if not value.is_absolute() else value.resolve())
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    members = []
    for value in config["ensemble_checkpoints"]:
        value = Path(value).expanduser()
        value = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        members.append(str(value))
    config["ensemble_checkpoints"] = members
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("Specify at least two distinct ensemble members")
    if config.get("finetune_domain_randomization", True) is not True:
        raise ValueError("This sweep requires domain randomization during finetuning")
    config["finetune_domain_randomization"] = True
    if type(config["iterations"]) is not int or config["iterations"] < 1:
        raise ValueError("Finetuning budget must be a positive number of PPO iterations")
    probability = config.get("injection_probability", 1.0)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Injection probability must be finite and between 0 and 1")
    tolerance = config.get("genesis_endpoint_height_error_m", 0.5)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Genesis endpoint height error tolerance must be finite and positive")
    ages = config["export_iterations"]
    if not ages or len(set(ages)) != len(ages) or any(not isinstance(age, int) or not 0 <= age < config["iterations"] for age in ages):
        raise ValueError("Export iterations must be distinct integers within the training budget")
    if not config["lambdas"] or any(not math.isfinite(x) or x < 0 for x in config["lambdas"]) or len(set(config["lambdas"])) != len(config["lambdas"]):
        raise ValueError("Lambdas must be distinct nonnegative finite numbers")
    if not config["eval_seeds"] or len(set(config["eval_seeds"])) != len(config["eval_seeds"]):
        raise ValueError("Specify distinct evaluation seeds")
    for key in ("num_envs", "genesis_num_envs"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get('genesis_initial_state_npz'):
        initial=Path(config['genesis_initial_state_npz']).expanduser()
        initial=(ROOT/initial).resolve() if not initial.is_absolute() else initial.resolve()
        if not initial.is_file():raise FileNotFoundError(initial)
        config['genesis_initial_state_npz']=str(initial)
        if len(config['eval_seeds'])!=1:
            raise ValueError('A fixed startup-state batch supports one evaluation seed; multiple seeds would repeat the same states')
    if smoke:
        config.update(iterations=2, export_iterations=[0, 1], lambdas=[1], num_envs=16,
                      genesis_num_envs=2, eval_seeds=[0], genesis_max_steps=25,
                      include_controls=False, include_original_base=False, genesis_initial_state_npz=None,
                      genesis_record_video=False)
    config["smoke"] = smoke
    return config


def build_plan(config, output):
    output = Path(output).resolve()
    stages = []
    candidates = []
    base, motion = config["base_checkpoint"], config["motion_file"]
    experiments = [(f"lambda_{lam:g}", lam, config.get("injection_probability", 1.0), config["export_iterations"]) for lam in config["lambdas"]]
    if config.get("include_controls", True):
        experiments.append(("no_delta_finetune", 0, 0.0, [config["iterations"] - 1]))
    if config.get('include_original_base',config.get('include_controls',True)):
        candidates.append(("original_base", None, 0, base))
    for label, lam, probability, ages in experiments:
        run = output / label / "training"
        train = [config["isaac_python"], "scripts/rsl_rl/train.py", "--headless", "--require_dr",
                 "--task", "Tracking-Flat-G1-DeltaWrench-Finetune-v0", "--motion_file", motion,
                 "--resume", "True", "--checkpoint", base, "--logger", "tensorboard",
                 "--output_dir", str(run), "--max_iterations", str(config["iterations"]),
                 "--num_envs", str(config["num_envs"]), "--seed", str(config["train_seed"]),
                 "--save_after_iterations", *map(str, ages), "--delta_policy_uncertainty_gate_scale", str(lam),
                 "--delta_policy_injection_probability", str(probability),
                 "--delta_policy_checkpoints", *config["ensemble_checkpoints"]]
        if config.get("preserve_frozen_policy_rng", False):
            train.append("--preserve_frozen_policy_rng")
        stages.append(dict(id=f"{label}/train", kind="train", command=train, outputs=[str(run / "update_checkpoints.json")],
                           iterations=ages, run_dir=str(run)))
        for age in ages:
            candidates.append((label, lam if probability else None, age, str(run / f"model_finetune_iter_{age:04d}.pt")))
    for label, lam, age, checkpoint in candidates:
        directory = output / label / f"iteration_{age:04d}"
        onnx = directory / "policy.onnx"
        export = [config["isaac_python"], "scripts/rsl_rl/play.py", "--headless", "--task", "Tracking-Flat-G1-v0",
                  "--motion_file", motion, "--checkpoint", checkpoint, "--num_envs", "1", "--seed", "0",
                  "--export_only", "--export_onnx_path", str(onnx)]
        stages.append(dict(id=f"{label}/{age}/export", kind="export", command=export, outputs=[str(onnx)]))
        for seed in config["eval_seeds"]:
            eval_dir = directory / f"genesis_seed_{seed}"
            result = eval_dir / "metrics.json"
            command = [config["genesis_python"], "scripts/eval_sim2sim_genesis.py", "--policy_path", str(onnx),
                       "--num_envs", str(config["genesis_num_envs"]), "--compute_metrics", "--metric_num_envs",
                       str(config["genesis_num_envs"]), "--seed", str(seed), "--output_json", str(result),
                       "--add_noise" if config.get("genesis_add_noise", True) else "--no-add_noise",
                       "--randomize_startup_qpos" if config.get("genesis_randomize_startup_qpos", True) else "--no-randomize_startup_qpos",
                       "--domain_randomization" if config.get("genesis_domain_randomization", False) else "--no-domain_randomization"]
            command.extend(["--motion_file", motion, "--endpoint-height-error-tolerance-m",
                            str(config.get("genesis_endpoint_height_error_m", 0.5))])
            if config.get("genesis_max_steps") is not None:
                command.extend(["--max_steps", str(config["genesis_max_steps"])])
            if config.get('genesis_initial_state_npz'):
                command.extend(['--initial_state_npz',config['genesis_initial_state_npz']])
            outputs=[]
            if config.get('genesis_record_video',False):
                video=eval_dir/'rollout.mp4'
                command.extend(['--record_video','--video_name',str(video)])
                outputs.append(str(video))
            stages.append(dict(id=f"{label}/{age}/genesis/{seed}", kind="genesis", command=command,
                               result_prefix=str(result), outputs=outputs, label=label, lambda_value=lam,
                               finetune_iteration=age, seed=seed))
    # Evaluate the reference first, then each trained candidate before starting
    # the next lambda so useful target-performance feedback arrives early.
    labels = ["original_base", *[experiment[0] for experiment in experiments]]
    return sorted(stages, key=lambda stage: labels.index(stage["id"].split("/")[0]))


def report_progress(output, plan, current=None, status="running", error=None):
    """Persist progress and seed-aggregated target results for a background run."""
    completed = []
    groups = {}
    for stage in plan["stages"]:
        marker = output / "stages" / (stage["id"].replace("/", "__") + ".json")
        if not marker.is_file():
            continue
        completed.append(stage)
        if stage["kind"] != "genesis":
            continue
        record = json.loads(marker.read_text())
        metrics = json.loads(Path(record["result_file"]).read_text())
        key = (stage["label"], stage["finetune_iteration"])
        group = groups.setdefault(key, dict(label=key[0], finetune_iteration=key[1], seeds=[],
                                          completion=[], survival=[], tracking=[]))
        group["seeds"].append(stage["seed"])
        group["completion"].append(metrics["full_clip_completion_pct"])
        group["survival"].append(metrics["clip_survival_pct"])
        values = [r["mpjpe"] for r in metrics["rollout_metrics"] if r["completed"] and math.isfinite(r["mpjpe"])]
        if values:
            group["tracking"].append(sum(values) / len(values))
    leaderboard = []
    for group in groups.values():
        leaderboard.append(dict(label=group["label"], finetune_iteration=group["finetune_iteration"],
                                seeds=group["seeds"], provisional=len(group["seeds"]) != len(plan["config"]["eval_seeds"]),
                                completion_pct=sum(group["completion"]) / len(group["completion"]),
                                survival_pct=sum(group["survival"]) / len(group["survival"]),
                                completed_mpjpe_mm=sum(group["tracking"]) / len(group["tracking"]) if group["tracking"] else None))
    leaderboard.sort(key=lambda row: (-row["completion_pct"], -row["survival_pct"],
                                     row["completed_mpjpe_mm"] if row["completed_mpjpe_mm"] is not None else math.inf))
    counts = {kind: sum(stage["kind"] == kind for stage in completed) for kind in ("train", "export", "genesis")}
    progress = dict(status=status, current_stage=current, completed_stages=len(completed), total_stages=len(plan["stages"]),
                    completed_by_kind=counts, leaderboard=leaderboard, error=error)
    write_json(output / "progress.json", progress)
    if leaderboard:
        best = leaderboard[0]
        print(f"Best so far: {best['label']} iteration {best['finetune_iteration']}; "
              f"completion {best['completion_pct']:.2f}%, survival {best['survival_pct']:.2f}%, "
              f"completed MPJPE {best['completed_mpjpe_mm']} mm; seeds {best['seeds']}" +
              (" (provisional)" if best["provisional"] else ""), flush=True)
    return progress


def validate_stage(stage, elapsed):
    for name in stage["outputs"]:
        if not Path(name).is_file():
            raise FileNotFoundError(f"Worker did not produce {name}")
    if stage["kind"] == "train":
        manifest = json.loads(Path(stage["outputs"][0]).read_text())
        if manifest["domain_randomization_enabled"] is not True:
            raise RuntimeError("Finetuning disabled domain randomization")
        records = {row["finetune_iteration"]: row for row in manifest["checkpoints"]}
        for age in stage["iterations"]:
            row = records[age]
            if row["iteration"] != manifest["initial_iteration"] + age or not Path(row["path"]).is_file():
                raise RuntimeError("Checkpoint finetuning iteration is inconsistent")
    result = dict(stage=stage["id"], seconds=elapsed, status="complete")
    if stage["kind"] == "genesis":
        prefix = Path(stage["result_prefix"])
        matches = list(prefix.parent.glob(prefix.stem + "_*.json"))
        if prefix.is_file():
            matches.append(prefix)
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one Genesis result for {prefix}; found {matches}")
        metrics = json.loads(matches[0].read_text())
        expected = int(stage["command"][stage["command"].index("--metric_num_envs") + 1])
        if metrics.get("metric_num_envs_done") != expected:
            raise RuntimeError("Genesis did not evaluate all requested environments")
        for flag,field in (('add_noise','add_noise'),('domain_randomization','domain_randomization'),
                           ('randomize_startup_qpos','randomize_startup_qpos')):
            desired=f'--{flag}' in stage['command']
            if bool(metrics.get(field))!=desired:raise RuntimeError(f'Unexpected Genesis setting {field}')
        if '--initial_state_npz' in stage['command']:
            expected_initial=stage['command'][stage['command'].index('--initial_state_npz')+1]
            if metrics.get('initial_state_npz')!=expected_initial:raise RuntimeError('Genesis startup archive differs from plan')
        if '--endpoint-height-error-tolerance-m' in stage['command']:
            tolerance = float(stage['command'][stage['command'].index('--endpoint-height-error-tolerance-m') + 1])
            if metrics.get('termination_definition', {}).get('endpoint_height_error_m') != tolerance:
                raise RuntimeError('Genesis endpoint tolerance differs from plan')
        result.update(result_file=str(matches[0]), label=stage["label"], lambda_value=stage["lambda_value"],
                      finetune_iteration=stage["finetune_iteration"], seed=stage["seed"])
    return result


def run_worker(stage, log, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(stage["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
            if status:
                raise RuntimeError(f"Worker exited {status}; inspect {log}")
        finally:
            # Terminate only descendants in this worker's own process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # The main worker can exit while an owned descendant is still alive.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return validate_stage(stage, time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/finetune/dance_ensemble_gate.json"))
    parser.add_argument("--output_dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute the full configured sweep (requires operator approval).")
    mode.add_argument("--smoke", action="store_true", help="Only lambda=1, two PPO iterations, 16 Isaac envs, two Genesis envs, 25 steps.")
    parser.add_argument("--resume", action="store_true", help="Reuse validated completed stages with identical configuration.")
    parser.add_argument("--worker_timeout", type=float, default=7200)
    parser.add_argument("--lambda_value", "--lambda", type=float, help="Run a single gate coefficient instead of the configured sweep.")
    parser.add_argument("--injection_prob", type=float, help="Episode-level delta injection probability in [0, 1].")
    parser.add_argument("--iterations", type=int, help="Number of completed PPO iterations to train.")
    parser.add_argument("--eval_iterations", type=int, nargs="+", help="Save/evaluate after these COMPLETED PPO iteration counts (1-based).")
    parser.add_argument("--train_seed", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--genesis_num_envs", type=int)
    parser.add_argument("--initial_state_npz", help="Fixed Genesis initial-state archive; requires one evaluation seed.")
    parser.add_argument("--endpoint_tolerance", type=float, help="Genesis endpoint vertical-error tolerance in metres.")
    parser.add_argument("--no_controls", action="store_true", help="Only run the supplied ensemble configuration.")
    parser.add_argument("--preserve_frozen_policy_rng", action="store_true", default=None,
                        help="Preserve Torch RNG during frozen-network loading, as in the matched controls.")
    args = parser.parse_args()
    if args.eval_iterations is not None and any(n < 1 for n in args.eval_iterations):
        parser.error("--eval_iterations uses positive completed PPO iteration counts")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error("--worker_timeout must be finite and positive")
    overrides = dict(lambdas=[args.lambda_value] if args.lambda_value is not None else None,
                     injection_probability=args.injection_prob, iterations=args.iterations,
                     export_iterations=[n - 1 for n in args.eval_iterations] if args.eval_iterations is not None else None,
                     train_seed=args.train_seed, num_envs=args.num_envs, genesis_num_envs=args.genesis_num_envs,
                     genesis_initial_state_npz=args.initial_state_npz,
                     genesis_endpoint_height_error_m=args.endpoint_tolerance,
                     preserve_frozen_policy_rng=args.preserve_frozen_policy_rng)
    if args.no_controls:
        overrides.update(include_controls=False, include_original_base=False)
    config = resolve_config(args.config, args.smoke, overrides)
    output = Path(args.output_dir).resolve()
    stages = build_plan(config, output)
    # Include content fingerprints for policies and reference to prevent accidental reuse.
    fingerprints = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in [config["base_checkpoint"], config["motion_file"], *config["ensemble_checkpoints"]]}
    if config.get('genesis_initial_state_npz'):
        name=config['genesis_initial_state_npz'];fingerprints[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    plan = dict(config=config, input_sha256=fingerprints, stages=stages)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Output directory contains a different plan; use a new output directory")
    write_json(plan_path, plan)
    print(f"Plan: {plan_path}; {len(stages)} stages; maximum {config['iterations']} PPO iterations/run; source DR enabled", flush=True)
    if not (args.execute or args.smoke):
        print("Plan only. No simulator or training process launched.")
        return
    for stage in stages:
        marker = output / "stages" / (stage["id"].replace("/", "__") + ".json")
        if args.resume and marker.exists():
            previous = json.loads(marker.read_text())
            validate_stage(stage, previous["seconds"])
            print(f"Reuse {stage['id']}", flush=True)
            continue
        if marker.exists():
            raise FileExistsError(f"Completed stage exists; use --resume: {marker}")
        print(f"Running {stage['id']}", flush=True)
        report_progress(output, plan, current=stage["id"])
        log = marker.with_suffix(".log")
        try:
            result = run_worker(stage, log, args.worker_timeout)
        except BaseException as exc:
            report_progress(output, plan, current=stage["id"], status="failed", error=str(exc))
            raise
        write_json(marker, result)
        report_progress(output, plan)
    report_progress(output, plan, status="complete")
    print("Sweep complete. Use summarize_ensemble_finetune.py to aggregate and plot.")


if __name__ == "__main__":
    main()
