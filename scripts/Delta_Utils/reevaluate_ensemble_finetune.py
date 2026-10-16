#!/usr/bin/env python3
"""Re-evaluate existing sweep ONNX policies in Genesis without training/export.

Defaults: physics DR off, observation noise off, random startup joints on.
Planning launches nothing; pass --execute to evaluate. Results stay separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from sweep_ensemble_finetune import ROOT, report_progress, run_worker, validate_stage, write_json


def set_boolean(command, option, enabled):
    command[:] = [arg for arg in command if arg not in (f"--{option}", f"--no-{option}")]
    command.append(f"--{option}" if enabled else f"--no-{option}")


def build_plan(args):
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if source == output:
        raise ValueError("Re-evaluation must use a separate output directory")
    original = json.loads((source / "plan.json").read_text())
    config = copy.deepcopy(original["config"])
    config.update(genesis_add_noise=args.add_noise, genesis_domain_randomization=args.domain_randomization,
                  genesis_randomize_startup_qpos=args.randomize_startup_qpos)
    stages = []
    fingerprints = {}
    for old in original["stages"]:
        if old["kind"] != "genesis":
            continue
        stage = copy.deepcopy(old)
        command = stage["command"]
        policy = Path(command[command.index("--policy_path") + 1]).resolve()
        if not policy.is_file():
            raise FileNotFoundError(policy)
        if str(policy) not in fingerprints:
            fingerprints[str(policy)] = hashlib.sha256(policy.read_bytes()).hexdigest()
        old_result = Path(stage["result_prefix"])
        result = output / old_result.relative_to(source)
        stage["result_prefix"] = str(result)
        command[command.index("--output_json") + 1] = str(result)
        set_boolean(command, "add_noise", args.add_noise)
        set_boolean(command, "domain_randomization", args.domain_randomization)
        set_boolean(command, "randomize_startup_qpos", args.randomize_startup_qpos)
        stages.append(stage)
    if not stages:
        raise ValueError("No Genesis stages found in source plan")
    return dict(config=config, source_dir=str(source), input_sha256=fingerprints,
                source_plan_sha256=hashlib.sha256((source / "plan.json").read_bytes()).hexdigest(), stages=stages)


def validate_settings(stage, config, record):
    metrics = json.loads(Path(record["result_file"]).read_text())
    for field, desired in (("add_noise", config["genesis_add_noise"]),
                           ("domain_randomization", config["genesis_domain_randomization"]),
                           ("randomize_startup_qpos", config["genesis_randomize_startup_qpos"])):
        if field not in metrics or bool(metrics[field]) != desired:
            raise ValueError(f"{stage['id']}: unexpected saved setting {field}={metrics.get(field)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--add_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--domain_randomization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--randomize_startup_qpos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker_timeout", type=float, default=7200)
    args = parser.parse_args()
    plan = build_plan(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output directory has a different plan")
    write_json(plan_path, plan)
    print(f"Re-evaluation plan: {plan_path}; {len(plan['stages'])} Genesis stages; no training or export", flush=True)
    print(f"Physics DR={args.domain_randomization}; observation noise={args.add_noise}; random startup joints={args.randomize_startup_qpos}", flush=True)
    if not args.execute:
        print("Plan only; no simulator launched.")
        return
    for stage in plan["stages"]:
        marker = output / "stages" / (stage["id"].replace("/", "__") + ".json")
        if marker.exists():
            if not args.resume:
                raise FileExistsError(f"Use --resume to reuse completed stage: {marker}")
            record = json.loads(marker.read_text())
            validate_stage(stage, record["seconds"])
            validate_settings(stage, plan["config"], record)
            continue
        print(f"Running {stage['id']}", flush=True)
        report_progress(output, plan, current=stage["id"])
        try:
            record = run_worker(stage, marker.with_suffix(".log"), args.worker_timeout)
            validate_settings(stage, plan["config"], record)
        except BaseException as exc:
            report_progress(output, plan, current=stage["id"], status="failed", error=str(exc))
            raise
        write_json(marker, record)
        report_progress(output, plan)
    report_progress(output, plan, status="complete")
    print("All Genesis re-evaluations complete. Generating summary plots.", flush=True)
    subprocess.run([plan["config"]["isaac_python"], str(ROOT / "scripts/Delta_Utils/summarize_ensemble_finetune.py"),
                    "--sweep_dir", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
