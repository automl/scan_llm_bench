"""
Sync LM-evaluation-harness json results from eval_results/ into NePS report info_dict (extra).

Expected eval directory names:
    config_<CONFIG_ID>_step_<STEP>

For each config, this script:
1) Collects all available steps and sorts them.
2) Loads the latest results_*.json file for every step.
3) Writes aggregated eval info into NePS report via neps.save_pipeline_results.

The data is stored under info_dict as task-name keys, each with a step-sorted list:
- hellaswag: [{step: 827, ...}, {step: 1654, ...}]
- piqa: [{step: 827, ...}, ...]

All fields from each task's JSON result are preserved in every list entry.

Usage:
    python scripts/eval_results_neps_reports.py \
        --eval_results_dir eval_results \
        --neps_root_dir neps/neps_configs/SlimPajama_50M_1BT \
"""

import argparse
import json
import re
from pathlib import Path

import yaml


DIR_PATTERN = re.compile(r"^config_(\d+)_step_(\d+)$")


def _safe_yaml_load(path: Path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _safe_json_load(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def _collect_eval_runs(eval_results_dir: Path):
    """
    Returns:
        config_to_steps: dict[int, list[int]]
        run_dir_by_config_step: dict[tuple[int, int], Path]
    """
    config_to_steps = {}
    run_dir_by_config_step = {}

    for entry in eval_results_dir.iterdir():
        if not entry.is_dir():
            continue

        match = DIR_PATTERN.match(entry.name)
        if not match:
            continue

        config_id = int(match.group(1))
        step = int(match.group(2))

        config_to_steps.setdefault(config_id, []).append(step)
        run_dir_by_config_step[(config_id, step)] = entry

    for config_id, steps in config_to_steps.items():
        config_to_steps[config_id] = sorted(steps)

    return config_to_steps, run_dir_by_config_step


def _find_latest_results_json(run_dir: Path):
    """Find latest results_*.json recursively by modification time."""
    candidates = list(run_dir.rglob("results_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_existing_report(config_report_path: Path):
    if not config_report_path.exists():
        return {}

    report = _safe_yaml_load(config_report_path)
    return report if isinstance(report, dict) else {}


def _merge_task_history(existing_info: dict, new_task_history: dict):
    """Merge task histories by step without losing previous entries."""
    merged_info = dict(existing_info)

    for task_name, new_entries in new_task_history.items():
        existing_entries = merged_info.get(task_name)
        if not isinstance(existing_entries, list):
            existing_entries = []

        merged_by_step = {}

        for entry in existing_entries:
            if isinstance(entry, dict) and isinstance(entry.get("step"), int):
                merged_by_step[entry["step"]] = entry

        for entry in new_entries:
            merged_by_step[entry["step"]] = entry

        merged_info[task_name] = [
            merged_by_step[step] for step in sorted(merged_by_step)
        ]

    return merged_info


def _build_user_result(existing_report: dict, new_task_history: dict):
    objective = existing_report.get("objective_to_minimize")
    if objective is None:
        objective = (existing_report.get("extra") or {}).get("test_loss")

    info_dict = existing_report.get("info_dict")
    if not isinstance(info_dict, dict):
        info_dict = existing_report.get("extra")
    if not isinstance(info_dict, dict):
        info_dict = {}

    merged_info = _merge_task_history(info_dict, new_task_history)

    all_steps = set()
    for task_name, task_entries in merged_info.items():
        if not isinstance(task_entries, list):
            continue
        for entry in task_entries:
            if isinstance(entry, dict) and isinstance(entry.get("step"), int):
                all_steps.add(entry["step"])
    merged_info["eval_steps"] = sorted(all_steps)

    return {
        "objective_to_minimize": objective,
        "cost": existing_report.get("cost"),
        "info_dict": merged_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sync eval_results JSON files into NePS reports (info_dict)."
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="/eval_results",
        help="Directory containing config_<id>_step_<step> folders.",
    )
    parser.add_argument(
        "--neps_root_dir",
        type=str,
        default="/neps/neps_configs/Neps_Scale_Search_Space",
        help="NePS root directory containing configs/config_<id>/report.yaml.",
    )
    parser.add_argument(
        "--dry_run",
        type=bool,
        default=False,
        help="Print planned updates without writing reports.",
    )
    args = parser.parse_args()

    eval_results_dir = Path(args.eval_results_dir)
    neps_root_dir = Path(args.neps_root_dir)

    if not eval_results_dir.is_dir():
        raise FileNotFoundError(f"eval_results_dir not found: {eval_results_dir}")
    if not neps_root_dir.is_dir():
        raise FileNotFoundError(f"neps_root_dir not found: {neps_root_dir}")

    config_to_steps, run_dir_by_config_step = _collect_eval_runs(eval_results_dir)

    if not config_to_steps:
        print(f"No config_<id>_step_<step> directories found in {eval_results_dir}")
        return

    print("Discovered config -> sorted steps:")
    for config_id in sorted(config_to_steps):
        print(f"  {config_id}: {config_to_steps[config_id]}")

    updated = 0

    for config_id in sorted(config_to_steps):
        eval_steps = config_to_steps[config_id]
        new_task_history = {}

        for step in eval_steps:
            run_dir = run_dir_by_config_step[(config_id, step)]
            latest_json = _find_latest_results_json(run_dir)

            if latest_json is None:
                print(f"  [WARN] config_{config_id}_step_{step}: no results_*.json found")
                continue

            payload = _safe_json_load(latest_json)
            results = payload.get("results")
            if not isinstance(results, dict):
                print(f"  [WARN] config_{config_id}_step_{step}: invalid results payload")
                continue

            for task_name, task_values in results.items():
                if not isinstance(task_values, dict):
                    continue

                entry = {
                    "step": step,
                    "json_path": str(latest_json),
                    "date": payload.get("date"),
                }
                entry.update(task_values)
                new_task_history.setdefault(task_name, []).append(entry)

        if not new_task_history:
            print(f"  [WARN] config_{config_id}: no valid eval JSON found, skipping")
            continue

        report_path = neps_root_dir / "configs" / f"config_{config_id}" / "report.yaml"
        existing_report = _read_existing_report(report_path)
        user_result = _build_user_result(existing_report, new_task_history)

        if args.dry_run:
            print(
                f"  [DRY-RUN] config_{config_id}: would merge {len(new_task_history)} "
                f"tasks across {len(eval_steps)} discovered steps"
            )
            continue

        try:
            from neps import save_pipeline_results

            save_pipeline_results(
                user_result=user_result,
                pipeline_id=config_id,
                root_directory=neps_root_dir,
            )
            updated += 1
            print(f"  [OK] Updated NePS report for config_{config_id}")
        except Exception as exc:
            print(f"  [FAIL] config_{config_id}: {exc}")

    if args.dry_run:
        print("Dry run finished.")
    else:
        print(f"Finished. Updated {updated} config reports.")


if __name__ == "__main__":
    main()
