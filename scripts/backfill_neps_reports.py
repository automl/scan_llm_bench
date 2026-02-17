#!/usr/bin/env python3
"""
Backfill report.yaml for NePS configs from existing checkpoint runs.

Scans checkpoint directories (e.g. checkpoints/SlimPajama_50M_1BT_experiment/),
finds job_XXX/config_YY that have metrics.json (completed successfully),
extracts values and saves report.yaml to the NePS config directory.

Optionally uses neps.import_trials to add results to the NePS optimization state.

Usage:
    python scripts/backfill_neps_reports.py \
        --checkpoint_dir checkpoints/SlimPajama_50M_1BT_experiment \
        --neps_root_dir neps/neps_configs/SlimPajama_50M_1BT \
        [--import_to_neps]

Or with defaults:
    python scripts/backfill_neps_reports.py --checkpoint_dir checkpoints/SlimPajama_50M_1BT_experiment
"""

from utils import dict_to_config
from models.params_flops import total_training_flops, format_flops
import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _to_serializable(x):
    """Convert to YAML-serializable type."""
    if hasattr(x, 'item'):
        return x.item()
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    return x


def extract_final_metrics(metrics_dict):
    """Extract final values from metrics dict (keys -> lists)."""
    result = {}
    for k, v in metrics_dict.items():
        if isinstance(v, list) and len(v) > 0:
            result[k] = _to_serializable(v[-1])
        else:
            result[k] = _to_serializable(v)
    return result


def load_config_from_path(config_path):
    """Load config as namedtuple from a YAML path."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict.pop('neps_search_space', None)
    cfg, _ = dict_to_config(config_dict)
    return cfg


def _parse_expand(expand_val):
    """Parse expand (e.g. '8/3' or 2.666) to float."""
    if expand_val is None:
        return 8/3
    if isinstance(expand_val, (int, float)):
        return float(expand_val)
    if isinstance(expand_val, str) and '/' in expand_val:
        a, b = expand_val.split('/')
        return float(a) / float(b)
    return float(expand_val)


def compute_flops_from_config(cfg):
    """Compute total training FLOPs from config."""
    n_tokens = getattr(cfg, 'n_tokens', None)
    if n_tokens is None:
        return None
    try:
        return total_training_flops(
            n_layers=cfg.n_layers,
            d=cfg.d_model,
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            n_tokens=n_tokens,
            mlp_class=getattr(cfg, 'mlp_class', 'glu'),
            expand=_parse_expand(getattr(cfg, 'expand', '8/3')),
        )
    except Exception:
        return None


def find_completed_configs(checkpoint_dir):
    """
    Find all config_YY directories that have metrics.json.
    Returns: list of (config_num, config_dir_path, metrics_path)
    For each config_num, we keep the most recent by metrics.json mtime.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    config_to_best = {}  # config_num -> (mtime, config_dir, metrics_path)

    for job_dir in sorted(checkpoint_path.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.startswith('job_'):
            continue
        for config_dir in job_dir.iterdir():
            if not config_dir.is_dir():
                continue
            match = re.match(r'^config_(\d+)$', config_dir.name)
            if not match:
                continue
            config_num = int(match.group(1))
            metrics_path = config_dir / 'metrics.json'
            config_yaml = config_dir / 'config.yaml'
            if not metrics_path.exists() or not config_yaml.exists():
                continue
            mtime = metrics_path.stat().st_mtime
            if config_num not in config_to_best or mtime > config_to_best[config_num][0]:
                config_to_best[config_num] = (
                    mtime, str(config_dir), str(metrics_path))

    return [
        (config_num, path, metrics_path)
        for config_num, (_, path, metrics_path) in sorted(config_to_best.items())
    ]


def build_report(ckpt_config_path, metrics_path, compute_flops):
    """Build report dict from config and metrics."""
    with open(ckpt_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    with open(metrics_path, 'r') as f:
        metrics_raw = json.load(f)

    final = extract_final_metrics(metrics_raw)
    cfg, _ = dict_to_config(config_dict)

    n_tokens = getattr(cfg, 'n_tokens', None)
    micro_batch = getattr(cfg, 'micro_batch_size', None)
    grad_accum = getattr(cfg, 'grad_accumulation_steps', 1)
    global_batch = (micro_batch * grad_accum) if micro_batch else None

    result = {
        'objective_to_minimize': final.get('valid/loss'),
        'cost': compute_flops,
        'learning_curve': metrics_raw.get('lr'),
        'info_dict': {
            'compute_formatted': format_flops(compute_flops) if compute_flops else None,
            'train_loss': final.get('train/loss'),
            'valid_loss': final.get('valid/loss'),
            'n_tokens': n_tokens,
            'n_data': n_tokens,
            'micro_batch_size': micro_batch,
            'global_batch_size': global_batch,
            'final_step': final.get('step'),
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Backfill NePS report.yaml from checkpoint runs')
    parser.add_argument(
        '--checkpoint_dir',
        default='checkpoints/SlimPajama_50M_Neps_Runs',
        help='Checkpoint experiment directory (e.g. checkpoints/SlimPajama_50M_1BT_experiment)',
    )
    parser.add_argument(
        '--neps_root_dir',
        default='neps/neps_configs/SlimPajama_50M_1BT',
        help='NePS configs directory (e.g. neps/neps_configs/SlimPajama_50M_1BT)',
    )
    parser.add_argument(
        '--import_to_neps',
        action='store_true',
        help='Also call neps.import_trials to add results to NePS optimization state',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be done without writing files',
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    neps_root_dir = Path(args.neps_root_dir)
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint_dir does not exist: {checkpoint_dir}")
        sys.exit(1)

    completed = find_completed_configs(checkpoint_dir)
    print(f"Found {len(completed)} completed configs in {checkpoint_dir}")

    # imported_trials = []

    for config_num, config_dir_path, metrics_path in completed:
        config_yaml = Path(config_dir_path) / 'config.yaml'
        report_dir = neps_root_dir / 'configs' / f'config_{config_num}'
        report_path = report_dir / 'report.yaml'

        # Load config and compute FLOPs
        cfg = load_config_from_path(config_yaml)
        compute_flops = compute_flops_from_config(cfg)
        report = build_report(config_yaml, metrics_path, compute_flops)
        print(f"report: {report}")
        if args.dry_run:
            obj = report['objective_to_minimize']
            obj_str = f"{obj:.4f}" if obj is not None else "N/A"
            print(
                f"Would write {report_path}: objective={obj_str}, compute_formatted={report['info_dict'].get('compute_formatted')}, learning_curve={report['learning_curve']}")
            continue

        try:
            from neps import save_pipeline_results
            save_pipeline_results(
                user_result=report,
                pipeline_id=config_num + 1,
                root_directory=neps_root_dir,
            )
            print(f"Saved NePS report to {neps_root_dir}")
        except Exception as e:
            print(f"Warning: Failed to save NePS report: {e}")

        # if args.import_to_neps:
        #     # Load NePS config (overrides only) for import_trials
        #     neps_config_path = report_dir / 'config.yaml'
        #     if neps_config_path.exists():
        #         with open(neps_config_path, 'r') as f:
        #             neps_config = yaml.safe_load(f)
        #         # Ensure extra has only serializable values
        #         extra = report.get('extra', {}) or {}
        #         extra_clean = {k: _to_serializable(v) for k, v in extra.items()}
        #         imported_trials.append((
        #             neps_config,
        #             {
        #                 'objective_to_minimize': report['objective'],
        #                 'cost': report.get('cost'),
        #                 'extra': extra_clean,
        #             },
        #         ))

    # if args.import_to_neps and imported_trials and not args.dry_run:
    #     try:
    #         import neps
    #         neps.import_trials(
    #             imported_trials,
    #             root_directory=str(neps_configs_dir),
    #         )
    #         print(f"Imported {len(imported_trials)} trials to NePS state at {neps_configs_dir}")
    #     except Exception as e:
    #         print(f"Warning: neps.import_trials failed: {e}")


if __name__ == '__main__':
    main()
