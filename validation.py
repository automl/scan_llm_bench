#!/usr/bin/env python3
"""
Run validation on a trained checkpoint and update the NePS report.yaml with the
validation loss as objective_to_minimize.

Use this when training saved report.yaml first (e.g. with train_loss as objective)
and you want to overwrite with validation loss after evaluating on the validation
set (e.g. in a separate job that loads checkpoints).

Usage:
    python validation.py \\
        --config config/chinchilla_scaling_configs/50M_1BT_scaled.yaml \\
        --config_number 5 \\
        --neps_root_dir neps/neps_configs/Neps_Scale_Search_Space \\
"""
from absl import flags
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from scan_llm_bench.torch_utils import pytorch_setup
import utils

from utils import load_config, dict_to_config
from data import get_dataloaders
from models import construct_model
from checkpoint_utils import match_state_dict_keys, maybe_load_checkpoint
from engine.engine import TorchEngine, _move_to_device
from torch.nn import CrossEntropyLoss


flags.DEFINE_string('config', 'config/config.yaml',
                    'Path to config.yaml file.')
flags.DEFINE_integer(
    'job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
flags.DEFINE_integer(
    'config_number', None,
    'Config index for sequential NePS runs (unique checkpoint path per config).')
flags.DEFINE_string('neps_root_dir', None,
                    'Path to directory where NePS report will be saved.')
flags.DEFINE_string('neps_report_dir', None,
                    'Alias for neps_root_dir (used by cluster scripts).')
flags.DEFINE_bool(
    'save_neps_report', False,
    'If True and config_overrides is set, save report.yaml to the NePS config directory.')
FLAGS = flags.FLAGS


def main():
    # Load config and checkpoint
    CFG_PATH = FLAGS.config
    cfg, _ = load_config(CFG_PATH)
    ckpt = maybe_load_checkpoint(cfg)
    local_rank, world_size, device, master_process = pytorch_setup(cfg)
    model, _, _ = construct_model(cfg)
    _, validloader = get_dataloaders(cfg)
    engine = TorchEngine(model, cfg, device, local_rank, ckpt)

    # Evaluate model on validation set
    valid_loss = engine.eval(validloader)
    print(f"Validation loss: {valid_loss}")

    # Update NePS report
    neps_dir = FLAGS.neps_report_dir or FLAGS.neps_root_dir
    if not neps_dir:
        print('Cannot save NePS report: neither neps_root_dir nor neps_report_dir set')
    else:
        report_dir = neps_dir / "configs" / f"config_{FLAGS.config_number}"
        report_path = report_dir / "report.yaml"
        if not report_path.exists():
            print(f"Report not found: {report_path}. Creating minimal report with valid_loss.")
            report = {
                "cost": None,
                "objective_to_minimize": valid_loss,
                "learning_curve": [],
                "extra": {"valid_loss": valid_loss},
            }
        else:
            with open(report_path, "r") as f:
                report = yaml.safe_load(f)
            report["objective_to_minimize"] = valid_loss
            extra = report.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            extra["valid_loss"] = valid_loss
            report["extra"] = extra
            if "info_dict" not in report and "extra" in report:
                report["info_dict"] = report["extra"]

        user_result = {
            "objective_to_minimize": valid_loss,
            "cost": report.get("cost"),
            "info_dict": report.get("extra") or report.get("info_dict") or {},
        }
        user_result["info_dict"]["valid_loss"] = valid_loss
        if "learning_curve" in report:
            user_result["learning_curve"] = report["learning_curve"]

        try:
            from neps import save_pipeline_results
            save_pipeline_results(
                user_result=user_result,
                pipeline_id=FLAGS.config_number,
                root_directory=Path(neps_dir),
            )
            print(f"Updated NePS report at {neps_dir} (config_{FLAGS.config_number})")
        except Exception as e:
            print(f"Failed to save NePS report: {e}")


if __name__ == "__main__":
    main()
