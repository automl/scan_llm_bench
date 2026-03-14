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
from absl import app, flags
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from models.params_flops import format_flops
from torch_utils import pytorch_setup
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


def main(_):
    # Load config and checkpoint
    CFG_PATH = FLAGS.config
    cfg, _ = load_config(CFG_PATH)
    local_rank, world_size, device, master_process = pytorch_setup(cfg)
    model, _, compute_flops = construct_model(cfg)
    _, validloader, testloader = get_dataloaders(cfg)
    print(f"Loaded model and dataloaders. Validloader length = {len(validloader)}, Testloader length = {len(testloader)}.")
    ckpt_dir = os.path.join(cfg.out_dir, cfg.resume_exp_name)
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')],
            key=lambda f: int(f.split(".")[0].split("_")[-1]))
    step_values = [step.split(".")[0].split("_")[-1] for step in files]
    valid_losses = []
    test_losses = []
    eval_steps = []
    last_valid_loss = None
    last_test_loss = None   
    for step in step_values:
        cfg = cfg._replace(resume_step=step)
        print(f"Loading checkpoint from step {cfg.resume_step} for validation...")
        ckpt = maybe_load_checkpoint(cfg)
        engine = TorchEngine(model, cfg, device, local_rank, ckpt)

        # Evaluate model on validation set
        valid_loss = engine.eval(validloader)
        print(f"Validation loss: {valid_loss} on step: {step}")
        test_loss = engine.eval(testloader)
        print(f"Test loss: {test_loss} on step: {step}")
        last_valid_loss = valid_loss
        last_test_loss = test_loss
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        eval_steps.append(int(step))


    # Update NePS report
    neps_dir = FLAGS.neps_report_dir or FLAGS.neps_root_dir
    if not neps_dir:
        print('Cannot save NePS report: neither neps_root_dir nor neps_report_dir set')
    else:
        report_dir = os.path.join(neps_dir, "configs", f"config_{FLAGS.config_number}")
        report_path = os.path.join(report_dir, "report.yaml")
        if not os.path.exists(report_path):
            n_params = (
                model.count_params(non_embedding=False)
                if hasattr(model, 'count_params')
                else sum(p.numel() for p in model.parameters())
            )
            print(f"Report not found: {report_path}. Creating minimal report with valid_loss.")
            report = {
                "cost": compute_flops if compute_flops else None,
                "objective_to_minimize": last_test_loss,
                "extra": 
                    {"last_valid_loss": last_valid_loss,
                     "valid_losses": valid_losses,
                     "last_test_loss": last_test_loss,
                     "test_losses": test_losses,
                     "eval_steps": eval_steps,
                     "n_params": n_params,
                     'compute_formatted': format_flops(compute_flops) if compute_flops else None,
                     "flops": compute_flops if compute_flops else None,
                     "n_data": cfg.n_tokens if hasattr(cfg, 'n_tokens') else None,
                     'micro_batch_size': cfg.micro_batch_size if hasattr(cfg, 'micro_batch_size') else None,
                    },
            }
        else:
            with open(report_path, "r") as f:
                report = yaml.safe_load(f)
            report["objective_to_minimize"] = last_test_loss
            extra = report.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            extra["last_valid_loss"] = last_valid_loss
            extra["valid_losses"] = valid_losses
            extra["last_test_loss"] = last_test_loss
            extra["test_losses"] = test_losses
            extra["eval_steps"] = eval_steps
            report["extra"] = extra
            if "info_dict" not in report and "extra" in report:
                report["info_dict"] = report["extra"]

        user_result = {
            "objective_to_minimize": last_test_loss,
            "cost": report.get("cost"),
            "info_dict": report.get("extra") or report.get("info_dict") or {},
        }
        user_result["info_dict"]["valid_loss"] = last_valid_loss
        user_result["info_dict"]["test_loss"] = last_test_loss

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
    app.run(main)
