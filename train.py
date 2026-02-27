"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict
import copy
import gc
import json
import os
import time
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
import utils
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from data import get_dataloaders
from checkpoint_utils import save_checkpoint, maybe_load_checkpoint, update_config_yaml_for_resume, get_checkpoint_save_steps
from models import construct_model
from models.params_flops import format_flops
from engine import TorchEngine
from scripts.probe_batch_size import (
    get_batch_config,
    get_target_global_batch_for_tokens,
    get_recommended_num_gpus,
)
import math

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

    CFG_PATH = FLAGS.config
    cfg, _ = utils.load_config(CFG_PATH)

    local_rank, world_size, device, master_process = pytorch_setup(cfg)
    print("Save NePS report: ", FLAGS.save_neps_report)
    if master_process:
        utils.maybe_make_dir(cfg)

    if cfg.use_wandb and master_process:
        utils.init_wandb(cfg)
        utils.log_job_info()

    # Barrier: ensure rank 0 finishes wandb before other ranks proceed to DDP.
    if dist.is_initialized():
        dist.barrier()

    # Load checkpoint
    ckpt = maybe_load_checkpoint(cfg)

    # Model (construct before DDP/compile so we can probe batch sizes)
    model, _, compute_flops = construct_model(cfg)
    n_params = (
        model.count_params(non_embedding=False)
        if hasattr(model, 'count_params')
        else sum(p.numel() for p in model.parameters())
    )

    # Probe batch size: find max micro_batch_size that fits, then set grad_accum from model size.
    # This is mainly intended for standalone runs. When configs are preprocessed
    # (e.g. by a resource-planning script) with fixed micro_batch_size and
    # grad_accumulation_steps, set cfg.auto_batch_size = False to skip probing.
    if getattr(cfg, "auto_batch_size", False) and torch.cuda.is_available():
        model_on_device = model.to(device)
        micro_bs, _ = get_batch_config(
            cfg, model_on_device, device, world_size)
        if dist.is_initialized():
            micro_bs_t = torch.tensor(
                [micro_bs], dtype=torch.int64, device=device)
            dist.all_reduce(micro_bs_t, op=dist.ReduceOp.MIN)
            micro_bs = micro_bs_t.item()
        target_global = get_target_global_batch_for_tokens(cfg.n_tokens)
        grad_accum = max(
            1, (target_global + micro_bs * world_size -
                1) // (micro_bs * world_size)
        )
        recommended_gpus = get_recommended_num_gpus(target_global, micro_bs)
        if world_size >= recommended_gpus:
            grad_accum = 1
        cfg = cfg._replace(micro_batch_size=micro_bs,
                           grad_accumulation_steps=grad_accum)
        print_master(
            f'Probe batch size: micro_batch_size={micro_bs}, grad_accumulation_steps={grad_accum} '
            f'(Token Budget={cfg.n_tokens:,}, target global batch={target_global}, '
            f'recommended GPUs={recommended_gpus}, using {world_size})'
        )
        # Clear GPU state after probe so training (DDP, torch.compile, optimizer) starts
        # with a clean allocation pattern and avoids fragmentation OOMs.
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print_master(
            'Using config micro_batch_size and grad_accumulation_steps (no auto batch-size probing).')

    # Dataset (uses the potentially updated micro_batch_size)
    trainloader, validloader = get_dataloaders(cfg)

    global_batch = cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
    steps_budget = max(1, cfg.n_tokens // (cfg.seq_len * global_batch))
    cfg = cfg._replace(steps_budget=steps_budget)
    tokens_per_step = cfg.seq_len * global_batch
    save_steps_set = get_checkpoint_save_steps(
        n_tokens=cfg.n_tokens,
        steps_budget=steps_budget,
        tokens_per_step=tokens_per_step,
        max_tokens_budget=getattr(cfg, "max_tokens_budget", None),
    )
    # Engine (handles DDP wrapping, torch.compile, optimizer, scheduler)
    engine = TorchEngine(model, cfg, device, local_rank, ckpt)

    # If we are just cooling down, we set budget = resume + cooldown
    steps_budget = (
        cfg.steps_budget if cfg.scheduler != 'linear_cooldown' else cfg.resume_step +
        engine.scheduler.cooldown_steps
    )

    micro_step_budget = steps_budget * cfg.grad_accumulation_steps
    if micro_step_budget > len(trainloader):
        raise ValueError('trainloader too short!')

    # Start the dataloader from the correct micro-batch
    step_start = cfg.resume_step if cfg.resume else 0
    micro_step_start = step_start * cfg.grad_accumulation_steps
    print_master(
        f'=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ==='
    )

    # Bookkeeping
    metrics = defaultdict(list)
    train_loss_array = []
    training_start_time = time.time()
    last_train_loss = None
    last_completed_step = step_start  # tracks the last step we actually trained on

    # Training
    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        # Train
        train_loss = engine.step(micro_batch)
        last_train_loss = train_loss.item() if hasattr(
            train_loss, 'item') else float(train_loss)
        train_loss_array.append(train_loss)
        if is_step:
            last_completed_step = step

        # Log
        if master_process and step % cfg.log_every_steps == 0 and is_step:
            utils.log(cfg, metrics, micro_step, train_loss,
                      train_loss_array, engine.optimizer, world_size)
            train_loss_array = []

        # Checkpoint
        if master_process and cfg.save_intermediate_checkpoints and is_step:
            if save_steps_set is not None:
                do_save = step in save_steps_set
            else:
                do_save = step % cfg.save_every_steps == 0
            if do_save:
                save_checkpoint(step, model, engine, cfg, metrics)

    # End of training: log and save checkpoint
    print_master('=== Training Completed! ===')
    if master_process and cfg.save_last_checkpoint:
        save_checkpoint(last_completed_step, model, engine, cfg, metrics)
        # Make config YAML ready for validation run
        try:
            update_config_yaml_for_resume(CFG_PATH, last_completed_step, cfg)
        except Exception as e:
            print_master(f'Could not update config for resume: {e}')

    # Save NePS report if requested
    if master_process and FLAGS.save_neps_report:
        neps_dir = FLAGS.neps_report_dir or FLAGS.neps_root_dir
        if not neps_dir:
            print_master('Cannot save NePS report: neither neps_root_dir nor neps_report_dir set')
        else:
            training_duration = time.time() - training_start_time
            global_batch = cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
            n_data = getattr(cfg, 'n_tokens', None)
            result = {
                'objective_to_minimize': last_train_loss,
                'cost': compute_flops,
                'evaluation_duration': training_duration,
                'info_dict': {
                    'duration': training_duration,
                    'compute_formatted': format_flops(compute_flops) if compute_flops else None,
                    'train_loss': last_train_loss,
                    'n_data': n_data,
                    'n_param': n_params,
                    'flops': compute_flops,
                    'micro_batch_size': cfg.micro_batch_size,
                    'global_batch_size': global_batch,
                    'time_duration_seconds': training_duration,
                    'final_step': last_completed_step,
                    'steps_budget': steps_budget,
                    'recommended_num_gpus': get_recommended_num_gpus(
                        get_target_global_batch_for_tokens(cfg.n_tokens),
                        cfg.micro_batch_size,
                    ),
                }
            }

            try:
                from neps import save_pipeline_results
                print(f'Result: {result}')
                print(f'Pipeline ID: {FLAGS.config_number}')
                save_pipeline_results(
                    user_result=result,
                    pipeline_id=FLAGS.config_number,
                    root_directory=Path(neps_dir),
                )
                print_master(f'Saved NePS report to {neps_dir}')
                
            except Exception as e:
                print_master(f'Failed to save NePS report: {e}')

    # DDP slaughtering
    destroy_ddp()


if __name__ == '__main__':
    app.run(main)
