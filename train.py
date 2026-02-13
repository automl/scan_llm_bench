"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict
import copy
import gc
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
from checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from models import construct_model
from models.params_flops import format_flops
from engine import TorchEngine
from scripts.probe_batch_size import (
    get_batch_config,
    get_target_global_batch_for_param_count,
    _model_param_count,
)
import math

flags.DEFINE_string('config', 'config/config.yaml',
                    'Path to config.yaml file.')
flags.DEFINE_string('config_overrides', None,
                    'Path to YAML with config overrides (merged on top of base config).')
flags.DEFINE_integer(
    'job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
flags.DEFINE_integer(
    'config_number', None,
    'Config index for sequential NePS runs (unique checkpoint path per config).')
flags.DEFINE_string('neps_root_dir', None,
                    'Path to directory where NePS report will be saved.')
flags.DEFINE_bool(
    'save_neps_report', False,
    'If True and config_overrides is set, save report.yaml to the NePS config directory.')
FLAGS = flags.FLAGS


# def _get_batch_sizes(cfg):
#     """Generate descending list of micro_batch_size candidates to try.
#     If cfg has a 'batch_sizes' field (list), use it directly.
#     Otherwise, halve cfg.micro_batch_size down to 1."""
#     if hasattr(cfg, 'batch_sizes') and cfg.batch_sizes is not None:
#         sizes = sorted(cfg.batch_sizes, reverse=True)
#         return [s for s in sizes if s >= 1]
#     # Auto-generate by halving
#     sizes = []
#     bs = cfg.micro_batch_size
#     while bs >= 1:
#         sizes.append(bs)
#         bs //= 2
#     return sizes


# def _is_oom_error(exc):
#     """Check if an exception is a CUDA out-of-memory error."""
#     if isinstance(exc, torch.cuda.OutOfMemoryError):
#         return True
#     if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
#         return True
#     return False


# def _cleanup_gpu():
#     """Free GPU memory after a failed probe attempt."""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()


# def find_working_batch_size(cfg, model, device, local_rank, ckpt, world_size, n_probe_steps=3):
#     """
#     Probe descending micro_batch_size values to find the largest that fits in GPU memory.

#     Runs a few forward+backward steps with each candidate. On OOM, cleans up and tries
#     the next smaller size. Model state is saved before probing and restored after.

#     Args:
#         cfg: Config namedtuple.
#         model: The model (NOT yet compiled, NOT yet wrapped in DDP).
#         device: Device string (e.g. 'cuda:0').
#         local_rank: Local rank for DDP.
#         ckpt: Checkpoint dict or None.
#         world_size: Number of processes.
#         n_probe_steps: Number of micro-steps to run for each probe.

#     Returns:
#         int: The largest working micro_batch_size.
#     """
#     batch_sizes = _get_batch_sizes(cfg)
#     print_master(f'Auto batch size tuning: trying {batch_sizes}')

#     # Save model state before probing so we can restore it after
#     model_state = copy.deepcopy(model.state_dict())

#     working_bs = None

#     for bs in batch_sizes:
#         print_master(f'  Probing micro_batch_size={bs}...')

#         # Restore clean model state for each probe
#         model.load_state_dict(model_state)
#         model_on_device = model.to(device)

#         # Build a temporary dataloader with this batch size
#         probe_cfg = cfg._replace(
#             micro_batch_size=bs,
#             torch_compile=False,  # Don't compile during probing
#         )
#         try:
#             probe_loader, _ = get_dataloaders(probe_cfg)
#         except Exception as e:
#             print_master(f'  Failed to build dataloader with bs={bs}: {e}')
#             _cleanup_gpu()
#             continue

#         # Build a lightweight engine (no compile, no checkpoint restore)
#         # We use a minimal forward+backward to test memory
#         try:
#             from torch.nn import CrossEntropyLoss
#             from contextlib import nullcontext

#             device_type = 'cuda' if 'cuda' in device else 'cpu'
#             ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
#             ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
#             criterion = CrossEntropyLoss()

#             # Run probe steps
#             probe_iter = iter(probe_loader)
#             for probe_step in range(n_probe_steps):
#                 batch = next(probe_iter)
#                 inputs = batch['input_ids'][:, :cfg.seq_len]
#                 targets = batch['input_ids'][:, 1:(cfg.seq_len + 1)]
#                 if 'cuda' in device:
#                     inputs = inputs.to(device, non_blocking=True)
#                     targets = targets.to(device, non_blocking=True)

#                 with ctx:
#                     output = model_on_device(inputs, None)
#                     logits = getattr(output, 'logits', output)
#                     loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

#                 loss.backward()
#                 model_on_device.zero_grad(set_to_none=True)

#             # Success: this batch size works
#             print_master(f'  micro_batch_size={bs} fits in memory.')
#             working_bs = bs
#             break

#         except Exception as e:
#             if _is_oom_error(e):
#                 print_master(f'  micro_batch_size={bs} OOM. Trying smaller...')
#             else:
#                 print_master(f'  micro_batch_size={bs} failed with non-OOM error: {e}. Trying smaller...')
#         finally:
#             # Clean up probe resources
#             del probe_loader
#             model_on_device.cpu()
#             _cleanup_gpu()

#     # Restore original model state (clean, on CPU)
#     model.load_state_dict(model_state)
#     del model_state

#     # Synchronize chosen batch size across ranks in DDP.
#     # Each rank finds its own largest working bs; we take the MIN across all ranks
#     # so every rank uses a size that fits on the most memory-constrained GPU.
#     if dist.is_initialized():
#         # Use 0 to signal "all sizes OOM" so MIN picks the smallest positive value
#         local_bs = working_bs if working_bs is not None else 0
#         bs_tensor = torch.tensor([local_bs], dtype=torch.int64, device=device)
#         dist.all_reduce(bs_tensor, op=dist.ReduceOp.MIN)
#         working_bs = bs_tensor.item()
#         # Move the tensor back and free GPU memory
#         del bs_tensor
#         _cleanup_gpu()

#     if working_bs is None or working_bs == 0:
#         raise RuntimeError(
#             f'All batch sizes {batch_sizes} caused OOM on at least one rank. '
#             'Reduce model size or use a GPU with more memory.'
#         )

#     return working_bs


def main(_):

    CFG_PATH = FLAGS.config
    cfg, _ = utils.load_config(CFG_PATH, overrides_path=FLAGS.config_overrides)

    local_rank, world_size, device, master_process = pytorch_setup(cfg)
    print("Save NePS report: ", FLAGS.save_neps_report)
    if master_process:
        utils.maybe_make_dir(cfg)

    if cfg.use_wandb and master_process:
        utils.init_wandb(cfg)
        utils.log_job_info()

    # Barrier: ensure rank 0 finishes wandb before other ranks proceed to DDP.
    # Without this, rank 1 reaches DDP init first and waits for rank 0, causing NCCL timeout.
    if dist.is_initialized():
        dist.barrier()

    # Load checkpoint
    ckpt = maybe_load_checkpoint(cfg)

    # Model (construct before DDP/compile so we can probe batch sizes)
    model, _, compute_flops = construct_model(cfg)

    # Probe batch size: find max micro_batch_size that fits, then set grad_accum from model size.
    # Target global batch from param count: <5M→16, 5–20M→64, 20–70M→128, 70M+→256.
    if torch.cuda.is_available():
        model_on_device = model.to(device)
        micro_bs, _ = get_batch_config(
            cfg, model_on_device, device, world_size)
        if dist.is_initialized():
            micro_bs_t = torch.tensor(
                [micro_bs], dtype=torch.int64, device=device)
            dist.all_reduce(micro_bs_t, op=dist.ReduceOp.MIN)
            micro_bs = micro_bs_t.item()
        n_params = _model_param_count(model)
        target_global = get_target_global_batch_for_param_count(n_params)
        grad_accum = max(
            1, (target_global + micro_bs * world_size -
                1) // (micro_bs * world_size)
        )
        cfg = cfg._replace(micro_batch_size=micro_bs,
                           grad_accumulation_steps=grad_accum)
        print_master(
            f'Probe batch size: micro_batch_size={micro_bs}, grad_accumulation_steps={grad_accum} '
            f'(model params={n_params:,}, target global batch={target_global})'
        )
        # Clear GPU state after probe so training (DDP, torch.compile, optimizer) starts
        # with a clean allocation pattern and avoids fragmentation OOMs.
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print_master(
            'CUDA not available; using config micro_batch_size and grad_accumulation_steps.')

    # Dataset (uses the potentially updated micro_batch_size)
    trainloader, validloader = get_dataloaders(cfg)

    global_batch = cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
    steps_budget = max(1, cfg.n_tokens // (cfg.seq_len * global_batch))
    cfg = cfg._replace(steps_budget=steps_budget)
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
    last_valid_loss = None

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

        # Eval
        valid_loss = None
        if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
            print_master('Evaluating on validation set')
            valid_loss = engine.eval(validloader)
            last_valid_loss = valid_loss

        # Log
        if master_process and step % cfg.log_every_steps == 0 and is_step:
            utils.log(cfg, metrics, micro_step, train_loss,
                      train_loss_array, valid_loss, engine.optimizer, world_size)
            train_loss_array = []

        # Checkpoint
        if master_process and cfg.save_intermediate_checkpoints and step % cfg.save_every_steps == 0 and is_step:
            save_checkpoint(step, model, engine, cfg, metrics)

    # End of training: log and save checkpoint
    print_master('=== Training Completed! ===')
    if master_process and cfg.save_last_checkpoint:
        save_checkpoint(step, model, engine, cfg, metrics)

    # Save NePS report if requested
    if master_process and FLAGS.save_neps_report:
        training_duration = time.time() - training_start_time
        global_batch = cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
        result = {
            'objective_to_minimize': last_train_loss,
            'cost': compute_flops,
            'info_dict': {
                'duration': training_duration,
                'compute_formatted': format_flops(compute_flops) if compute_flops else None,
                'train_loss': last_train_loss,
                'valid_loss': last_valid_loss,
                'n_tokens': getattr(cfg, 'n_tokens', None),
                'micro_batch_size': cfg.micro_batch_size,
                'global_batch_size': global_batch,
                'time_duration_seconds': training_duration,
                'final_step': step,
                'steps_budget': steps_budget,
            }
        }

        try:
            from neps import save_pipeline_results
            print(f'Saving NePS report to {FLAGS.neps_root_dir}')
            print(f'Result: {result}')
            print(f'Pipeline ID: {FLAGS.config_number}')
            save_pipeline_results(
                user_result=result,
                pipeline_id=FLAGS.config_number,
                root_directory=Path(FLAGS.neps_root_dir),
            )
            print_master(f'Saved NePS report to {FLAGS.neps_root_dir}')
        except Exception as e:
            print_master(f'Failed to save NePS report: {e}')

    # DDP slaughtering
    destroy_ddp()


if __name__ == '__main__':
    app.run(main)
