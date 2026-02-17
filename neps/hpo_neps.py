"""
Hyperparameter Optimization using NePS.

A flexible way to define hyperparameter ranges in a config file
and let NePS search over those ranges to find the best configuration (only sampling configurations currently via running NePS asynchronously).

Config file format for hyperparameters (add a 'neps_search_space' section):
```yaml
neps_search_space:
  lr:
    type: float
    lower: 1e-5
    upper: 1e-2
    log: true  
```

Usage:
    python hpo_neps.py --config path/to/config.yaml --max_evaluations 50
"""

import argparse
import logging
import os
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import neps
import torch
import yaml


import utils
from checkpoint_utils import save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model
from scripts.probe_batch_size import get_target_global_batch_for_tokens
from torch_utils import destroy_ddp, pytorch_setup
from utils import print_master


def load_config_with_search_space(config_path: str) -> tuple[dict, dict]:
    """
    Load a config file and extract the NePS search space definition.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Tuple of (base_config_dict, search_space_dict)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    search_space_def = config.pop('neps_search_space', {})

    return config, search_space_def


def create_neps_space(search_space_def: dict) -> dict:
    """
    Create a NePS search space from a dictionary definition.

    Args:
        search_space_def: Dictionary defining the search space. Each key is a
            hyperparameter name, and the value is a dict with 'type' and other params.

    Returns:
        Dictionary of NePS parameter objects.
    """
    space = {}

    for param_name, param_def in search_space_def.items():
        param_type = param_def.get('type', 'float')
        default_val = param_def.get('default')
        prior_kw = {}
        if default_val is not None:
            prior_kw['prior'] = default_val
            prior_kw['prior_confidence'] = param_def.get(
                'prior_confidence', 'medium')

        if param_type == 'float':
            space[param_name] = neps.Float(
                lower=param_def['lower'],
                upper=param_def['upper'],
                log=param_def.get('log', False),
                **prior_kw,
            )
        elif param_type == 'int' or param_type == 'integer':
            space[param_name] = neps.Integer(
                lower=param_def['lower'],
                upper=param_def['upper'],
                log=param_def.get('log', False),
                **prior_kw,
            )
        elif param_type == 'categorical':
            choices = tuple(param_def['choices'])
            cat_prior_kw = {}
            if default_val is not None:
                try:
                    idx = choices.index(default_val)
                    cat_prior_kw['prior'] = idx
                    cat_prior_kw['prior_confidence'] = param_def.get(
                        'prior_confidence', 'medium')
                except ValueError:
                    pass
            space[param_name] = neps.Categorical(
                choices=choices,
                **cat_prior_kw,
            )
        elif param_type == 'constant':
            space[param_name] = neps.Constant(value=param_def['value'])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return space


def create_pipeline_space(search_space_def: dict):
    """
    Create a NePS PipelineSpace from a dictionary definition.

    Args:
        search_space_def: Dictionary defining the search space (same format as
            create_neps_space).

    Returns:
        A PipelineSpace instance suitable for neps.run(pipeline_space=...).
    """
    space_dict = create_neps_space(search_space_def)

    class DynamicPipelineSpace(neps.PipelineSpace):
        """Pipeline space built from config; attributes set below."""

    for name, value in space_dict.items():
        setattr(DynamicPipelineSpace, name, value)

    return DynamicPipelineSpace()


def train_with_config(
    base_config: dict,
    pipeline_directory: str,
    **sampled_hyperparams,
) -> float:
    """
    Train a model with the given configuration and sampled hyperparameters.

    Args:
        base_config: Base configuration dictionary.
        pipeline_directory: Directory for checkpoints/logs for this trial.
        **sampled_hyperparams: Hyperparameters sampled by NePS.

    Returns:
        Validation loss after training.
    """
    config = base_config.copy()
    config.update(sampled_hyperparams)

    config['out_dir'] = pipeline_directory
    config['wandb_dir'] = pipeline_directory

    if 'fused_optim' not in config:
        config['fused_optim'] = False

    if 'n_tokens' in config:
        n_tokens = int(config['n_tokens'])
        seq_len = config.get('seq_len', 2048)
        micro_batch = config.get('micro_batch_size', 16)
        grad_accum = config.get('grad_accumulation_steps', 16)
        # world_size unknown until pytorch_setup; use 1 for initial config
        config['steps_budget'] = max(
            1, n_tokens // (seq_len * micro_batch * grad_accum))

    cfg, _ = utils.dict_to_config(config)

    rank, world_size, device, master_process = pytorch_setup(cfg)

    if 'n_tokens' in config and world_size >= 1:
        n_tokens = int(config['n_tokens'])
        global_batch = cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
        steps_budget = max(1, n_tokens // (cfg.seq_len * global_batch))
        cfg = cfg._replace(steps_budget=steps_budget)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if master_process:
        utils.maybe_make_dir(cfg)

    if cfg.use_wandb and master_process:
        utils.init_wandb(cfg)

    trainloader, validloader = get_dataloaders(cfg)

    model, _, _ = construct_model(cfg)

    ckpt = None
    engine = TorchEngine(model, cfg, device, local_rank, ckpt)

    steps_budget = cfg.steps_budget
    if cfg.scheduler == "linear_cooldown":
        steps_budget = cfg.resume_step + engine.scheduler.cooldown_steps

    micro_step_budget = steps_budget * cfg.grad_accumulation_steps
    if micro_step_budget > len(trainloader):
        raise ValueError("trainloader too short!")

    step_start = cfg.resume_step if cfg.resume else 0
    micro_step_start = step_start * cfg.grad_accumulation_steps

    print_master(
        f"=== Start Training from step: {step_start}/{steps_budget}, "
        f"micro_step: {micro_step_start}/{micro_step_budget} ==="
    )
    print_master(f"Sampled hyperparameters: {sampled_hyperparams}")

    metrics = defaultdict(list)
    train_loss_array = []

    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        train_loss = engine.step(micro_batch)
        train_loss_array.append(train_loss)

        valid_loss = None
        if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
            print_master("Evaluating on validation set")
            valid_loss = engine.eval(validloader)

        if master_process and step % cfg.log_every_steps == 0 and is_step:
            utils.log(
                cfg, metrics, micro_step, train_loss, train_loss_array,
                valid_loss, engine.optimizer, world_size
            )
            train_loss_array = []

        #
        if (
            master_process
            and cfg.save_intermediate_checkpoints
            and step % cfg.save_every_steps == 0
            and is_step
        ):
            save_checkpoint(step, model, engine, cfg, metrics)

    final_valid_loss = float('inf')
    if getattr(cfg, 'eval', True):
        print_master("Final evaluation on validation set")
        final_valid_loss = engine.eval(validloader)
        if master_process:
            metrics["valid/loss"].append(final_valid_loss)

    print_master(f"=== Training Completed! Valid loss: {final_valid_loss} ===")

    if master_process and cfg.save_last_checkpoint:
        save_checkpoint(step, model, engine, cfg, metrics)

    destroy_ddp()

    return final_valid_loss


def adjust_n_heads_for_rotary(d_model: int, n_heads: int, available_heads: list) -> int:
    """
    Adjust n_heads so that d_model % n_heads == 0 for rotary embeddings.

    Args:
        d_model: Model dimension.
        n_heads: Sampled number of heads.
        available_heads: List of valid head counts to choose from.

    Returns:
        Adjusted n_heads that divides d_model evenly.
    """
    if d_model % n_heads == 0:
        return n_heads

    valid_heads = [h for h in available_heads if d_model % h == 0]

    if not valid_heads:
        valid_heads = [h for h in range(1, d_model + 1) if d_model % h == 0]

    closest = min(valid_heads, key=lambda h: abs(h - n_heads))

    if closest != n_heads:
        logging.info(
            f"Adjusted n_heads from {n_heads} to {closest} "
            f"(d_model={d_model} requires d_model % n_heads == 0)"
        )

    return closest


def create_evaluate_pipeline(base_config: dict, search_space_def: dict):
    """
    Create an evaluation function for NePS that uses the base config.

    Args:
        base_config: Base configuration dictionary.
        search_space_def: Search space definition for extracting valid choices.

    Returns:
        A function that NePS can call to evaluate a configuration.
    """
    def evaluate_pipeline(pipeline_directory: str, **hyperparams) -> float:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            pipeline_directory: NePS provides this directory for trial outputs.
            **hyperparams: Sampled hyperparameters from the search space.

        Returns:
            Validation loss (objective to minimize).
        """
        try:
            if 'd_model' in hyperparams and 'n_heads' in hyperparams:
                # NePS may sample raw ints: d_model raw 1-16 -> actual = raw * 64; n_heads raw 1-8 -> actual = raw * 2
                d_model_raw = int(hyperparams['d_model'])
                n_heads_raw = int(hyperparams['n_heads'])
                d_model_scale = search_space_def.get(
                    'd_model', {}).get('scale', 64)
                n_heads_scale = search_space_def.get(
                    'n_heads', {}).get('scale', 2)
                d_model = d_model_raw * d_model_scale
                n_heads = n_heads_raw * n_heads_scale
                n_heads_upper = search_space_def.get(
                    'n_heads', {}).get('upper', 8)
                available_heads = [
                    i * n_heads_scale for i in range(1, n_heads_upper + 1)]
                adjusted_heads = adjust_n_heads_for_rotary(
                    d_model, n_heads, available_heads)
                hyperparams['d_model'] = d_model
                hyperparams['n_heads'] = adjusted_heads

            full_config = base_config.copy()
            full_config.update(hyperparams)

            n_tokens = full_config.get('n_tokens')
            if n_tokens is not None:
                n_tokens = int(n_tokens)
                frac = random.uniform(0.001, 0.005)
                valid_from_frac = max(1, int(n_tokens * frac))
                seq_len = full_config.get('seq_len', 2048)
                target_global_batch = get_target_global_batch_for_tokens(n_tokens)
                min_valid_tokens = (seq_len + 1) * target_global_batch
                full_config['valid_tokens'] = max(valid_from_frac, min_valid_tokens)

            config_path = Path(pipeline_directory) / 'config.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False,
                          sort_keys=False)

            logging.info(f"Saved full config to {config_path}")

            # Only sample configurations for now
            return None
            loss = train_with_config(
                base_config=base_config,
                pipeline_directory=pipeline_directory,
                **hyperparams,
            )
            return loss
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Return a high loss for failed configurations
            return float('inf')

    return evaluate_pipeline


def run_hpo(
    config_path: str,
    root_directory: str,
    max_evaluations: int = 50,
    optimizer: Any = None,
) -> None:
    """
    Run hyperparameter optimization using NePS.

    Args:
        config_path: Path to the config file with neps_search_space section.
        root_directory: Directory to store NePS results.
        max_evaluations: Maximum number of configurations to evaluate.
        optimizer: NePS optimizer to use (default: Bayesian Optimization).
    """
    logging.basicConfig(level=logging.INFO)

    base_config, search_space_def = load_config_with_search_space(config_path)

    if not search_space_def:
        raise ValueError(
            "No 'neps_search_space' section found in config. "
            "Please define hyperparameter ranges to search over."
        )

    pipeline_space = create_pipeline_space(search_space_def)

    logging.info(f"Search space: {list(pipeline_space.get_attrs().keys())}")
    logging.info(f"Base config loaded with {len(base_config)} parameters")

    has_priors = any(
        p.get("default") is not None for p in search_space_def.values()
    )
    if optimizer is neps.algorithms.random_search and has_priors:
        optimizer = partial(optimizer, use_priors=True)

    evaluate_fn = create_evaluate_pipeline(base_config, search_space_def)

    neps.run(
        evaluate_pipeline=evaluate_fn,
        pipeline_space=pipeline_space,
        root_directory=root_directory,
        evaluations_to_spend=max_evaluations,
        optimizer=optimizer,
    )


def main():
    parser = argparse.ArgumentParser(description="Run HPO with NePS")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file with neps_search_space section"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="neps/neps_configs/Neps_New_Test",
        help="Directory to store NePS results"
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=50,
        help="Maximum number of configurations to evaluate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="random",
        choices=["bo", "random", "regularized_evolution"],
        help="NePS optimizer to use"
    )

    args = parser.parse_args()

    optimizer_map = {
        "bo": None,
        "random": neps.algorithms.random_search,
        "regularized_evolution": neps.algorithms.neps_regularized_evolution,
    }
    optimizer = optimizer_map.get(args.optimizer)

    run_hpo(
        config_path=args.config,
        root_directory=args.root_dir,
        max_evaluations=args.max_evaluations,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
