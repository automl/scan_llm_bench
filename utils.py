import os
import yaml
import math
import shutil
import wandb
import torch

from itertools import product
from collections import namedtuple

from absl import flags

FLAGS = flags.FLAGS


def load_config(path, overrides_path=None):
    """
    Parse a yaml file and return the correspondent config as a namedtuple.
    If the config files has multiple entries, returns the one corresponding to job_idx.

    Args:
      path: Path to the base config YAML file.
      overrides_path: Optional path to a YAML file with config overrides.
                      Values in this file will override the base config.
    """

    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Remove neps_search_space if present (not a training config parameter)
    config_dict.pop('neps_search_space', None)

    # Apply overrides if provided (from NePS)
    if overrides_path is not None:
        with open(overrides_path, 'r') as file:
            overrides = yaml.safe_load(file)
        if overrides:
            config_dict.update(overrides)

    Config = namedtuple('Config', config_dict.keys())

    if FLAGS.job_idx is None:
        cfg = config_dict
        sweep_size = 1

    else:
        keys = list(config_dict.keys())
        values = [val if isinstance(val, list) else [val]
                  for val in config_dict.values()]
        combinations = list(product(*values))

        sweep_size = len(combinations)
        if FLAGS.job_idx >= sweep_size:
            raise ValueError(
                "job_idx exceeds the total number of hyperparam combinations.")

        combination = combinations[FLAGS.job_idx]
        cfg = {keys[i]: combination[i] for i in range(len(keys))}

    return Config(**cfg), sweep_size


def dict_to_config(config_dict):
    """
    Convert a dictionary to a Config namedtuple.
    Useful for creating configs (e.g., for NePS HPO).

    Args:
      config_dict: Dictionary containing config values.

    Returns:
      Tuple of (Config namedtuple, sweep_size=1)
    """
    Config = namedtuple('Config', config_dict.keys())
    return Config(**config_dict), 1


def init_wandb(cfg):
    """Initalizes a wandb run"""
    os.environ['WANDB__SERVICE_WAIT'] = '600'
    os.environ['WANDB_SILENT'] = 'true'

    if getattr(cfg, 'check_existing_wandb_run', False):
        if _matching_wandb_run_exists(cfg):
            raise FileExistsError(
                'A run with the same config exists. Aborting.')

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
               dir=cfg.wandb_dir, config=cfg._asdict())


def log_job_info():
    """Logs info about cluster job."""
    if FLAGS.job_cluster is not None and FLAGS.job_idx is not None:
        print(f'JOB_CLUSER = {FLAGS.job_cluster}')
        print(f'JOB_INDEX = {FLAGS.job_idx}')
        print(f'JOB_ID = {FLAGS.job_cluster}.{FLAGS.job_idx}')
        wandb.log({'JOB_CLUSTER': FLAGS.job_cluster})
        wandb.log({'JOB_INDEX': FLAGS.job_idx})
        wandb.log(
            {
                'JOB_ID': f'{FLAGS.job_cluster}.{FLAGS.job_idx}',
            }
        )


def _matching_wandb_run_exists(cfg):
    """Check for runs on wandb with the same config. Return True if such run exists."""

    api = wandb.Api()
    runs = api.runs(f'ajnico/{cfg.wandb_project}')

    # Extract important flags from configs
    to_match_config = {
        k: getattr(cfg, k)
        for k in {
            'trainset_path',
            'vocab_size',
            'seq_len',
            'eval',
            'validset_path',
            'eval_every_steps',
            'model',
            'd_model',
            'expand',
            'n_layers',
            'n_heads',
            'mlp_class',
            'tie_embeddings',
            'steps_budget',
            'micro_batch_size',
            'grad_accumulation_steps',
            'dtype',
            'optim',
            'lr',
            'weight_decay',
            'beta1',
            'beta2',
            'grad_clip',
            'eps',
            'scheduler',
            'warmup_steps',
            'cooldown_steps',
            'lr_start',
            'lr_end',
            'lr_end_pct',
            'sampler_seed',
            'seed',
        }
    }
    print('Checking for wand runs with the same config....')
    # print(f"Matching config: {to_match_config}\n\n")

    # Separate non-float and float keys
    non_float_config = {
        k: v for k, v in to_match_config.items() if not isinstance(v, float)}
    float_config = {k: v for k, v in to_match_config.items()
                    if isinstance(v, float)}

    # Step 1: Fetch and filter runs using non-float keys
    runs = api.runs(
        f'ajnico/{cfg.wandb_project}', filters={'$and': [{'config.{}'.format(k): v} for k, v in non_float_config.items()]}
    )
    if not runs:  # If no matches are found in the first stage
        return False

    # Step 2: Refine matches using float attributes only
    for run in runs:
        if all(math.isclose(v, run.config.get(k), rel_tol=1e-5, abs_tol=1e-5) for k, v in float_config.items()):
            print(f'Found matching wandb run with ID: {run.id}')
            return True

    return False


def get_exp_dir_path(cfg):
    """Build a exp_dir path from config. It supports job arrays."""
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    if FLAGS.job_idx is not None:  # subfolder for each job in the sweep
        exp_dir = os.path.join(exp_dir, f"job_idx_{FLAGS.job_idx}")
    elif FLAGS.job_cluster is not None:  # unique folder for single jobs
        exp_dir = os.path.join(exp_dir, f"job_{FLAGS.job_cluster}")
        if FLAGS.config_number is not None:  # distinguish configs in sequential NePS runs
            exp_dir = os.path.join(exp_dir, f"config_{FLAGS.config_number}")
    return exp_dir


def maybe_make_dir(cfg):
    """Creates an experiment directory if checkpointing is enabled"""
    if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
        return
    if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
        return

    exp_dir = get_exp_dir_path(cfg)

    if os.path.exists(exp_dir):
        if not cfg.over_write:
            raise ValueError(f'Found existing exp_dir at {exp_dir}.')
        print(f'Removing experiment dir: {exp_dir}')
        shutil.rmtree(exp_dir)

    print(f'Creating experiment directory: {exp_dir}')
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
        yaml.dump(cfg._asdict(), file, default_flow_style=False)


def _safe_exp(loss) -> float:
    """Compute exp(loss) for perplexity, avoiding OverflowError when loss is very large.
    When loss > ~709, exp overflows; cap to a large finite value for logging."""
    try:
        x = loss.item() if hasattr(loss, 'item') else float(loss)
        if math.isfinite(x) and x < 709:  # exp(709) ~ 1.8e307, near float max
            return math.exp(x)
        return float('inf')  # loss too high or nan/inf -> ppl = inf
    except (OverflowError, ValueError):
        return float('inf')


def log(cfg, metrics, micro_step, train_loss, train_loss_array, valid_loss, optimizer, world_size):
    """Update metrics, print to console, log on wandb."""

    if isinstance(train_loss_array, list):
        train_loss_avg = torch.stack(train_loss_array).mean().item()
    elif isinstance(train_loss_array, torch.Tensor):
        train_loss_avg = train_loss_array.item()

    new_metrics = {
        'micro_step': micro_step,
        'step': micro_step // cfg.grad_accumulation_steps,
        'tokens': micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
        'lr': optimizer.param_groups[0].get('lr', float('NaN')),
        'train/loss': train_loss.item(),
        'train/loss_avg': train_loss_avg,
        'train/ppl': _safe_exp(train_loss),
        'train/ppl_avg': _safe_exp(train_loss_avg),
    }

    if valid_loss is not None:
        new_metrics['valid/loss'] = valid_loss
        new_metrics['valid/ppl'] = _safe_exp(valid_loss)

    for k, v in new_metrics.items():
        metrics[k].append(v)

    if cfg.print_progress:
        msg = ' | '.join(
            f'{key}: {value:.3e}' if isinstance(value, float) else f'{key}: {value}' for key, value in new_metrics.items()
        )
        print(msg)

    if cfg.use_wandb:
        wandb.log(new_metrics)


def print_master(msg):
    """Prints only in master process if using multiple GPUs."""
    rank = os.environ.get('RANK', -1)
    ddp = int(rank) != -1
    master_process = (not ddp) or (int(rank) == 0)
    if master_process:
        print(msg)
