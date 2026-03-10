import os

import torch

from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Subset

from data.datasamplers import StatefulSequentialSampler, StatefulRandomSampler, StatefulDistributedSampler
from data.bin_chunk_dataset import _is_bin_chunk_format, BinChunkDataset, BinChunkMapDataset


def get_dataloaders(cfg):
  """Load trainset and perhaps validset. Returns correspondent DataLoaders."""

  def collate_fn(batch):
    return {
      'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
      'docs_lengths': [x['docs_lengths'].tolist() for x in batch],
    }

  ddp = dist.is_initialized()
  world_size = dist.get_world_size() if ddp else 1
  rank = dist.get_rank() if ddp else 0

  # Target sequence length: seq_len + 1 (input tokens + 1 for next-token prediction target)
  # Ensure all sequences have the same length for batching and model compatibility
  target_seq_len = cfg.seq_len + 1

  if _is_bin_chunk_format(cfg.trainset_path):
    shuffle = cfg.sampler in ('random', 'stateful_random')
    start_iter = 0
    if cfg.sampler.startswith('stateful') and cfg.resume:
      start_iter = cfg.resume_step * cfg.grad_accumulation_steps * cfg.micro_batch_size

    train_set = BinChunkDataset(
      cfg.trainset_path,
      shuffle=shuffle,
      seed=getattr(cfg, 'sampler_seed', 42),
      start_iter=start_iter,
      world_size=world_size,
      rank=rank,
      target_seq_len=target_seq_len,
      pad_token_id=0,  # GPT-2 uses 0 as pad token
      vocab_size=getattr(cfg, 'vocab_size', None), 
    )

    trainloader = DataLoader(
      train_set,
      batch_size=cfg.micro_batch_size,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=True if cfg.num_workers > 0 else False,
    )
  else:
    # Dataset (Arrow format) - use existing  logic
    train_set = load_from_disk(cfg.trainset_path)
    if not isinstance(train_set, Dataset):
      raise ValueError('dataset should be a datasets.Dataset')

    train_sampler = _get_sampler(train_set, cfg)

    trainloader = DataLoader(
      train_set,
      sampler=train_sampler,
      batch_size=cfg.micro_batch_size,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=True if cfg.num_workers > 0 else False,
      collate_fn=collate_fn if 'docs_lengths' in train_set.column_names else None,
    )

  if not cfg.validset_path:
    validloader = None
    test_loader = None
  else:
    if _is_bin_chunk_format(cfg.validset_path):
      valid_set = BinChunkMapDataset(
        cfg.validset_path,
        target_seq_len=target_seq_len,
        pad_token_id=0,
        vocab_size=getattr(cfg, 'vocab_size', None),
      )
    else:
      valid_set = load_from_disk(cfg.validset_path)
      if not isinstance(valid_set, Dataset):
        raise ValueError("'dataset' should be a datasets.Dataset")
    test_path = cfg.validset_path.replace('validation', 'test')
    print(f"test data path: {test_path}.")
    if _is_bin_chunk_format(test_path):
      test_set = BinChunkMapDataset(
        test_path,
        target_seq_len=target_seq_len,
        pad_token_id=0,
        vocab_size=getattr(cfg, 'vocab_size', None),
      )
    else:
      test_set = load_from_disk(test_path) if os.path.exists(test_path) else None

    if getattr(cfg, 'valid_tokens', False):
      min_valid_tokens = 4_000_000 # for enough samples to get stable eval results, especially for larger models
      valid_tokens = max(cfg.valid_tokens, min_valid_tokens)
      valid_rows = valid_tokens // (cfg.seq_len + 1)
      test_rows = valid_tokens // (cfg.seq_len + 1)

      if hasattr(valid_set, 'take'):
        valid_set = valid_set.take(valid_rows)
        test_set = test_set.take(test_rows) if test_set else None
      else:
        valid_set = Subset(valid_set, range(min(valid_rows, len(valid_set))))
        test_set = Subset(test_set, range(min(test_rows, len(test_set)))) if test_set else None

    if ddp:
      valid_sampler = DistributedSampler(valid_set, drop_last=True)
      test_sampler = DistributedSampler(test_set, drop_last=True) if test_set else None
    else:
      valid_sampler = SequentialSampler(valid_set)
      test_sampler = SequentialSampler(test_set) if test_set else None

    if hasattr(valid_set, 'column_names'):
      valid_columns = valid_set.column_names
      test_columns = test_set.column_names if test_set else None
    elif hasattr(valid_set, 'dataset') and hasattr(valid_set.dataset, 'column_names'):
      valid_columns = valid_set.dataset.column_names
      test_columns = test_set.dataset.column_names if test_set else None
    else:
      valid_columns = []
      test_columns = []
      
    validloader = DataLoader(
      valid_set,
      batch_size=cfg.micro_batch_size,
      drop_last=True,  # makes eval with DDP easier
      shuffle=False,
      sampler=valid_sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      collate_fn=collate_fn if 'docs_lengths' in valid_columns else None,
    )
    test_loader = None
    if test_set is not None:
      test_loader = DataLoader(
        test_set,
        batch_size=cfg.micro_batch_size,
        drop_last=True,  # makes eval with DDP easier
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=collate_fn if 'docs_lengths' in test_columns else None,
      )

  return trainloader, validloader, test_loader


def _get_sampler(train_set, cfg):
  """Initlaizes a sampler for a torch.Dataloader.
  Options:
    - random sampler
    - sequential sampler
    - stateful random sampler
    - stateful sequential sampler
  We implement "stateful" sequential samplers for resuming training from a specified step.
  """
  ddp = dist.is_initialized()

  if cfg.sampler == 'random':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.sampler_seed, drop_last=True)
    else:
      sampler = RandomSampler(
        train_set, generator=torch.Generator().manual_seed(cfg.sampler_seed) if cfg.sampler_seed else None
      )

  elif cfg.sampler == 'sequential':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
    else:
      sampler = SequentialSampler(train_set)

  elif cfg.sampler == 'stateful_random':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      # TODO: allow support for drop_last=True!
      sampler = StatefulDistributedSampler(
        train_set, batch_size=cfg.micro_batch_size, seed=cfg.sampler_seed, start_iter=micro_step_start
      )
    else:
      sampler = StatefulRandomSampler(
        train_set, batch_size=cfg.micro_batch_size, shuffle=True, seed=cfg.sampler_seed, start_idx=micro_step_start
      )

  elif cfg.sampler == 'stateful_sequential':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      raise NotImplementedError('StatefulDistributedSampler currently needs a seed.')
    else:
      sampler = StatefulSequentialSampler(train_set, batch_size=cfg.micro_batch_size, start_idx=micro_step_start)

  else:
    raise NotImplementedError(f'Sampler {cfg.sampler} is not implemented.')

  return sampler
