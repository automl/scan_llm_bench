"""
Dataset for SlimPajama preprocessed data in bin + index.json format.

Format: directory contains index.json and chunk-*.bin files.
- index.json: "chunks" list with "filename", "chunk_size", "chunk_bytes"; "config.data_format"
- .bin files: raw uint16 token IDs, no header, stored as (chunk_size, seq_len) row-major.

Use IterableDataset with chunked shuffling for memory efficiency.

The bin files may have variable sequence lengths per chunk, so `target_seq_len` is used
to enforce a fixed length (truncate longer, pad shorter with pad_token_id).
"""
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def _is_bin_chunk_format(dir_path: str) -> bool:
  """True if dir_path contains index.json and at least one .bin file."""
  p = Path(dir_path)
  if not p.is_dir():
    return False
  index_file = p / "index.json"
  if not index_file.exists():
    return False
  return any(p.glob("*.bin"))


def load_bin_chunk_index(dir_path: str) -> list[dict]:
  """Load index.json; return chunks list."""
  with open(Path(dir_path) / "index.json") as f:
    data = json.load(f)
  chunks = data["chunks"]
  if not chunks:
    raise ValueError(f"No chunks in index at {dir_path}")
  return chunks


def _normalize_seq_len(row_data: np.ndarray, target_seq_len: int, pad_token_id: int) -> np.ndarray:
  """Truncate or pad row_data to target_seq_len."""
  current_len = len(row_data)
  if current_len == target_seq_len:
    return row_data
  elif current_len > target_seq_len:
    return row_data[:target_seq_len]
  else:
    padded = np.full(target_seq_len, pad_token_id, dtype=row_data.dtype)
    padded[:current_len] = row_data
    return padded


def _clamp_to_vocab(row_data: np.ndarray, vocab_size: Optional[int]) -> np.ndarray:
  """Clamp token IDs to [0, vocab_size - 1]."""
  if vocab_size is None:
    return row_data
  return np.clip(row_data, 0, vocab_size - 1)


class BinChunkDataset(IterableDataset):
  """
  Memory-efficient IterableDataset for bin+index.json format.
  """

  def __init__(
    self,
    dir_path: str,
    shuffle: bool = True,
    seed: Optional[int] = None,
    start_iter: int = 0,
    world_size: int = 1,
    rank: int = 0,
    target_seq_len: Optional[int] = None,
    pad_token_id: int = 0,
    vocab_size: Optional[int] = None,
  ):
    self.dir_path = Path(dir_path)
    self.chunks = load_bin_chunk_index(dir_path)
    self.shuffle = shuffle
    self.seed = seed if seed is not None else 42
    self.start_iter = start_iter
    self.world_size = world_size
    self.rank = rank
    self.target_seq_len = target_seq_len
    self.pad_token_id = pad_token_id
    self.vocab_size = vocab_size

    # Compute total number of rows
    self._total_rows = sum(c["chunk_size"] for c in self.chunks)
    self.column_names = ["input_ids"]

  def __len__(self) -> int:
    """Approximate length (total rows / world_size). Used for progress estimation."""
    return self._total_rows // self.world_size

  def _get_chunk_indices_for_worker(self) -> list[int]:
    """
    Determine which chunk indices this worker should process.
    Handles both DDP (rank/world_size) and DataLoader workers.
    """
    all_chunk_indices = list(range(len(self.chunks)))

    if self.shuffle:
      rng = np.random.default_rng(self.seed)
      rng.shuffle(all_chunk_indices)

    rank_chunks = all_chunk_indices[self.rank::self.world_size]

    worker_info = get_worker_info()
    if worker_info is not None:
      worker_id = worker_info.id
      num_workers = worker_info.num_workers
      rank_chunks = rank_chunks[worker_id::num_workers]

    return rank_chunks

  def _iter_chunk(self, chunk_idx: int, rng: np.random.Generator) -> Iterator[dict]:
    """Yield all rows from a single chunk, optionally shuffled."""
    c = self.chunks[chunk_idx]
    path = self.dir_path / c["filename"]
    n_rows = c["chunk_size"]
    n_cols = (c["chunk_bytes"] // 2) // n_rows

    arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))

    row_indices = np.arange(n_rows)
    if self.shuffle:
      rng.shuffle(row_indices)

    for local_row in row_indices:
      row_data = np.array(arr[local_row], dtype=np.int64)
      # Normalize to target_seq_len if specified
      if self.target_seq_len is not None:
        row_data = _normalize_seq_len(row_data, self.target_seq_len, self.pad_token_id)
      # Clamp to vocab so embedding indices are in bounds (bin data can be uint16 > vocab_size)
      row_data = _clamp_to_vocab(row_data, self.vocab_size)
      yield {"input_ids": torch.from_numpy(row_data)}

    # Release memmap file handle
    del arr

  def __iter__(self) -> Iterator[dict]:
    """
    Iterate over all assigned chunks, yielding one row at a time.
    """
    chunk_indices = self._get_chunk_indices_for_worker()

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    rng = np.random.default_rng(self.seed + self.rank * 1000 + worker_id)

    iter_count = 0
    for chunk_idx in chunk_indices:
      for sample in self._iter_chunk(chunk_idx, rng):
        if iter_count >= self.start_iter:
          yield sample
        iter_count += 1


class BinChunkMapDataset(torch.utils.data.Dataset):
  """
  Map-style Dataset for bin+index.json format.
  Used for small datasets (validation) where random access is needed.
  For large training sets, BinChunkDataset should be used instead.
  """

  def __init__(
    self,
    dir_path: str,
    target_seq_len: Optional[int] = None,
    pad_token_id: int = 0,
    vocab_size: Optional[int] = None,
  ):
    self.dir_path = Path(dir_path)
    self.chunks = load_bin_chunk_index(dir_path)
    self.target_seq_len = target_seq_len
    self.pad_token_id = pad_token_id
    self.vocab_size = vocab_size
    # cumulative sizes for global index -> (chunk_idx, local_row)
    self.cumulative = [0]
    for c in self.chunks:
      self.cumulative.append(self.cumulative[-1] + c["chunk_size"])
    self._length = self.cumulative[-1]
    self.column_names = ["input_ids"]

  def __len__(self) -> int:
    return self._length

  def _chunk_and_row(self, idx: int) -> tuple[int, int]:
    """Return (chunk_index, row_within_chunk)."""
    if idx < 0 or idx >= self._length:
      raise IndexError(idx)
    chunk_idx = np.searchsorted(self.cumulative, idx, side="right") - 1
    local_row = idx - self.cumulative[chunk_idx]
    return chunk_idx, local_row

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    chunk_idx, local_row = self._chunk_and_row(idx)
    c = self.chunks[chunk_idx]
    path = self.dir_path / c["filename"]
    n_rows = c["chunk_size"]
    n_cols = (c["chunk_bytes"] // 2) // n_rows
    arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))
    row = np.array(arr[local_row], dtype=np.int64)
    # Normalize to target_seq_len if specified
    if self.target_seq_len is not None:
      row = _normalize_seq_len(row, self.target_seq_len, self.pad_token_id)
    # Clamp to vocab so embedding indices are in bounds
    row = _clamp_to_vocab(row, self.vocab_size)
    return {"input_ids": torch.from_numpy(row)}
