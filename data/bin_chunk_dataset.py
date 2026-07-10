"""Datasets for SlimPajama preprocessed bin + index.json chunks.

The files we received are LitData/PyTree-style chunks, not rectangular memmaps:
  uint32 item_count
  uint32 row_offset_0
  uint32 row_offset_1
  ...
  row payloads

Each row payload starts with a uint32 byte-length prefix, followed by uint32 token
IDs. Older/simple chunks that really are rectangular uint16 matrices are still
handled as a fallback.
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
  if not (p / "index.json").exists():
    return False
  return any(p.glob("*.bin"))


def load_bin_chunk_index(dir_path: str) -> list[dict]:
  """Load index.json and return its chunk list."""
  with open(Path(dir_path) / "index.json") as f:
    data = json.load(f)
  chunks = data["chunks"]
  if not chunks:
    raise ValueError(f"No chunks in index at {dir_path}")
  return chunks


def _normalize_seq_len(row_data: np.ndarray, target_seq_len: int, pad_token_id: int) -> tuple[np.ndarray, np.ndarray]:
  """Truncate or pad row_data to target_seq_len and return a real-token mask."""
  current_len = len(row_data)
  valid_len = min(current_len, target_seq_len)
  loss_mask = np.zeros(target_seq_len, dtype=np.float32)
  loss_mask[:valid_len] = 1.0

  if current_len == target_seq_len:
    return row_data, loss_mask
  if current_len > target_seq_len:
    return row_data[:target_seq_len], loss_mask

  padded = np.full(target_seq_len, pad_token_id, dtype=row_data.dtype)
  padded[:current_len] = row_data
  return padded, loss_mask


def _format_sample(row_data: np.ndarray, target_seq_len: Optional[int], pad_token_id: int) -> dict[str, torch.Tensor]:
  """Create one sample with token IDs plus a mask for non-padding positions."""
  if target_seq_len is not None:
    row_data, loss_mask = _normalize_seq_len(row_data, target_seq_len, pad_token_id)
  else:
    loss_mask = np.ones(len(row_data), dtype=np.float32)
  return {
    "input_ids": torch.from_numpy(np.asarray(row_data, dtype=np.int64)),
    "loss_mask": torch.from_numpy(loss_mask),
  }


def _clamp_to_vocab(row_data: np.ndarray, vocab_size: Optional[int]) -> np.ndarray:
  """Clamp token IDs to [0, vocab_size - 1]."""
  if vocab_size is None:
    return row_data
  return np.clip(row_data, 0, vocab_size - 1)


def _is_rectangular_uint16_chunk(chunk: dict) -> bool:
  """True if a chunk can be read as uint16[chunk_size, n_cols]."""
  n_rows = int(chunk["chunk_size"])
  chunk_bytes = int(chunk["chunk_bytes"])
  return n_rows > 0 and chunk_bytes % (2 * n_rows) == 0


def _load_litdata_chunk(path: Path, chunk: dict) -> tuple[np.memmap, np.ndarray]:
  """Return raw byte memmap and absolute row byte offsets."""
  n_items = int(chunk["chunk_size"])
  index_chunk_bytes = int(chunk["chunk_bytes"])
  raw = np.memmap(path, dtype=np.uint8, mode="r")
  file_bytes = len(raw)

  if file_bytes < 8:
    raise ValueError(f"Unexpectedly small chunk file for {path}: file={file_bytes}")

  first_values = np.frombuffer(raw[:8], dtype="<u4")
  item_count = int(first_values[0])
  first_payload_offset = int(first_values[1])

  if item_count != n_items:
    raise ValueError(f"Unexpected item count in {path}: index={n_items}, file={item_count}")
  if first_payload_offset <= 8 or first_payload_offset > file_bytes or first_payload_offset % 4 != 0:
    raise ValueError(f"Invalid first payload offset in {path}: {first_payload_offset}")
  if index_chunk_bytes not in (file_bytes, file_bytes - first_payload_offset):
    raise ValueError(
      f"Unexpected chunk size for {path}: index={index_chunk_bytes}, "
      f"file={file_bytes}, payload={file_bytes - first_payload_offset}"
    )

  header_values = np.frombuffer(raw[:first_payload_offset], dtype="<u4")
  offsets = header_values[1:]
  if len(offsets) < n_items:
    raise ValueError(f"Not enough offsets in {path}: got {len(offsets)}, need {n_items}")

  offsets = offsets[: n_items + 1].astype(np.int64, copy=False)
  if len(offsets) == n_items:
    offsets = np.concatenate([offsets, np.array([file_bytes], dtype=np.int64)])

  starts = offsets[:-1]
  ends = offsets[1:]
  if np.any(starts < first_payload_offset) or np.any(ends > file_bytes) or np.any(ends <= starts):
    raise ValueError(f"Invalid row offsets in {path}")

  return raw, offsets


def _read_litdata_row(raw: np.memmap, offsets: np.ndarray, row_idx: int) -> np.ndarray:
  """Read one row from a LitData/PyTree chunk as int64 token IDs."""
  start = int(offsets[row_idx])
  end = int(offsets[row_idx + 1])
  row_bytes = end - start
  if row_bytes < 4 or row_bytes % 4 != 0:
    raise ValueError(f"Invalid row byte span: start={start}, end={end}")

  row_u32 = np.frombuffer(raw[start:end], dtype="<u4")
  payload_nbytes = int(row_u32[0])
  token_start = 1 if payload_nbytes == row_bytes - 4 else 0
  return row_u32[token_start:].astype(np.int64, copy=False)


class BinChunkDataset(IterableDataset):
  """Memory-efficient iterable dataset for bin+index.json chunks."""

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
    self._total_rows = sum(c["chunk_size"] for c in self.chunks)
    self.column_names = ["input_ids", "loss_mask"]

  def __len__(self) -> int:
    """Approximate length (total rows / world_size)."""
    return self._total_rows // self.world_size

  def _get_chunk_indices_for_worker(self) -> list[int]:
    all_chunk_indices = list(range(len(self.chunks)))

    if self.shuffle:
      rng = np.random.default_rng(self.seed)
      rng.shuffle(all_chunk_indices)

    rank_chunks = all_chunk_indices[self.rank::self.world_size]

    worker_info = get_worker_info()
    if worker_info is not None:
      rank_chunks = rank_chunks[worker_info.id::worker_info.num_workers]

    return rank_chunks

  def _iter_chunk(self, chunk_idx: int, rng: np.random.Generator) -> Iterator[dict]:
    c = self.chunks[chunk_idx]
    path = self.dir_path / c["filename"]
    n_rows = c["chunk_size"]

    row_indices = np.arange(n_rows)
    if self.shuffle:
      rng.shuffle(row_indices)

    if _is_rectangular_uint16_chunk(c):
      n_cols = (c["chunk_bytes"] // 2) // n_rows
      arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))
      for local_row in row_indices:
        row_data = np.array(arr[local_row], dtype=np.int64)
        row_data = _clamp_to_vocab(row_data, self.vocab_size)
        yield _format_sample(row_data, self.target_seq_len, self.pad_token_id)
      del arr
      return

    raw, offsets = _load_litdata_chunk(path, c)
    for local_row in row_indices:
      row_data = _read_litdata_row(raw, offsets, int(local_row))
      row_data = _clamp_to_vocab(row_data, self.vocab_size)
      yield _format_sample(row_data, self.target_seq_len, self.pad_token_id)
    del raw

  def __iter__(self) -> Iterator[dict]:
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
  """Map-style dataset for small validation/test bin+index.json chunks."""

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
    self.cumulative = [0]
    for c in self.chunks:
      self.cumulative.append(self.cumulative[-1] + c["chunk_size"])
    self._length = self.cumulative[-1]
    self.column_names = ["input_ids", "loss_mask"]
    self._chunk_cache: dict[int, tuple[np.memmap, np.ndarray]] = {}

  def __len__(self) -> int:
    return self._length

  def _chunk_and_row(self, idx: int) -> tuple[int, int]:
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

    if _is_rectangular_uint16_chunk(c):
      n_cols = (c["chunk_bytes"] // 2) // n_rows
      arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(n_rows, n_cols))
      row = np.array(arr[local_row], dtype=np.int64)
      del arr
    else:
      if chunk_idx not in self._chunk_cache:
        self._chunk_cache[chunk_idx] = _load_litdata_chunk(path, c)
      raw, offsets = self._chunk_cache[chunk_idx]
      row = _read_litdata_row(raw, offsets, local_row)

    row = _clamp_to_vocab(row, self.vocab_size)
    return _format_sample(row, self.target_seq_len, self.pad_token_id)
