"""A script to count number of parameters and FLOPs in the current transformer."""

from typing import Literal

multiple_of = 256


def attn_params(d: int) -> int:
  """Attention parameters: W_qkv (3*d*d) + W_out (d*d) = 4*d^2"""
  return 4 * d**2


def glu_params(d: int, expand: float = 8/3) -> int:
  """GLU parameters: fc1 (d * 2*hidden) + fc2 (hidden * d)"""
  hidden_dim = int(d * expand)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return 3 * d * hidden_dim  # 2*d*h for fc1, d*h for fc2


def mlp_params(d: int, expand: float = 4.0) -> int:
  """MLP parameters: fc1 (d * hidden) + fc2 (hidden * d)"""
  hidden_dim = int(d * expand)
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return 2 * d * hidden_dim


def rmsnorm_params(d: int) -> int:
  """RMSNorm parameters: scale weights only"""
  return d


def block_params(d: int, mlp_class: Literal['mlp', 'glu'] = 'glu', expand: float = 8/3) -> int:
  """Total parameters in one transformer block."""
  mlp_fn = glu_params if mlp_class == 'glu' else mlp_params
  return attn_params(d) + mlp_fn(d, expand) + 2 * rmsnorm_params(d)


def tot_params(
    n_layers: int,
    d: int,
    vocab_size: int,
    weight_tying: bool = True,
    mlp_class: Literal['mlp', 'glu'] = 'glu',
    expand: float = 8/3,
) -> int:
  """
  Total parameters in the transformer.
  
  Args:
    n_layers: Number of transformer layers.
    d: Model dimension.
    vocab_size: Vocabulary size.
    weight_tying: Whether input/output embeddings are tied.
    mlp_class: 'mlp' or 'glu'.
    expand: MLP expansion factor.
    
  Returns:
    Total number of parameters.
  """
  nparams = n_layers * block_params(d, mlp_class, expand)
  nparams += d * vocab_size  # input embeddings
  nparams += d  # final RMSNorm
  if not weight_tying:
    nparams += d * vocab_size  # output projection (lm_head)
  return nparams


def non_embedding_params(
    n_layers: int,
    d: int,
    vocab_size: int,
    weight_tying: bool = True,
    mlp_class: Literal['mlp', 'glu'] = 'glu',
    expand: float = 8/3,
) -> int:
  """Non-embedding parameters (used for FLOPs estimation)."""
  nparams = n_layers * block_params(d, mlp_class, expand)
  nparams += d  # final RMSNorm
  return nparams



def get_hidden_dim(d: int, expand: float, multiple_of: int = 256) -> int:
  """Compute the actual hidden dimension after rounding."""
  hidden_dim = int(d * expand)
  return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def flops_per_token_forward(
    n_layers: int,
    d: int,
    vocab_size: int,
    seq_len: int,
    mlp_class: Literal['mlp', 'glu'] = 'glu',
    expand: float = 8/3,
    include_attn_computation: bool = True,
) -> int:
  """
  Compute forward-pass FLOPs per token for a transformer.
  
  This gives a detailed count based on actual operations:
  - Matrix multiplications: 2 * M * N * K FLOPs for (M,K) @ (K,N)
  - Attention computation: O(T * d) per token (amortized from O(T² * d) total)
  
  Args:
    n_layers: Number of transformer layers.
    d: Model dimension.
    vocab_size: Vocabulary size.
    seq_len: Sequence length (needed for attention computation).
    mlp_class: 'mlp' or 'glu'.
    expand: MLP expansion factor.
    include_attn_computation: Whether to include attention Q@K, softmax@V.
    
  Returns:
    FLOPs per token for a forward pass.
  """
  hidden_dim = get_hidden_dim(d, expand)

  layer_flops = 0

  # 1. Attention projections: QKV = 3 * (2*d*d), out = 2*d*d
  #    Total: 8 * d^2
  layer_flops += 8 * d * d

  # 2. Attention computation
  #    Q @ K^T: 2 * seq_len * d (per token, amortized)
  #    softmax @ V: 2 * seq_len * d (per token, amortized)
  #    Total: 4 * seq_len * d per token
  if include_attn_computation:
    layer_flops += 4 * seq_len * d

  # 3. MLP/GLU
  if mlp_class == 'glu':
    # fc1: d -> 2*hidden (2 * d * 2 * hidden = 4 * d * hidden)
    # fc2: hidden -> d (2 * hidden * d = 2 * d * hidden)
    # Total: 6 * d * hidden
    layer_flops += 6 * d * hidden_dim 
  else:
    # fc1: d -> hidden (2 * d * hidden)
    # fc2: hidden -> d (2 * hidden * d)
    # Total: 4 * d * hidden
    layer_flops += 4 * d * hidden_dim

  # Total for all layers
  total_flops = n_layers * layer_flops

  # Output projection: lm_head (d -> vocab_size)
  # 2 * d * vocab_size
  total_flops += 2 * d * vocab_size

  return total_flops


def flops_per_token_training(
    n_layers: int,
    d: int,
    vocab_size: int,
    seq_len: int,
    mlp_class: Literal['mlp', 'glu'] = 'glu',
    expand: float = 8/3,
    include_attn_computation: bool = True,
) -> int:
  """
  Compute training FLOPs per token (forward + backward).
  
  Backward pass is approximately 2x forward:
  - Gradient w.r.t. activations (same as forward)
  - Gradient w.r.t. weights (same as forward)
  
  Total: ~3x forward FLOPs.
  
  Args:
    Same as flops_per_token_forward.
    
  Returns:
    FLOPs per token for training (forward + backward).
  """
  forward_flops = flops_per_token_forward(
      n_layers, d, vocab_size, seq_len, mlp_class, expand, include_attn_computation
  )
  return 3 * forward_flops


def flops_kaplan_approximation(n_params: int, backward_ratio: float = 2.0) -> int:
  """
  Kaplan et al. (2020) approximation: C ≈ 6N per token.
  
  Assumptions:
  - Forward: 2N FLOPs (each weight used once in matmul)
  - Backward: 4N FLOPs (2x for activations, 2x for weights)
  
  Args:
    n_params: Number of parameters (ideally non-embedding).
    backward_ratio: Backward/forward ratio (default 2.0).
    
  Returns:
    FLOPs per token using the 6N approximation.
  """
  forward_flops = 2 * n_params
  backward_flops = backward_ratio * forward_flops
  return forward_flops + backward_flops  # = 6N for backward_ratio=2


def total_training_flops(
    n_layers: int,
    d: int,
    vocab_size: int,
    seq_len: int,
    n_tokens: int,
    mlp_class: Literal['mlp', 'glu'] = 'glu',
    expand: float = 8/3,
    include_attn_computation: bool = True,
) -> int:
  """
  Total FLOPs for training on n_tokens.
  
  Args:
    n_layers: Number of transformer layers.
    d: Model dimension.
    vocab_size: Vocabulary size.
    seq_len: Sequence length.
    n_tokens: Total number of training tokens.
    mlp_class: 'mlp' or 'glu'.
    expand: MLP expansion factor.
    include_attn_computation: Whether to include attention computation.
    
  Returns:
    Total training FLOPs.
  """
  flops_per_token = flops_per_token_training(
      n_layers, d, vocab_size, seq_len, mlp_class, expand, include_attn_computation
  )
  return flops_per_token * n_tokens


def flops_from_config(cfg, n_tokens: int = None, method: Literal['detailed', 'kaplan'] = 'detailed') -> dict:
  """
  Compute FLOPs from a config object.
  
  Args:
    cfg: Config namedtuple with model parameters.
    n_tokens: Number of training tokens (optional, for total FLOPs).
    method: 'detailed' for exact calculation, 'kaplan' for 6N approximation.
    
  Returns:
    Dictionary with FLOPs information.
  """
  d = getattr(cfg, 'd_model', getattr(cfg, 'dim', None))
  n_layers = cfg.n_layers
  vocab_size = cfg.vocab_size
  seq_len = cfg.seq_len
  mlp_class = getattr(cfg, 'mlp_class', 'glu')
  expand_str = getattr(cfg, 'expand', '8/3')
  weight_tying = getattr(cfg, 'tie_embeddings', True)

  if isinstance(expand_str, str) and '/' in expand_str:
    num, denom = expand_str.split('/')
    expand = float(num) / float(denom)
  else:
    expand = float(expand_str)

  total_params = tot_params(n_layers, d, vocab_size,
                            weight_tying, mlp_class, expand)
  non_emb_params = non_embedding_params(
      n_layers, d, vocab_size, weight_tying, mlp_class, expand)

  if method == 'kaplan':
    flops_per_token = flops_kaplan_approximation(non_emb_params)
  else:
    flops_per_token = flops_per_token_training(
        n_layers, d, vocab_size, seq_len, mlp_class, expand
    )

  result = {
      'total_params': total_params,
      'non_embedding_params': non_emb_params,
      'flops_per_token_forward': flops_per_token // 3,
      'flops_per_token_training': flops_per_token,
      'method': method,
  }

  if n_tokens is not None:
    result['total_training_flops'] = flops_per_token * n_tokens
    result['n_tokens'] = n_tokens

  return result


def format_flops(flops: int) -> str:
  """Format FLOPs with appropriate suffix (T, P, E)."""
  if flops >= 1e18:
    return f"{flops / 1e18:.2f} EFLOPs"
  elif flops >= 1e15:
    return f"{flops / 1e15:.2f} PFLOPs"
  elif flops >= 1e12:
    return f"{flops / 1e12:.2f} TFLOPs"
  elif flops >= 1e9:
    return f"{flops / 1e9:.2f} GFLOPs"
  elif flops >= 1e6:
    return f"{flops / 1e6:.2f} MFLOPs"
  else:
    return f"{flops:_} FLOPs"


if __name__ == '__main__':
  d = 1536
  n_layers = 24
  vocab_size = 50280
  seq_len = 2048
  mlp_class = 'glu'
  expand = 8/3
  weight_tying = False
  n_tokens = 10_000_000_000 

  n_params = tot_params(n_layers, d, vocab_size,
                        weight_tying, mlp_class, expand)
  n_non_emb = non_embedding_params(
      n_layers, d, vocab_size, weight_tying, mlp_class, expand)

  print("="*60)
  print("Model Configuration")
  print("="*60)
  print(f"  d_model:        {d}")
  print(f"  n_layers:       {n_layers}")
  print(f"  vocab_size:     {vocab_size}")
  print(f"  seq_len:        {seq_len}")
  print(f"  mlp_class:      {mlp_class}")
  print(f"  expand:         {expand:.4f}")
  print(f"  weight_tying:   {weight_tying}")

  print("\n" + "="*60)
  print("Parameter Count")
  print("="*60)
  print(f"  Total params:         {n_params:>15_}")
  print(f"  Non-embedding params: {n_non_emb:>15_}")

  print("\n" + "="*60)
  print("FLOPs Analysis (per token)")
  print("="*60)

  flops_fwd = flops_per_token_forward(
      n_layers, d, vocab_size, seq_len, mlp_class, expand)
  flops_train = flops_per_token_training(
      n_layers, d, vocab_size, seq_len, mlp_class, expand)

  flops_kaplan = flops_kaplan_approximation(n_non_emb)

  print(
      f"  Detailed (forward):   {flops_fwd:>15_} ({format_flops(flops_fwd)})")
  print(
      f"  Detailed (training):  {flops_train:>15_} ({format_flops(flops_train)})")
  print(
      f"  Kaplan 6N (training): {flops_kaplan:>15_} ({format_flops(flops_kaplan)})")
  print(f"  Ratio (detailed/6N):  {flops_train / flops_kaplan:.3f}")

  print("\n" + "="*60)
  print(f"Total Training FLOPs ({n_tokens/1e9:.1f}B tokens)")
  print("="*60)
  total_flops = total_training_flops(
      n_layers, d, vocab_size, seq_len, n_tokens, mlp_class, expand)
  total_kaplan = flops_kaplan * n_tokens
  print(f"  Detailed:             {format_flops(total_flops)}")
  print(f"  Kaplan 6N:            {format_flops(total_kaplan)}")
