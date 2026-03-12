"""Export a .pth checkpoint to a Hugging Face model directory.

The exported directory can be loaded directly by lm-evaluation-harness via:

    lm_eval --model hf \\
        --model_args pretrained=<output_dir>,trust_remote_code=True \\
        --tasks hellaswag,arc_easy,piqa,boolq \\
        --device cuda:0 --batch_size auto

Usage examples:

python scripts/export_to_hf.py \\
    --ckpt /work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/checkpoints/NePS_Full_Search_Space_100_Configs/job_27404939/config_100/ckpt_step_1574.pth \\
    --output_dir ./hf_exported/config_100_step_1574

# Override tokenizer (default: EleutherAI/gpt-neox-20b)
python scripts/export_to_hf.py \\
    --ckpt /path/to/ckpt_step_1574.pth \\
    --output_dir ./hf_exported/my_model \\
    --tokenizer EleutherAI/gpt-neox-20b

# Explicit config YAML
python scripts/export_to_hf.py \\
    --ckpt /path/to/ckpt_step_1574.pth \\
    --config /path/to/config.yaml \\
    --output_dir ./hf_exported/my_model
"""

import argparse
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

from models.configuration_transformer import TransformerHuggingFaceConfig
from models.modeling_transformer import TransformerHuggingFace
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_HF_CACHE_DIR = "/work/dlclarge2/sermaxha-scan_thesis/storage_hub/hf_cache/huggingface"


def _strip_ddp_compile_prefixes(state_dict: dict) -> dict:
    """Remove ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("_orig_mod.", "")
        cleaned[k] = v
    return cleaned


def _load_training_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Export a .pth checkpoint to a Hugging Face model directory."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file (e.g. ckpt_step_1574.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the training config.yaml. "
            "If omitted, looks for config.yaml in the same directory as --ckpt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the exported HF model to.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help=(
            "HuggingFace tokenizer name or local path. "
            "Must match the tokenizer used during data preparation. "
            "Default: EleutherAI/gpt-neox-20b (same as training)."
        ),
    )
    parser.add_argument(
        "--disable_safe_serialization",
        action="store_true",
        help=(
            "Disable safetensors serialization and save PyTorch bin weights instead. "
            "Only use this if a specific Transformers version still errors on shared tied weights."
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=DEFAULT_HF_CACHE_DIR,
        help=(
            "Hugging Face cache root directory (HF_HOME). "
            "Used for tokenizer/model downloads and cache writes."
        ),
    )
    parser.add_argument(
        "--tokenizer_local_files_only",
        action="store_true",
        help=(
            "Load tokenizer from local cache/files only (no network). "
            "Useful on cluster nodes without internet access."
        ),
    )
    args = parser.parse_args()

    hf_home = args.hf_cache_dir
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_assets_cache = os.path.join(hf_home, "assets")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_assets_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(args.ckpt), "config.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: Config not found at {config_path}")

    train_cfg = _load_training_config(config_path)
    print(f"Loaded training config from {config_path}")

    expand_raw = train_cfg.get("expand", train_cfg.get("expand", 4.0))
    try:
        expand_val = float(Fraction(str(expand_raw)))
    except (ValueError, ZeroDivisionError):
        expand_val = float(expand_raw)

    hf_config = TransformerHuggingFaceConfig(
        vocab_size=train_cfg["vocab_size"],
        seq_len=train_cfg["seq_len"],
        dim=train_cfg["d_model"],
        expand=expand_val,
        n_layers=train_cfg["n_layers"],
        n_heads=train_cfg["n_heads"],
        mlp=train_cfg.get("mlp_class", "glu"),
        rmsnorm_eps=1e-6,
        tie_embeddings=train_cfg.get("tie_embeddings", False),
    )

    AutoConfig.register("transformer_hugging_face", TransformerHuggingFaceConfig)
    AutoModelForCausalLM.register(TransformerHuggingFaceConfig, TransformerHuggingFace)

    hf_model = TransformerHuggingFace(hf_config)
    print(f"Instantiated TransformerHuggingFace "
          f"({sum(p.numel() for p in hf_model.parameters()):,} params)")

    print(f"Loading checkpoint {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    raw_sd = ckpt["state_dict"]
    clean_sd = _strip_ddp_compile_prefixes(raw_sd)

    missing, unexpected = hf_model.model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  WARNING - missing keys: {missing}")
    if unexpected:
        print(f"  WARNING - unexpected keys: {unexpected}")
    if not missing and not unexpected:
        print("  All weights loaded successfully (strict match).")

    if hf_model.config.tie_word_embeddings:
        hf_model.tie_weights()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(
        output_dir,
        safe_serialization=not args.disable_safe_serialization,
    )
    print(f"Saved HF model to {output_dir}")

    models_dir = REPO_ROOT / "models"
    for src_name in [
        "configuration_transformer.py",
        "modeling_transformer.py",
        "transformer.py",
        "components.py",
        "embeddings.py",
    ]:
        src = models_dir / src_name
        dst = os.path.join(output_dir, src_name)
        shutil.copy2(str(src), dst)

    print(f"Saving tokenizer ({args.tokenizer}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=hf_hub_cache,
        local_files_only=args.tokenizer_local_files_only,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    import json

    config_json_path = os.path.join(output_dir, "config.json")
    with open(config_json_path, "r") as f:
        cfg_json = json.load(f)

    cfg_json["auto_map"] = {
        "AutoConfig": "configuration_transformer.TransformerHuggingFaceConfig",
        "AutoModelForCausalLM": "modeling_transformer.TransformerHuggingFace",
    }
    with open(config_json_path, "w") as f:
        json.dump(cfg_json, f, indent=2)
    print("Wrote auto_map to config.json")

    print(f"\n=== Export complete! ===")
    print(f"To evaluate with lm-evaluation-harness:")
    print(f"  lm_eval --model hf \\")
    print(f"      --model_args pretrained={output_dir},trust_remote_code=True \\")
    print(f"      --tasks hellaswag,arc_easy,piqa,boolq \\")
    print(f"      --device cuda:0 --batch_size auto")


if __name__ == "__main__":
    main()
