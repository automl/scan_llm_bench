"""Export a .pth checkpoint to a Hugging Face model directory.

The exported directory can then be loaded directly by lm-evaluation-harness via:

    lm_eval --model hf \\
        --model_args pretrained=<output_dir>,trust_remote_code=True \\
        --tasks hellaswag,arc_easy,piqa,boolq \\
        --device cuda:0 --batch_size auto

Usage examples:

# Single checkpoint export
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

# Bulk export all checkpoints under an experiment directory
python scripts/export_to_hf.py \
    --exp_dir /work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/checkpoints/NePS_Full_Search_Space_100_Configs \
    --output_root /work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/hf_checkpoints

# Bulk export all checkpoints under a single job directory
python scripts/export_to_hf.py \
    --job_dir /work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/checkpoints/NePS_Full_Search_Space_100_Configs/job_27404939 \
    --output_root /work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/hf_checkpoints
"""

import argparse
import os
import re
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


def _extract_step_name(ckpt_path: Path) -> str:
    match = re.search(r"step_(\d+)", ckpt_path.stem)
    if match:
        return match.group(1)
    return ckpt_path.stem


def _iter_config_checkpoints(root_dir: Path):
    config_pattern = re.compile(r"^config_\d+$")

    for config_dir in sorted(
        [p for p in root_dir.rglob("config_*") if p.is_dir() and config_pattern.match(p.name)],
        key=lambda p: str(p),
    ):
        ckpts = sorted(config_dir.glob("*.pth"), key=lambda p: p.name)
        for ckpt_path in ckpts:
            yield config_dir, ckpt_path


def _export_single_checkpoint(
    ckpt_path: Path,
    output_dir: Path,
    config_path: Path,
    tokenizer: str,
    disable_safe_serialization: bool,
    hf_hub_cache: str,
    tokenizer_local_files_only: bool,
):
    train_cfg = _load_training_config(str(config_path))
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

    hf_model = TransformerHuggingFace(hf_config)
    print(
        f"Instantiated TransformerHuggingFace "
        f"({sum(p.numel() for p in hf_model.parameters()):,} params)"
    )

    print(f"Loading checkpoint {ckpt_path} ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    raw_sd = ckpt.get("state_dict", ckpt)
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

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(
        str(output_dir),
        safe_serialization=not disable_safe_serialization,
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
        dst = output_dir / src_name
        shutil.copy2(str(src), str(dst))

    print(f"Saving tokenizer ({tokenizer}) ...")
    tokenizer_obj = AutoTokenizer.from_pretrained(
        tokenizer,
        cache_dir=hf_hub_cache,
        local_files_only=tokenizer_local_files_only,
    )
    tokenizer_obj.save_pretrained(str(output_dir))
    print(f"Tokenizer saved to {output_dir}")

    import json

    config_json_path = output_dir / "config.json"
    with open(config_json_path, "r") as f:
        cfg_json = json.load(f)

    cfg_json["auto_map"] = {
        "AutoConfig": "configuration_transformer.TransformerHuggingFaceConfig",
        "AutoModelForCausalLM": "modeling_transformer.TransformerHuggingFace",
    }
    with open(config_json_path, "w") as f:
        json.dump(cfg_json, f, indent=2)
    print("Wrote auto_map to config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Export a .pth checkpoint to a Hugging Face model directory."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the .pth checkpoint file (e.g. ckpt_step_1574.pth).",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help=(
            "Experiment root directory (contains job_*/config_*). "
            "All .pth files in all config_* directories are exported."
        ),
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default=None,
        help=(
            "Single job directory (contains config_*). "
            "All .pth files in all config_* directories are exported."
        ),
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
        default=None,
        help="Directory to write the exported HF model to.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="test_hf_checkpoints",
        help=(
            "Root directory for bulk export mode (--exp_dir/--job_dir). "
            "Each checkpoint is exported under <output_root>/config_<n>_step_<m>."
        ),
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

    selected_modes = sum(bool(x) for x in [args.ckpt, args.exp_dir, args.job_dir])
    if selected_modes != 1:
        sys.exit("ERROR: Provide exactly one of --ckpt, --exp_dir, or --job_dir")
    if args.ckpt and not args.output_dir:
        sys.exit("ERROR: --output_dir is required when using --ckpt")

    hf_home = args.hf_cache_dir
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_assets_cache = os.path.join(hf_home, "assets")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_assets_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

    AutoConfig.register("transformer_hugging_face", TransformerHuggingFaceConfig)
    AutoModelForCausalLM.register(TransformerHuggingFaceConfig, TransformerHuggingFace)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        config_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
        if not config_path.is_file():
            sys.exit(f"ERROR: Config not found at {config_path}")

        _export_single_checkpoint(
            ckpt_path=ckpt_path,
            output_dir=Path(args.output_dir),
            config_path=config_path,
            tokenizer=args.tokenizer,
            disable_safe_serialization=args.disable_safe_serialization,
            hf_hub_cache=hf_hub_cache,
            tokenizer_local_files_only=args.tokenizer_local_files_only,
        )
        print("\n=== Export complete! ===")
        print("To evaluate with lm-evaluation-harness:")
        print("  lm_eval --model hf \\")
        print(f"      --model_args pretrained={args.output_dir},trust_remote_code=True \\")
        print("      --tasks hellaswag,arc_easy,piqa,boolq \\")
        print("      --device cuda:0 --batch_size auto")
        return

    root_to_scan = Path(args.exp_dir or args.job_dir)
    if not root_to_scan.is_dir():
        sys.exit(f"ERROR: Directory not found: {root_to_scan}")

    output_root = Path(args.output_dir) if args.output_dir else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for config_dir, ckpt_path in _iter_config_checkpoints(root_to_scan):
        config_path = Path(args.config) if args.config else config_dir / "config.yaml"
        if not config_path.is_file():
            print(f"WARNING: Skipping {ckpt_path} because config is missing: {config_path}")
            skipped_count += 1
            continue

        step_name = _extract_step_name(ckpt_path)
        export_dir = output_root / f"{config_dir.name}_step_{step_name}"

        print("\n============================================================")
        print(f"Exporting checkpoint: {ckpt_path}")
        print(f"Config: {config_path}")
        print(f"Output: {export_dir}")

        _export_single_checkpoint(
            ckpt_path=ckpt_path,
            output_dir=export_dir,
            config_path=config_path,
            tokenizer=args.tokenizer,
            disable_safe_serialization=args.disable_safe_serialization,
            hf_hub_cache=hf_hub_cache,
            tokenizer_local_files_only=args.tokenizer_local_files_only,
        )
        exported_count += 1

    if exported_count == 0:
        print(f"\nNo checkpoints exported from {root_to_scan}")
    else:
        print(f"\n=== Bulk export complete ===")
        print(f"Exported checkpoints: {exported_count}")
        print(f"Skipped checkpoints: {skipped_count}")
        print(f"HF export root: {output_root}")


if __name__ == "__main__":
    main()
