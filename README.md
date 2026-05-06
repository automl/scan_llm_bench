This pipeline is based on the [PlainLM](https://github.com/Niccolo-Ajroldi/plainLM) pipeline.

# Pretrain a Transformer on Language Modeling

A minimal yet efficient implementation of causal language modeling in PyTorch.

It features a custom torch-compilable Transformer model implementation supporting RoPE, GLU, and RMSNorm.
It supports distributed training via Distributed Data Parallel (DDP).

A dedicated script is included for downloading, tokenizing, and chunking data, making data preparation seamless.

## 🛠 Installation

We recommend running the pipeline in a dedicated Python environment. To install dependencies in an Anaconda environment from the .yml file, execute:

```bash
conda env create -f environment.yml
```

## 📚 Data

We provide a script for downloading, tokenizing, chunking and saving Hugging Face datasets: `data/datasets/prepare.py`.
You can specify any HF dataset and tokenizer. To avoid downloading the entire corpus, we support streaming, tokenizing, and chunking data on-the-fly. We provide an example for SlimPajama in `data/datasets/prepare_slim_pajama.sh`.

## ⚡️ Usage

## Scaling & Hyperparameter Search with X

This pipeline integrates [Neural-Pipeline-Search]([https://github.com/automl/neps]) for scaling & hyperparameter search. The workflow is:

1. Sample configs → 2. Train each config → 3. Validate each config

### 1. Config Setup

A NePS config is a standard training config with an extra `neps_search_space` section that defines which hyperparameters to search over and their ranges. See `config/neps_configs/hpo_example.yaml` for a full example. Each parameter defined in the search space overrides the base config value during training.

```yaml
# Standard training config fields
lr: 3.0e-3
weight_decay: 0.1
# ... other training fields ...

# NePS search space: define hyperparameters to optimize
neps_search_space:
  lr:
    type: float
    lower: 1.0e-5
    upper: 1.0e-1
    log: true
  weight_decay:
    type: float
    lower: 1.0e-3
    upper: 0.2
    log: true
```

Supported parameter types: `float`, `int`, `categorical`, `constant`. Use `log: true` for log-scale sampling.

### 2. Sample Configs

Use `neps/hpo_neps.py` to sample configurations from the search space. This creates a `configs/` directory under `--root_dir` with one sub-folder per config:

```bash
python neps/hpo_neps.py \
  --config config/neps_configs/hpo_example.yaml \
  --root_dir neps/neps_configs/MyExperiment \
  --max_evaluations 50
```

On a cluster, generate and submit SLURM scripts automatically via `cluster/slurm_manager.py` (see `--help`).

### 3. Train a Config

Train a single sampled config (e.g. `config_5`):

```bash
python train.py \
  --config neps/neps_configs/MyExperiment/configs/config_5/config.yaml \
  --save_neps_report \
  --neps_root_dir neps/neps_configs/MyExperiment
```

To loop over a range of configs sequentially, set `START_CONFIG` / `END_CONFIG` and use `cluster/single_gpu/run_neps_configs.sh`.

### 4. Validate a Config

After training, run validation to evaluate a checkpoint and write the validation loss back into the NePS report:

```bash
python validation.py \
  --config neps/neps_configs/MyExperiment/configs/config_5/config.yaml \
  --config_number 5 \
  --neps_root_dir neps/neps_configs/MyExperiment \
  --save_neps_report
```

`--config_number` identifies which config folder to update. See `cluster/single_gpu/run_validation.sh` for the SLURM equivalent.

### SLURM Script Generation

`cluster/slurm_manager.py` automates generating and submitting SLURM scripts for all stages (sample, train array, validate array, multi-GPU, etc.). Run with `--help` for all options.

## 📂 Structure

```
scan_llm_bench/
├── cluster/             # HPC scripts
├── config/              # Configuration files for training and model setup
├── data/                # Everything regarding data preparation and data stream
│   └── datasets/        # Data preprocessing files to download, tokenize, chunk and save data
│   └── dataloaders.py   # Dataloader utilities
│   └── datasamplers.py  # Custom stateful distributed samplers
├── engine/              # Core implementation of the model engine: a torch.nn.Module implementing training steps and evaluations
├── models/              # Model architectures
├── optim/               # Optimization utilities
├── scripts/             # Helpful scripts
├── llm_bench/           # Surrogate benchmark (Can be ignored in this repo)
├── checkpoint_utils.py  # Checkpoint utilities
├── torch_utils.py       # PyTorch utilities (DDP, seed, TF32...)
├── train.py             # Main training script
├── validation.py        # Main evaluation script
└── utils.py             # Miscellaneous helper functions
```

## Credits

This project was inspired and is an adaptation of the work from: [PlainLM](https://github.com/Niccolo-Ajroldi/plainLM)

## Acknowledgements

This project is based on PlainLM by Niccolò Ajroldi.
Original repository: https://github.com/Niccolo-Ajroldi/plainLM

Modifications made:
- Split training and validation
- Modified specific training methodology (Neps Sampling, Search Space, Resource Preparing, Batch Size allocation, Neps Reports)
- Slurm Manager for job clustering
- Surrogate Benchmark for LLM
