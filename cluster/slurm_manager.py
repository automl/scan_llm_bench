"""
Slurm script generator for scan_llm_bench.

Generates Slurm batch scripts for every stage of the HPO pipeline:
  1. Sample NePS configs         (hpo_neps.py)
  2. Train sequentially          (train.py in a loop)
  3. Train single config         (train.py)
  4. Train as SLURM array job    (train.py, one task per config)
  5. Validate sequentially       (validation.py in a loop)
  6. Validate single config      (validation.py)
  7. Validate as SLURM array job (validation.py, one task per config)
  8. Probe batch size             (scripts/probe_batch_size.py)
  9. Plot                         (plot_utils.py)

Supports single-GPU (python) and multi-GPU (torchrun) execution.

Usage examples at the bottom of this file and via:
    python cluster/slurm_manager.py --help
"""

import textwrap
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class ClusterConfig:
    partition: str
    account: Optional[str] = None
    load_env: str = ""
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 48
    time_hours: int = 24


CLUSTER_CONFIGS: dict[str, ClusterConfig] = {
    "mldlc2": ClusterConfig(
        partition="mldlc2_gpu-l40s",
        load_env=(
            "source ~/anaconda3/bin/activate\n"
            "conda activate scan_llm\n"
        ),
    ),
    "meta": ClusterConfig(
        partition="gpu",
        load_env="source ~/miniconda3/bin/activate && conda activate test_scaling_laws\n",
    ),
    "juwels": ClusterConfig(
        partition="gpu",
        account="projectnucleus",
        load_env=(
            "ml --force purge\n"
            "ml Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1\n"
            "source /p/project1/projectnucleus/sinani1/envs/openclip/bin/activate\n"
        ),
    ),
}


class ModelSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"


def _dedent(text: str) -> str:
    """Strip common leading whitespace so f-strings inside methods stay clean."""
    return textwrap.dedent(text).strip() + "\n"


def _write_script(path: Path, content: str) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dedent(content), encoding="utf-8")
    return path


class SlurmManager:
    """Generate and (optionally) submit Slurm batch jobs."""

    def __init__(
        self,
        cluster: str = "mldlc2",
        workspace_dir: str | Path = ".",
        job_name: str = "scan_llm",
        gpus: int = 1,
        cpus_per_task: int | None = None,
        mem_gb: int | None = None,
        time_hours: int | None = None,
        partition: str | None = None,
        output_dir: str | Path = "cluster/generated",
    ) -> None:
        cfg = CLUSTER_CONFIGS.get(cluster)
        if cfg is None:
            raise ValueError(f"Unknown cluster {cluster!r}. Choose from {list(CLUSTER_CONFIGS)}")
        self._cluster_cfg = cfg
        self.cluster = cluster
        self.workspace_dir = Path(workspace_dir).resolve()
        self.job_name = job_name
        self.gpus = gpus
        self.cpus_per_task = cpus_per_task or cfg.cpus_per_task
        self.mem_gb = mem_gb or cfg.mem_gb
        self.time_hours = time_hours or cfg.time_hours
        self.partition = partition or cfg.partition
        # Resolve output_dir relative to workspace so it's independent of cwd
        out = Path(output_dir)
        self.output_dir = (self.workspace_dir / out if not out.is_absolute() else out).resolve()

    def _sbatch_header(
        self,
        job_name: str | None = None,
        gpus: int | None = None,
        mem_gb: int | None = None,
        time_hours: int | None = None,
        log_subdir: str = "",
        array_range: str | None = None,
    ) -> str:
        name = job_name or self.job_name
        g = gpus if gpus is not None else self.gpus
        m = mem_gb or self.mem_gb
        t = time_hours or self.time_hours
        log_dir = f"logs/{log_subdir}" if log_subdir else "logs"
        log_pattern = "%x-%A_%a" if array_range else "%x-%A"

        lines = [
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --job-name={name}",
        ]
        if g > 0:
            lines.append(f"#SBATCH --gres=gpu:{g}")
        lines += [
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={m}GB",
            f"#SBATCH --time={t:02d}:00:00",
        ]
        if self._cluster_cfg.account:
            lines.append(f"#SBATCH --account={self._cluster_cfg.account}")
        if array_range:
            lines.append(f"#SBATCH --array={array_range}")
        lines += [
            f"#SBATCH --output={log_dir}/{log_pattern}.out",
            f"#SBATCH --error={log_dir}/{log_pattern}.err",
        ]
        return "\n".join(lines)

    def _preamble(self, log_subdir: str = "") -> str:
        log_dir = f"logs/{log_subdir}" if log_subdir else "logs"
        return f"""{self._cluster_cfg.load_env}
export PYTHONPATH="${{PWD}}:${{PYTHONPATH}}"

mkdir -p {log_dir}
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME (JID $SLURM_JOB_ID) on $SLURM_JOB_PARTITION"
"""

    def _run_cmd(self, script: str, args: str, gpus: int | None = None) -> str:
        """Wrap a python command with optional torchrun for multi-GPU."""
        g = gpus if gpus is not None else self.gpus
        if g > 1:
            return (
                f"torchrun \\\n"
                f"    --standalone \\\n"
                f"    --nnodes=1 \\\n"
                f"    --nproc_per_node={g} \\\n"
                f"    {script} {args}"
            )
        return f"python -u {script} {args}"


    # Sample NePS configs (hpo_neps.py)
    def build_sample_neps_configs(
        self,
        hpo_config: str,
        neps_root_dir: str = "neps/neps_configs/Neps_Test",
        max_evaluations: int = 50,
        optimizer: str = "random",
        job_name: str | None = None,
        time_hours: int | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Generate a script to sample NePS configs via hpo_neps.py (no training)."""
        name = job_name or f"Sample_Neps_{Path(neps_root_dir).name}"
        t = time_hours or self.time_hours
        header = self._sbatch_header(job_name=name, gpus=0, mem_gb=self.mem_gb, time_hours=t,
                                     log_subdir="neps_sample")
        preamble = self._preamble(log_subdir="neps_sample")

        cmd = (
            f"python -u neps/hpo_neps.py \\\n"
            f"    --config \"{hpo_config}\" \\\n"
            f"    --root_dir \"{neps_root_dir}\" \\\n"
            f"    --max_evaluations {max_evaluations} \\\n"
            f"    --optimizer {optimizer}"
        )

        script = f"""#!/bin/bash
{header}

{preamble}
echo "Sampling NePS configs -> {neps_root_dir}"

{cmd}

echo "Finished at $(date)"
"""
        path = save_to or self.output_dir / "sample_neps_configs.sh"
        return _write_script(path, script)


    # Train configs sequentially (run_neps_configs.sh)
    def build_train_sequential(
        self,
        neps_root_dir: str,
        start_config: int = 1,
        end_config: int = 50,
        save_neps_report: bool = True,
        job_name: str | None = None,
        gpus: int | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Run each NePS config one after another in a single SLURM job."""
        name = job_name or f"Seq_Train_{Path(neps_root_dir).name}"
        g = gpus if gpus is not None else self.gpus
        header = self._sbatch_header(job_name=name, gpus=g,
                                     log_subdir="from_neps_configs")
        preamble = self._preamble(log_subdir="from_neps_configs")
        train_cmd = self._run_cmd("train.py", '"${TRAIN_ARGS[@]}"', gpus=g)
        report_flag = '1' if save_neps_report else '0'

        script = f"""#!/bin/bash
{header}

{preamble}
NEPS_CONFIGS_DIR="{neps_root_dir}"
CONFIGS_DIR="${{NEPS_CONFIGS_DIR}}/configs"
START_CONFIG="${{START_CONFIG:-{start_config}}}"
END_CONFIG="${{END_CONFIG:-{end_config}}}"
SAVE_NEPS_REPORT="${{SAVE_NEPS_REPORT:-{report_flag}}}"

echo "=============================================="
echo "NePS configs dir: $NEPS_CONFIGS_DIR"
echo "Configs dir:      $CONFIGS_DIR"
echo "Config range:     ${{START_CONFIG}} to ${{END_CONFIG}}"
echo "=============================================="

total_configs=0
for config_dir in $(find "$CONFIGS_DIR" -maxdepth 1 -type d -name "config_*" | sort -V); do
    config_num=$(basename "$config_dir" | sed 's/config_//')
    if [[ "$config_num" =~ ^[0-9]+$ ]] && [[ "$config_num" -ge "$START_CONFIG" ]] && [[ "$config_num" -le "$END_CONFIG" ]]; then
        ((total_configs++))
    fi
done
echo "Found $total_configs configurations to run"
echo ""

completed=0
failed=0
start_total=$(date +%s)

for config_dir in $(find "$CONFIGS_DIR" -maxdepth 1 -type d -name "config_*" | sort -V); do
    config_num=$(basename "$config_dir" | sed 's/config_//')
    if [[ ! "$config_num" =~ ^[0-9]+$ ]]; then continue; fi
    if [[ "$config_num" -lt "$START_CONFIG" ]] || [[ "$config_num" -gt "$END_CONFIG" ]]; then continue; fi

    config_file="${{config_dir}}/config.yaml"
    if [[ ! -f "$config_file" ]]; then
        echo "WARNING: No config.yaml in $config_dir, skipping"
        continue
    fi

    echo "=============================================="
    echo "Running config_${{config_num}} ($(($completed + $failed + 1))/$total_configs)"
    echo "Started at $(date)"
    echo "=============================================="

    start_config=$(date +%s)

    TRAIN_ARGS=(--config="$config_file" --job_cluster="$SLURM_JOB_ID" --config_number="$config_num" --neps_root_dir="$NEPS_RESULTS_DIR")
    [[ "$SAVE_NEPS_REPORT" == "1" ]] && TRAIN_ARGS+=(--save_neps_report)

    {train_cmd}

    exit_code=$?
    end_config=$(date +%s)
    runtime_config=$((end_config - start_config))

    if [[ $exit_code -eq 0 ]]; then
        echo "config_${{config_num}} completed successfully in ${{runtime_config}}s"
        ((completed++))
    else
        echo "ERROR: config_${{config_num}} failed with exit code $exit_code"
        ((failed++))
    fi
    echo ""
done

end_total=$(date +%s)
runtime_total=$((end_total - start_total))

echo "=============================================="
echo "All configurations processed!"
echo "Completed: $completed / $total_configs"
echo "Failed:    $failed"
echo "Total runtime: ${{runtime_total}}s ($(($runtime_total / 60)) minutes)"
echo "Finished at $(date)"
echo "=============================================="
"""
        path = save_to or self.output_dir / "run_neps_configs.sh"
        return _write_script(path, script)


    # Train a single config (run_train.sh)
    def build_train_single(
        self,
        config_path: str,
        config_number: int | None = None,
        neps_root_dir: str | None = None,
        save_neps_report: bool = True,
        job_name: str | None = None,
        gpus: int | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Run train.py for one specific config."""
        name = job_name or f"Train_Config_{config_number or 'single'}"
        g = gpus if gpus is not None else self.gpus
        header = self._sbatch_header(job_name=name, gpus=g,
                                     log_subdir="from_neps_configs")
        preamble = self._preamble(log_subdir="from_neps_configs")

        args = f'--config="{config_path}"'
        args += ' --job_cluster="$SLURM_JOB_ID"'
        if config_number is not None:
            args += f" --config_number={config_number}"
        if neps_root_dir:
            args += f' --neps_root_dir="{neps_root_dir}"'
        if save_neps_report:
            args += " --save_neps_report"
        train_cmd = self._run_cmd("train.py", args, gpus=g)

        script = f"""#!/bin/bash
{header}

{preamble}
start=$(date +%s)

{train_cmd}

end=$(date +%s)
runtime=$((end - start))
echo "Job execution complete."
echo "Runtime: ${{runtime}}s"
"""
        path = save_to or self.output_dir / "run_train.sh"
        return _write_script(path, script)


    # 4. Train as SLURM array job
    def build_train_array(
        self,
        neps_root_dir: str,
        array_start: int = 1,
        array_end: int = 50,
        max_concurrent: int | None = None,
        save_neps_report: bool = True,
        job_name: str | None = None,
        gpus: int | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Each SLURM array task trains one NePS config in parallel."""
        name = job_name or f"Array_Train_{Path(neps_root_dir).name}"
        g = gpus if gpus is not None else self.gpus
        array_spec = f"{array_start}-{array_end}"
        if max_concurrent:
            array_spec += f"%{max_concurrent}"
        header = self._sbatch_header(job_name=name, gpus=g,
                                     log_subdir="from_neps_configs",
                                     array_range=array_spec)
        preamble = self._preamble(log_subdir="from_neps_configs")
        report_flag = "--save_neps_report" if save_neps_report else ""

        train_cmd = self._run_cmd(
            "train.py",
            f'--config="$CONFIG_FILE" '
            f'--job_cluster="$SLURM_JOB_ID" '
            f'--config_number="${{SLURM_ARRAY_TASK_ID}}" '
            f'--neps_root_dir="{neps_root_dir}" '
            f'{report_flag}',
            gpus=g,
        )

        script = f"""#!/bin/bash
{header}

{preamble}
CONFIGS_DIR="{neps_root_dir}/configs"
CONFIG_DIR="${{CONFIGS_DIR}}/config_${{SLURM_ARRAY_TASK_ID}}"
CONFIG_FILE="${{CONFIG_DIR}}/config.yaml"

echo "Array task ${{SLURM_ARRAY_TASK_ID}}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: No config at $CONFIG_FILE"
    exit 1
fi

{train_cmd}
"""
        path = save_to or self.output_dir / "run_train_array.sh"
        return _write_script(path, script)


    def build_train_array_grouped(
        self,
        neps_root_dir: str,
        start_config: int = 1,
        end_config: int = 50,
        max_concurrent: int | None = None,
        save_neps_report: bool = True,
        job_name_prefix: str | None = None,
        default_gpus: int | None = None,
        save_to_dir: Path | None = None,
    ) -> dict[int, Path]:
        """
        Group configs by recommended_num_gpus and generate one SLURM array script
        per GPU count.

        Assumes scripts/prepare_config_resources.py has been run so that each
        config_*/config.yaml contains recommended_num_gpus.
        """
        root = Path(neps_root_dir) / "configs"
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        groups: dict[int, list[int]] = {}
        for config_dir in sorted(root.iterdir()):
            if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
                continue
            try:
                num = int(config_dir.name.replace("config_", ""))
            except ValueError:
                continue
            if num < start_config or num > end_config:
                continue
            cfg_path = config_dir / "config.yaml"
            if not cfg_path.is_file():
                continue
            try:
                with cfg_path.open("r") as f:
                    cfg_dict = yaml.safe_load(f)
                g = cfg_dict.get("recommended_num_gpus") or cfg_dict.get("num_gpus")
                if g is None:
                    g = default_gpus or self.gpus
                g = int(g)
            except Exception:
                g = default_gpus or self.gpus
            groups.setdefault(g, []).append(num)

        scripts: dict[int, Path] = {}
        out_dir = save_to_dir or self.output_dir
        out_dir = out_dir.resolve()

        for g, config_nums in groups.items():
            if not config_nums:
                continue
            config_nums = sorted(config_nums)
            array_body = ",".join(str(n) for n in config_nums)
            if max_concurrent:
                array_spec = f"{array_body}%{max_concurrent}"
            else:
                array_spec = array_body

            name = job_name_prefix or f"Array_Train_{Path(neps_root_dir).name}_{g}g"
            header = self._sbatch_header(
                job_name=name,
                gpus=g,
                log_subdir="from_neps_configs",
                array_range=array_spec,
            )
            preamble = self._preamble(log_subdir="from_neps_configs")
            report_flag = "--save_neps_report" if save_neps_report else ""

            train_cmd = self._run_cmd(
                "train.py",
                f'--config="$CONFIG_FILE" '
                f'--job_cluster="$SLURM_JOB_ID" '
                f'--config_number="${{SLURM_ARRAY_TASK_ID}}" '
                f'--neps_root_dir="{neps_root_dir}" '
                f'{report_flag}',
                gpus=g,
            )

            script = f"""#!/bin/bash
{header}

{preamble}
CONFIGS_DIR="{neps_root_dir}/configs"
CONFIG_DIR="${{CONFIGS_DIR}}/config_${{SLURM_ARRAY_TASK_ID}}"
CONFIG_FILE="${{CONFIG_DIR}}/config.yaml"

echo "Grouped array task ${{SLURM_ARRAY_TASK_ID}} on {g} GPU(s)"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: No config at $CONFIG_FILE"
    exit 1
fi

{train_cmd}
"""
            path = out_dir / f"run_train_array_{g}g.sh"
            scripts[g] = _write_script(path, script)

        return scripts


    # 5. Validate configs sequentially
    def build_validation_sequential(
        self,
        neps_root_dir: str,
        start_config: int = 1,
        end_config: int = 50,
        save_neps_report: bool = True,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Run validation.py for each config sequentially."""
        name = job_name or f"Seq_Valid_{Path(neps_root_dir).name}"
        header = self._sbatch_header(job_name=name, gpus=1,
                                     log_subdir="validation")
        preamble = self._preamble(log_subdir="validation")
        report_flag = '1' if save_neps_report else '0'

        script = f"""#!/bin/bash
{header}

{preamble}
NEPS_RESULTS_DIR="{neps_root_dir}"
CONFIGS_DIR="${{NEPS_RESULTS_DIR}}/configs"
START_CONFIG="${{START_CONFIG:-{start_config}}}"
END_CONFIG="${{END_CONFIG:-{end_config}}}"
SAVE_NEPS_REPORT="${{SAVE_NEPS_REPORT:-{report_flag}}}"

echo "=============================================="
echo "Validation sweep"
echo "NePS results dir: $NEPS_RESULTS_DIR"
echo "Config range:     ${{START_CONFIG}} to ${{END_CONFIG}}"
echo "=============================================="

completed=0
failed=0
start_total=$(date +%s)

for config_dir in $(find "$CONFIGS_DIR" -maxdepth 1 -type d -name "config_*" | sort -V); do
    config_num=$(basename "$config_dir" | sed 's/config_//')
    if [[ ! "$config_num" =~ ^[0-9]+$ ]]; then continue; fi
    if [[ "$config_num" -lt "$START_CONFIG" ]] || [[ "$config_num" -gt "$END_CONFIG" ]]; then continue; fi

    config_file="${{config_dir}}/config.yaml"
    if [[ ! -f "$config_file" ]]; then
        echo "WARNING: No config.yaml in $config_dir, skipping"
        continue
    fi

    echo "=============================================="
    echo "Validating config_${{config_num}} at $(date)"
    echo "=============================================="

    python -u validation.py \
        --config="$config_file" \
        --config_number="$config_num" \
        --job_cluster="$SLURM_JOB_ID" \
        --neps_root_dir="$NEPS_RESULTS_DIR"

    if [[ $? -eq 0 ]]; then
        echo "config_${{config_num}} validation OK"
        ((completed++))
    else
        echo "ERROR: config_${{config_num}} validation failed"
        ((failed++))
    fi
    echo ""
done

end_total=$(date +%s)
runtime_total=$((end_total - start_total))

echo "=============================================="
echo "Validation complete!"
echo "Completed: $completed"
echo "Failed:    $failed"
echo "Total runtime: ${{runtime_total}}s ($(($runtime_total / 60)) minutes)"
echo "Finished at $(date)"
echo "=============================================="
"""
        path = save_to or self.output_dir / "run_validation.sh"
        return _write_script(path, script)


    # 6. Validate a single config
    def build_validation_single(
        self,
        config_path: str,
        config_number: int,
        neps_root_dir: str,
        save_neps_report: bool = True,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Run validation.py for one specific config."""
        name = job_name or f"Valid_Config_{config_number}"
        header = self._sbatch_header(job_name=name, gpus=1,
                                     log_subdir="validation")
        preamble = self._preamble(log_subdir="validation")
        report_flag = "--save_neps_report" if save_neps_report else ""

        script = f"""#!/bin/bash
{header}

{preamble}
start=$(date +%s)

python -u validation.py \
    --config="{config_path}" \
    --config_number={config_number} \
    --job_cluster="$SLURM_JOB_ID" \
    --neps_root_dir="{neps_root_dir}" \
    {report_flag}

end=$(date +%s)
runtime=$((end - start))
echo "Validation complete. Runtime: ${{runtime}}s"
"""
        path = save_to or self.output_dir / "run_validation_single.sh"
        return _write_script(path, script)


    # 7. Validate as SLURM array job
    def build_validation_array(
        self,
        neps_root_dir: str,
        array_start: int = 1,
        array_end: int = 50,
        max_concurrent: int | None = None,
        save_neps_report: bool = True,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """Each SLURM array task validates one NePS config."""
        name = job_name or f"Array_Valid_{Path(neps_root_dir).name}"
        array_spec = f"{array_start}-{array_end}"
        if max_concurrent:
            array_spec += f"%{max_concurrent}"
        header = self._sbatch_header(job_name=name, gpus=1,
                                     log_subdir="validation",
                                     array_range=array_spec)
        preamble = self._preamble(log_subdir="validation")
        report_flag = "--save_neps_report" if save_neps_report else ""

        script = f"""#!/bin/bash
{header}

{preamble}
CONFIGS_DIR="{neps_root_dir}/configs"
CONFIG_DIR="${{CONFIGS_DIR}}/config_${{SLURM_ARRAY_TASK_ID}}"
CONFIG_FILE="${{CONFIG_DIR}}/config.yaml"

echo "Validation array task ${{SLURM_ARRAY_TASK_ID}}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: No config at $CONFIG_FILE"
    exit 1
fi

python -u validation.py \
    --config="$CONFIG_FILE" \
    --config_number="${{SLURM_ARRAY_TASK_ID}}" \
    --job_cluster="$SLURM_JOB_ID" \
    --neps_root_dir="{neps_root_dir}" \
    {report_flag}
"""
        path = save_to or self.output_dir / "run_validation_array.sh"
        return _write_script(path, script)


    # 8. Probe batch size
    def build_probe_batch_size(
        self,
        config_path: str,
        model_size: ModelSize | None = None,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        name = job_name or "probe_batch_size"
        header = self._sbatch_header(job_name=name, gpus=1, mem_gb=self.mem_gb,
                                     time_hours=1)
        preamble = self._preamble()
        size_arg = f" --model_size={model_size.value}" if model_size else ""

        script = f"""#!/bin/bash
{header}

{preamble}
python -u scripts/probe_batch_size.py --config="{config_path}"{size_arg}
"""
        path = save_to or self.output_dir / "probe_batch_size.sh"
        return _write_script(path, script)


    def build_estimate_checkpoint_disk(
        self,
        neps_root_dir_or_config: str,
        max_tokens_budget: float | None = None,
        default_global_batch: int = 256,
        bytes_per_param: float | None = None,
        size_per_billion_params_gb: float | None = None,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """
        Generate a script to run scripts/estimate_checkpoint_disk.py for either a
        NePS configs directory or a single config.yaml file.
        """
        name = job_name or f"Estimate_Checkpoint_Disk_{Path(neps_root_dir_or_config).name}"
        header = self._sbatch_header(job_name=name, gpus=0, mem_gb=self.mem_gb, time_hours=self.time_hours)
        preamble = self._preamble()

        cmd_parts = ["python -u scripts/estimate_checkpoint_disk.py"]
        p = Path(neps_root_dir_or_config)
        if p.is_file():
            cmd_parts.append(f'--config_path "{neps_root_dir_or_config}"')
        else:
            cmd_parts.append(f'--neps_configs_dir "{neps_root_dir_or_config}/configs"')

        if bytes_per_param is not None:
            cmd_parts.append(f"--bytes-per-param {float(bytes_per_param)}")
        if size_per_billion_params_gb is not None:
            cmd_parts.append(f"--size-per-billion-params-gb {float(size_per_billion_params_gb)}")

        cmd = " \\\n+    ".join(cmd_parts)

        script = f"""#!/bin/bash
{header}

{preamble}
{cmd}
"""
        path = save_to or self.output_dir / "estimate_checkpoint_disk.sh"
        return _write_script(path, script)


    # 9. Plot
    def build_plot(
        self,
        neps_configs_dir: str,
        num_gpus: int = 1,
        out_gpu_hours: str | None = None,
        out_dist: str | None = None,
        out_mfu_distribution: str | None = None,
        out_flops_vs_mfu: str | None = None,
        gpu_name: str | None = None,
        flops_per_gpu_hour: float | None = None,
        job_name: str | None = None,
        save_to: Path | None = None,
    ) -> Path:
        name = job_name or "plot"
        header = self._sbatch_header(job_name=name, gpus=0, mem_gb=self.mem_gb,
                                     time_hours=self.time_hours)
        preamble = self._preamble()

        cmd_parts = [
            "python plot_utils.py",
            f'--neps_configs_dir "{neps_configs_dir}"',
            f"--num-gpus {num_gpus}",
        ]
        if out_gpu_hours:
            cmd_parts.append(f'--out-gpu-hours "{out_gpu_hours}"')
        if out_dist:
            cmd_parts.append(f'--out-dist "{out_dist}"')
        if out_mfu_distribution:
            cmd_parts.append(f'--out-mfu-distribution "{out_mfu_distribution}"')
        if out_flops_vs_mfu:
            cmd_parts.append(f'--out-flops-vs-mfu "{out_flops_vs_mfu}"')
        if gpu_name:
            cmd_parts.append(f'--gpu-name "{gpu_name}"')
        if flops_per_gpu_hour is not None:
            cmd_parts.append(f'--flops-per-gpu-hour {float(flops_per_gpu_hour)}')
            
        cmd = " \\\n    ".join(cmd_parts)

        script = f"""#!/bin/bash
{header}

{preamble}
{cmd}
"""
        path = save_to or self.output_dir / "plot.sh"
        return _write_script(path, script)


    # 10. Prepare resources (batch size, grad_accum, GPUs) for NePS configs
    def build_prepare_resources(
        self,
        neps_root_dir: str,
        max_gpus_cap: int | None = None,
        job_name: str | None = None,
        time_hours: int | None = None,
        save_to: Path | None = None,
    ) -> Path:
        """
        Generate a script that runs scripts/prepare_config_resources.py over all
        config_*/config.yaml under neps_root_dir/configs.

        This precomputes micro_batch_size, grad_accumulation_steps, and
        recommended_num_gpus so that train.py can run without probing.
        """
        name = job_name or f"Prepare_Resources_{Path(neps_root_dir).name}"
        t = time_hours or self.time_hours
        header = self._sbatch_header(
            job_name=name,
            gpus=1,
            mem_gb=self.mem_gb,
            time_hours=t,
            log_subdir="prepare_resources",
        )
        preamble = self._preamble(log_subdir="prepare_resources")

        cmd_parts = ["python -u scripts/prepare_config_resources.py"]
        # Allow either a NePS configs directory or a single config file path.
        p = Path(neps_root_dir)
        if p.is_file():
            cmd_parts.append(f'--config_path "{neps_root_dir}"')
        else:
            cmd_parts.append(f'--neps_configs_dir "{neps_root_dir}"')
        if max_gpus_cap is not None:
            cmd_parts.append(f"--max-gpus-cap {int(max_gpus_cap)}")
        cmd = " \\\n    ".join(cmd_parts)

        script = f"""#!/bin/bash
{header}

{preamble}
{cmd}
"""
        path = save_to or self.output_dir / "prepare_resources.sh"
        return _write_script(path, script)


    # 11. Generate ALL scripts for a full pipeline
    def generate_all(
        self,
        hpo_config: str,
        neps_root_dir: str,
        max_evaluations: int = 50,
        start_config: int = 1,
        end_config: int = 50,
        optimizer: str = "random",
        max_gpus_cap: int | None = None,
    ) -> dict[str, Path]:
        """
        One-shot: generate every script needed for a full pipeline.

        Returns a dict mapping script name to path.
        """
        out = self.output_dir
        scripts: dict[str, Path] = {}

        scripts["sample_neps_configs"] = self.build_sample_neps_configs(
            hpo_config=hpo_config,
            neps_root_dir=neps_root_dir,
            max_evaluations=max_evaluations,
            optimizer=optimizer,
        )
        scripts["prepare_resources"] = self.build_prepare_resources(
            neps_root_dir=neps_root_dir,
            max_gpus_cap=max_gpus_cap,
        )
        scripts["train_sequential"] = self.build_train_sequential(
            neps_root_dir=neps_root_dir,
            start_config=start_config,
            end_config=end_config,
        )
        scripts["train_single"] = self.build_train_single(
            config_path=f"{neps_root_dir}/configs/config_{start_config}/config.yaml",
            config_number=start_config,
            neps_root_dir=neps_root_dir,
        )
        scripts["train_array"] = self.build_train_array(
            neps_root_dir=neps_root_dir,
            array_start=start_config,
            array_end=end_config,
        )
        scripts["validation_sequential"] = self.build_validation_sequential(
            neps_root_dir=neps_root_dir,
            start_config=start_config,
            end_config=end_config,
        )
        scripts["validation_single"] = self.build_validation_single(
            config_path=f"{neps_root_dir}/configs/config_{start_config}/config.yaml",
            config_number=start_config,
            neps_root_dir=neps_root_dir,
        )
        scripts["validation_array"] = self.build_validation_array(
            neps_root_dir=neps_root_dir,
            array_start=start_config,
            array_end=end_config,
        )
        return scripts

    # Submit
    def submit(self, script_path: Path, wait: bool = False) -> int | None:
        """Submit a generated script via sbatch."""
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = ["sbatch"]
        if wait:
            cmd.append("--wait")
        cmd.append(str(script_path))

        print("Submitting:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        if stdout.startswith("Submitted batch job"):
            job_id = int(stdout.split()[-1])
            print(f"Job ID: {job_id}")
            return job_id
        print("STDOUT:", stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return None



# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for the scan_llm_bench pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        EXAMPLES
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Generate ALL scripts for a full pipeline (recommended first step):
        python cluster/slurm_manager.py all \\
            --hpo_config config/neps_configs/hpo_scale_search_space.yaml \\
            --neps_root_dir neps/neps_configs/Neps_Test \\
            --max_evaluations 100 --start_config 1 --end_config 100

        # Generate just the config-sampling script:
        python cluster/slurm_manager.py sample \\
            --hpo_config config/neps_configs/hpo_scale_search_space.yaml \\
            --neps_root_dir neps/neps_configs/Neps_Test \\
            --max_evaluations 100
            
        # Generate the resource preparation script:
        python cluster/slurm_manager.py prepare_resources \\
            --neps_root_dir neps/neps_test_configs/neps_configs_samples/configs \\
            --max_gpus_cap 8

        # Generate a sequential training script:
        python cluster/slurm_manager.py train_seq \\
            --neps_root_dir /neps/neps_configs/Neps_Test \\
            --start_config 1 --end_config 50

        # Generate a single-config training script:
        python cluster/slurm_manager.py train_single \\
            --config_path /work/.../configs/config_5/config.yaml \\
            --config_number 5 \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test

        # Generate a SLURM array job for training:
        python cluster/slurm_manager.py train_array \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test \\
            --start_config 1 --end_config 50 --max_concurrent 10

        # Multi-GPU training (sequential, 4 GPUs with torchrun):
        python cluster/slurm_manager.py train_seq \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test \\
            --gpus 4

        # Generate sequential validation script:
        python cluster/slurm_manager.py valid_seq \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test \\
            --start_config 1 --end_config 50

        # Generate single validation script:
        python cluster/slurm_manager.py valid_single \\
            --config_path /work/.../configs/config_11/config.yaml \\
            --config_number 11 \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test

        # Generate validation array job:
        python cluster/slurm_manager.py valid_array \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test \\
            --start_config 1 --end_config 50

        # Generate and immediately submit:
        python cluster/slurm_manager.py train_seq \\
            --neps_root_dir /work/.../neps/neps_configs/Neps_Test \\
            --submit



        TYPICAL WORKFLOW

        # Step 1: Generate all scripts
        python cluster/slurm_manager.py all \\
            --hpo_config config/neps_configs/hpo_scale_search_space.yaml \\
            --neps_root_dir neps/neps_configs/MyExperiment \\
            --max_evaluations 50 --cluster mldlc2

        # Step 2: Sample configs & prepare resources
        sbatch cluster/generated/sample_neps_configs.sh
        sbatch cluster/generated/prepare_resources.sh

        
        # Step 3: Train (pick one approach)
        sbatch cluster/generated/run_neps_configs.sh          # sequential
        sbatch cluster/generated/run_train_array.sh           # array (parallel)
        sbatch cluster/generated/run_train.sh                 # single config

        # Step 4: Validate
        sbatch cluster/generated/run_validation.sh            # sequential
        sbatch cluster/generated/run_validation_array.sh      # array (parallel)

        # Override config range at submit time:
        START_CONFIG=10 END_CONFIG=25 sbatch cluster/generated/run_neps_configs.sh
        """),
    )

    # -- shared arguments --
    parser.add_argument("--cluster", default="mldlc2",
                        choices=list(CLUSTER_CONFIGS),
                        help="Cluster preset (default: mldlc2)")
    parser.add_argument("--workspace_dir", type=Path, default=None,
                        help="Project root; scripts written under this (default: parent of cluster/)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs per job (>1 uses torchrun)")
    parser.add_argument("--mem_gb", type=int, default=None)
    parser.add_argument("--time_hours", type=int, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--job_name", default=None)
    parser.add_argument("--output_dir", default="cluster/generated",
                        help="Where to write generated scripts (relative to workspace_dir)")
    parser.add_argument("--submit", action="store_true",
                        help="Submit the script after generating it")

    sub = parser.add_subparsers(dest="mode", required=True)

    # -- all --
    p_all = sub.add_parser("all", help="Generate ALL pipeline scripts at once")
    p_all.add_argument("--hpo_config", required=True)
    p_all.add_argument("--neps_root_dir", required=True)
    p_all.add_argument("--max_evaluations", type=int, default=50)
    p_all.add_argument("--optimizer", default="random")
    p_all.add_argument("--start_config", type=int, default=1)
    p_all.add_argument("--end_config", type=int, default=50)
    p_all.add_argument(
        "--max_gpus_cap",
        type=int,
        default=8,
        help="Upper cap for recommended_num_gpus in prepare_resources.",
    )

    # -- sample --
    p_sample = sub.add_parser("sample", help="Sample NePS configs (hpo_neps.py)")
    p_sample.add_argument("--hpo_config", required=True)
    p_sample.add_argument("--neps_root_dir", default="neps/neps_configs/Neps_Test")
    p_sample.add_argument("--max_evaluations", type=int, default=50)
    p_sample.add_argument("--optimizer", default="random")

    # -- prepare_resources --
    p_prep = sub.add_parser(
        "prepare_resources",
        help="Precompute micro_batch_size, grad_accumulation_steps, and recommended_num_gpus for NePS configs or a single config.",
    )
    group_prep = p_prep.add_mutually_exclusive_group(required=True)
    group_prep.add_argument("--neps_root_dir", help="Path to NePS root (contains 'configs' subdir)")
    group_prep.add_argument("--config_path", help="Path to a single config.yaml file")
    p_prep.add_argument(
        "--max_gpus_cap",
        type=int,
        default=8,
        help="Upper cap for recommended_num_gpus.",
    )

    # -- train_seq --
    p_tseq = sub.add_parser("train_seq", help="Sequential training over configs")
    p_tseq.add_argument("--neps_root_dir", required=True)
    p_tseq.add_argument("--start_config", type=int, default=1)
    p_tseq.add_argument("--end_config", type=int, default=50)

    # -- train_single --
    p_tsingle = sub.add_parser("train_single", help="Train a single config")
    p_tsingle.add_argument("--config_path", required=True)
    p_tsingle.add_argument("--config_number", type=int, default=None)
    p_tsingle.add_argument("--neps_root_dir", default=None)

    # -- train_array --
    p_tarray = sub.add_parser("train_array", help="SLURM array job for training")
    p_tarray.add_argument("--neps_root_dir", required=True)
    p_tarray.add_argument("--start_config", type=int, default=1)
    p_tarray.add_argument("--end_config", type=int, default=50)
    p_tarray.add_argument("--max_concurrent", type=int, default=None,
                          help="Max simultaneous array tasks (e.g. 10)")

    # -- train_array_grouped --
    p_tarray_grp = sub.add_parser(
        "train_array_grouped",
        help="Grouped SLURM array jobs for training (one array per GPU count).",
    )
    p_tarray_grp.add_argument("--neps_root_dir", required=True)
    p_tarray_grp.add_argument("--start_config", type=int, default=1)
    p_tarray_grp.add_argument("--end_config", type=int, default=50)
    p_tarray_grp.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Max simultaneous array tasks per group (e.g. 10)",
    )

    # -- valid_seq --
    p_vseq = sub.add_parser("valid_seq", help="Sequential validation over configs")
    p_vseq.add_argument("--neps_root_dir", required=True)
    p_vseq.add_argument("--start_config", type=int, default=1)
    p_vseq.add_argument("--end_config", type=int, default=50)

    # -- valid_single --
    p_vsingle = sub.add_parser("valid_single", help="Validate a single config")
    p_vsingle.add_argument("--config_path", required=True)
    p_vsingle.add_argument("--config_number", type=int, required=True)
    p_vsingle.add_argument("--neps_root_dir", required=True)

    # -- valid_array --
    p_varray = sub.add_parser("valid_array", help="SLURM array job for validation")
    p_varray.add_argument("--neps_root_dir", required=True)
    p_varray.add_argument("--start_config", type=int, default=1)
    p_varray.add_argument("--end_config", type=int, default=50)
    p_varray.add_argument("--max_concurrent", type=int, default=None)

    # -- probe --
    p_probe = sub.add_parser("probe", help="Probe max batch size")
    p_probe.add_argument("--config_path", required=True)
    p_probe.add_argument("--model_size", default=None,
                         choices=["tiny", "small", "medium"])

    # -- plot --
    p_plot = sub.add_parser("plot", help="Generate plots")
    p_plot.add_argument("--neps_configs_dir", required=True)
    p_plot.add_argument("--num_gpus", type=int, default=1)
    p_plot.add_argument("--out_gpu_hours", default=None)
    p_plot.add_argument("--out_dist", default=None)
    p_plot.add_argument("--out_mfu_distribution", default=None)
    p_plot.add_argument("--out_flops_vs_mfu", default=None)
    p_plot.add_argument("--gpu_name", default=None)
    p_plot.add_argument("--flops_per_gpu_hour", default=None)
    
    # -- estimate_disk --
    p_est = sub.add_parser(
        "estimate_disk",
        help="Estimate checkpoint disk usage for NePS configs or single config",
    )
    group_est = p_est.add_mutually_exclusive_group(required=True)
    group_est.add_argument("--neps_configs_dir", help="Path to NePS root (contains 'configs' subdir)")
    group_est.add_argument("--config_path", help="Path to a single config.yaml file")
    p_est.add_argument("--bytes_per_param", type=float, default=None)
    p_est.add_argument("--size_per_billion_params_gb", type=float, default=None)

    args = parser.parse_args()

    # Default workspace to repo root (parent of cluster/) so paths are independent of cwd
    workspace_dir = args.workspace_dir
    if workspace_dir is None:
        workspace_dir = Path(__file__).resolve().parent.parent

    mgr = SlurmManager(
        cluster=args.cluster,
        workspace_dir=workspace_dir,
        gpus=args.gpus,
        cpus_per_task=args.cpus_per_task,
        mem_gb=args.mem_gb,
        time_hours=args.time_hours,
        partition=args.partition,
        job_name=args.job_name or "scan_llm",
        output_dir=args.output_dir,
    )

    if args.mode == "all":
        scripts = mgr.generate_all(
            hpo_config=args.hpo_config,
            neps_root_dir=args.neps_root_dir,
            max_evaluations=args.max_evaluations,
            start_config=args.start_config,
            end_config=args.end_config,
            optimizer=args.optimizer,
            max_tokens_budget=args.max_tokens_budget,
            max_gpus_cap=args.max_gpus_cap,
        )
        print(f"\nGenerated {len(scripts)} scripts in {mgr.output_dir.resolve()}/:")
        for name, path in scripts.items():
            print(f"  {name:30s} -> {path}")
        if args.submit:
            print("\n--submit with 'all' mode: submitting sample_neps_configs only.")
            mgr.submit(scripts["sample_neps_configs"])
        return

    path: Path
    if args.mode == "sample":
        path = mgr.build_sample_neps_configs(
            hpo_config=args.hpo_config,
            neps_root_dir=args.neps_root_dir,
            max_evaluations=args.max_evaluations,
            optimizer=args.optimizer,
        )
    elif args.mode == "prepare_resources":
        target = args.neps_root_dir if getattr(args, "neps_root_dir", None) else args.config_path
        path = mgr.build_prepare_resources(
            neps_root_dir=target,
            max_gpus_cap=args.max_gpus_cap,
        )
    elif args.mode == "train_array_grouped":
        scripts_by_gpus = mgr.build_train_array_grouped(
            neps_root_dir=args.neps_root_dir,
            start_config=args.start_config,
            end_config=args.end_config,
            max_concurrent=args.max_concurrent,
        )
        if not scripts_by_gpus:
            print("No grouped array scripts were generated.")
            return
        print("\nGenerated grouped array scripts:")
        for g, script_path in sorted(scripts_by_gpus.items()):
            print(f"  {g} GPU(s): {script_path}")
        if args.submit:
            print("\nSubmitting grouped array jobs:")
            for g, script_path in sorted(scripts_by_gpus.items()):
                print(f"Submitting script for {g} GPU(s): {script_path}")
                mgr.submit(script_path)
        return
    elif args.mode == "train_seq":
        path = mgr.build_train_sequential(
            neps_root_dir=args.neps_root_dir,
            start_config=args.start_config,
            end_config=args.end_config,
        )
    elif args.mode == "train_single":
        # If the config has been preprocessed by scripts/prepare_config_resources.py,
        # it will contain a recommended_num_gpus field. Use that for this job,
        # falling back to the global --gpus value otherwise.
        single_gpus: int | None = None
        try:
            with open(args.config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            single_gpus = cfg_dict.get("recommended_num_gpus") or cfg_dict.get("num_gpus")
            if single_gpus is not None:
                single_gpus = int(single_gpus)
        except Exception:
            single_gpus = None

        path = mgr.build_train_single(
            config_path=args.config_path,
            config_number=args.config_number,
            neps_root_dir=args.neps_root_dir,
            gpus=single_gpus,
        )
    elif args.mode == "train_array":
        path = mgr.build_train_array(
            neps_root_dir=args.neps_root_dir,
            array_start=args.start_config,
            array_end=args.end_config,
            max_concurrent=args.max_concurrent,
        )
    elif args.mode == "valid_seq":
        path = mgr.build_validation_sequential(
            neps_root_dir=args.neps_root_dir,
            start_config=args.start_config,
            end_config=args.end_config,
        )
    elif args.mode == "valid_single":
        path = mgr.build_validation_single(
            config_path=args.config_path,
            config_number=args.config_number,
            neps_root_dir=args.neps_root_dir,
        )
    elif args.mode == "valid_array":
        path = mgr.build_validation_array(
            neps_root_dir=args.neps_root_dir,
            array_start=args.start_config,
            array_end=args.end_config,
            max_concurrent=args.max_concurrent,
        )
    elif args.mode == "probe":
        model_sz = ModelSize(args.model_size) if args.model_size else None
        path = mgr.build_probe_batch_size(
            config_path=args.config_path,
            model_size=model_sz,
        )
    elif args.mode == "estimate_disk":
        target = args.neps_configs_dir if getattr(args, "neps_configs_dir", None) else args.config_path
        path = mgr.build_estimate_checkpoint_disk(
            neps_root_dir_or_config=target,
            bytes_per_param=args.bytes_per_param,
            size_per_billion_params_gb=args.size_per_billion_params_gb,
        )
    elif args.mode == "plot":
        path = mgr.build_plot(
            neps_configs_dir=args.neps_configs_dir,
            num_gpus=args.num_gpus,
            out_gpu_hours=args.out_gpu_hours,
            out_dist=args.out_dist,
            out_mfu_distribution=args.out_mfu_distribution,
            out_flops_vs_mfu=args.out_flops_vs_mfu,
            gpu_name=args.gpu_name,
            flops_per_gpu_hour=args.flops_per_gpu_hour,
        )
    else:
        parser.error(f"Unknown mode: {args.mode}")
        return

    print(f"Wrote script -> {path}")
    if args.submit:
        mgr.submit(path)


if __name__ == "__main__":
    main()
