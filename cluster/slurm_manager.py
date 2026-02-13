"""
Slurm script generator for scan_llm_bench.

Generates Slurm batch scripts for training runs with NePS configs,
with support for model-size-based global batch size targets and
auto batch size tuning (probe low → high to find max micro_batch_size that fits).
"""

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


@dataclass
class ClusterConfig:
    partition: str
    account: Optional[str] = None
    load_env: str = ""


# Cluster configurations
CLUSTER_CONFIGS: dict[str, ClusterConfig] = {
    "mldlc2": ClusterConfig(
        partition="mldlc2_gpu-l40s",
        load_env="""source ~/anaconda3/bin/activate
                    conda activate scan_llm
                    """,
    ),
    "meta": ClusterConfig(
        partition="gpu",
        load_env="source ~/miniconda3/bin/activate && conda activate test_scaling_laws",
    ),
    "juwels": ClusterConfig(
        partition="gpu",
        account="projectnucleus",
        load_env="""ml --force purge
                    ml Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
                    source /p/project1/projectnucleus/sinani1/envs/openclip/bin/activate
                    """,
    ),
}


class SlurmManager:
    """Manager to create and submit Slurm batch jobs for scan_llm_bench."""

    def __init__(
        self,
        job_name: str = "scan_llm_train",
        partition: str = "mldlc2_gpu-l40s",
        gpus: int = 1,
        cpus_per_task: int = 8,
        mem_gb: int = 48,
        time_hours: int = 24,
        root_dir: Path | str = ".",
        workspace_dir: str = ".",
        cluster: str = "mldlc2",
    ) -> None:
        self.job_id: Optional[int] = None
        self.job_name = job_name
        self.partition = partition
        self.gpus = gpus
        self.cpus_per_task = cpus_per_task
        self.mem_gb = mem_gb
        self.time_hours = time_hours
        self.root_dir = Path(root_dir)
        self.workspace_dir = Path(workspace_dir)
        self.script_path = self.root_dir / "train.sh"
        self.cluster = cluster
        self._cluster_cfg = CLUSTER_CONFIGS.get(
            cluster, CLUSTER_CONFIGS["mldlc2"])

    def _account_line(self) -> str:
        if self._cluster_cfg.account:
            return f"#SBATCH --account={self._cluster_cfg.account}\n"
        return ""

    def _python_path_setup(self) -> str:
        return f"""
                export PYTHONPATH="${{PWD}}:${{PYTHONPATH}}"
                cd "{self.workspace_dir}"
                """

    def build_train_script(
        self,
        base_config: str,
        config_overrides: Optional[str] = None,
        config_number: Optional[int] = None,
        neps_report_dir: Optional[str] = None,
        save_neps_report: bool = False,
        save_to: Optional[Path] = None,
    ) -> Path:
        """Build a Slurm script for a single training run."""

        cmd_parts = [
            "python -u train.py",
            f'--config="{base_config}"',
        ]
        if config_overrides:
            cmd_parts.append(f'--config_overrides="{config_overrides}"')
        if config_number is not None:
            cmd_parts.append(f"--config_number={config_number}")
        if neps_report_dir:
            cmd_parts.append(f'--neps_report_dir="{neps_report_dir}"')
        if save_neps_report:
            cmd_parts.append("--save_neps_report")

        cmd = " \\\n    ".join(cmd_parts)

        script = f"""#!/bin/bash
                        #SBATCH --partition={self.partition}
                        #SBATCH --job-name={self.job_name}
                        #SBATCH --gres=gpu:{self.gpus}
                        #SBATCH --cpus-per-task={self.cpus_per_task}
                        #SBATCH --mem={self.mem_gb}GB
                        #SBATCH --time={self.time_hours}:00:00
                        {self._account_line()}#SBATCH --output=logs/%x-%j.out
                        #SBATCH --error=logs/%x-%j.err

                        {self._cluster_cfg.load_env}

                        mkdir -p logs
                        echo "Workingdir: $PWD"
                        echo "Started at $(date)"
                        echo "Running job $SLURM_JOB_NAME (JID $SLURM_JOB_ID) on $SLURM_JOB_PARTITION"

                        {self._python_path_setup()}

                        {cmd}
                        """
        path = save_to or self.script_path
        Path(path).write_text(script)
        return Path(path)

    def build_neps_configs_script(
        self,
        base_config: str,
        neps_results_dir: str,
        start_config: int = 1,
        end_config: int = 50,
        save_neps_report: bool = True,
        save_to: Optional[Path] = None,
    ) -> Path:
        """Build a Slurm script that runs NePS configs sequentially (like run_neps_configs.sh)."""

        configs_dir = f"{neps_results_dir}/configs"
        script = f"""#!/bin/bash
                    #SBATCH --partition={self.partition}
                    #SBATCH --job-name={self.job_name}
                    #SBATCH --gres=gpu:1
                    #SBATCH --cpus-per-task={self.cpus_per_task}
                    #SBATCH --mem={self.mem_gb}GB
                    #SBATCH --time={self.time_hours}:00:00
                    {self._account_line()}#SBATCH --output=logs/from_neps_configs/%x-%j.out
                    #SBATCH --error=logs/from_neps_configs/%x-%j.err

                    {self._cluster_cfg.load_env}

                    mkdir -p logs/from_neps_configs
                    echo "Workingdir: $PWD"
                    echo "Started at $(date)"

                    {self._python_path_setup()}

                    BASE_CONFIG="{base_config}"
                    NEPS_RESULTS_DIR="{neps_results_dir}"
                    CONFIGS_DIR="{configs_dir}"
                    START_CONFIG="{start_config}"
                    END_CONFIG="{end_config}"
                    SAVE_NEPS_REPORT={'1' if save_neps_report else '0'}

                    echo "Base config: $BASE_CONFIG"
                    echo "Configs dir: $CONFIGS_DIR"
                    echo "Config range: ${{START_CONFIG}} to ${{END_CONFIG}}"

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
                        echo "Running config_${{config_num}} at $(date)"
                        echo "=============================================="

                        TRAIN_ARGS=(--config="$BASE_CONFIG" --config_overrides="$config_file" --job_cluster="$SLURM_JOB_ID" --config_number="$config_num" --neps_report_dir="$NEPS_RESULTS_DIR")
                        [[ "$SAVE_NEPS_REPORT" == "1" ]] && TRAIN_ARGS+=(--save_neps_report)
                        python train.py "${{TRAIN_ARGS[@]}}"

                        if [[ $? -ne 0 ]]; then
                            echo "ERROR: config_${{config_num}} failed"
                        fi
                    done

                    echo "Finished at $(date)"
                    """
        path = save_to or self.script_path
        Path(path).write_text(script)
        return Path(path)

    def build_array_job_script(
        self,
        base_config: str,
        neps_results_dir: str,
        array_start: int,
        array_end: int,
        save_neps_report: bool = True,
        save_to: Optional[Path] = None,
    ) -> Path:
        """Build a Slurm array job where each task runs one NePS config."""

        configs_dir = f"{neps_results_dir}/configs"
        script = f"""#!/bin/bash
                        #SBATCH --partition={self.partition}
                        #SBATCH --job-name={self.job_name}
                        #SBATCH --gres=gpu:1
                        #SBATCH --cpus-per-task={self.cpus_per_task}
                        #SBATCH --mem={self.mem_gb}GB
                        #SBATCH --time={self.time_hours}:00:00
                        #SBATCH --array={array_start}-{array_end}
                        {self._account_line()}#SBATCH --output=logs/from_neps_configs/%x-%A_%a.out
                        #SBATCH --error=logs/from_neps_configs/%x-%A_%a.err

                        {self._cluster_cfg.load_env}

                        mkdir -p logs/from_neps_configs
                        echo "Array task ${{SLURM_ARRAY_TASK_ID}}"
                        echo "Started at $(date)"

                        {self._python_path_setup()}

                        CONFIG_DIR="{configs_dir}/config_${{SLURM_ARRAY_TASK_ID}}"
                        CONFIG_FILE="${{CONFIG_DIR}}/config.yaml"

                        if [[ ! -f "$CONFIG_FILE" ]]; then
                            echo "ERROR: No config at $CONFIG_FILE"
                            exit 1
                        fi

                        TRAIN_ARGS=(--config="{base_config}" --config_overrides="$CONFIG_FILE" --job_cluster="$SLURM_JOB_ID" --config_number="${{SLURM_ARRAY_TASK_ID}}" --neps_report_dir="{neps_results_dir}")
                        [[ "{'1' if save_neps_report else '0'}" == "1" ]] && TRAIN_ARGS+=(--save_neps_report)
                        python train.py "${{TRAIN_ARGS[@]}}"
                        """
        path = save_to or self.script_path
        Path(path).write_text(script)
        return Path(path)

    def build_probe_batch_size_script(
        self,
        config_path: str,
        model_size: Optional[ModelSize] = None,
        save_to: Optional[Path] = None,
    ) -> Path:
        """Build a Slurm script to probe max batch size from low to high."""

        script = f"""#!/bin/bash
                    #SBATCH --partition={self.partition}
                    #SBATCH --job-name=probe_batch_size
                    #SBATCH --gres=gpu:1
                    #SBATCH --cpus-per-task={self.cpus_per_task}
                    #SBATCH --mem={self.mem_gb}GB
                    #SBATCH --time=00:30:00
                    {self._account_line()}#SBATCH --output=logs/probe_batch_size-%j.out
                    #SBATCH --error=logs/probe_batch_size-%j.err

                    {self._cluster_cfg.load_env}

                    mkdir -p logs
                    {self._python_path_setup()}

                    python -u scripts/probe_batch_size.py --config="{config_path}"{f' --model_size={model_size.value}' if model_size else ''}
                    """
        path = save_to or (self.root_dir / "probe_batch_size.sh")
        Path(path).write_text(script)
        return Path(path)

    def submit_job(self, script_path: Optional[Path] = None, wait: bool = False) -> Optional[int]:
        """Submit the Slurm script using sbatch."""
        path = script_path or self.script_path
        if not path.exists():
            raise FileNotFoundError(f"Slurm script not found: {path}")

        cmd = ["sbatch"]
        if wait:
            cmd.append("--wait")
        cmd.append(str(path))

        print("Submitting:", " ".join(cmd))
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = result.stdout.strip()
        job_id = None
        if stdout.startswith("Submitted batch job"):
            job_id = int(stdout.split()[-1])
            self.job_id = job_id
            print("Job ID:", job_id)
        else:
            print("STDOUT:", stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        return job_id


def main():
    """Example: generate and optionally submit a Slurm script."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_config", default="/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/chinchilla_scaling_configs/50M_1BT_scaled.yaml")
    p.add_argument(
        "--neps_results", default="/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/neps/neps_configs/SlimPajama_50M_1BT")
    p.add_argument("--start_config", type=int, default=17)
    p.add_argument("--end_config", type=int, default=50)
    p.add_argument(
        "--mode", choices=["sequential", "array", "single", "probe"], default="sequential")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path("/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench")
    mgr = SlurmManager(
        job_name="Train_SlimPajama_50M_1BT_NePS",
        root_dir=root,
        workspace_dir=str(root),
    )

    out_path = Path(args.out) if args.out else None

    if args.mode == "sequential":
        path = mgr.build_neps_configs_script(
            base_config=args.base_config,
            neps_results_dir=args.neps_results,
            start_config=args.start_config,
            end_config=args.end_config,
            save_to=out_path,
        )
    elif args.mode == "array":
        path = mgr.build_array_job_script(
            base_config=args.base_config,
            neps_results_dir=args.neps_results,
            array_start=args.start_config,
            array_end=args.end_config,
            save_to=out_path,
        )
    elif args.mode == "single":
        config_file = f"{args.neps_results}/configs/config_{args.start_config}/config.yaml"
        path = mgr.build_train_script(
            base_config=args.base_config,
            config_overrides=config_file,
            config_number=args.start_config,
            neps_report_dir=args.neps_results,
            save_neps_report=True,
            save_to=out_path,
        )
    else:  # probe
        path = mgr.build_probe_batch_size_script(
            config_path=args.base_config,
            model_size=ModelSize.MEDIUM,
            save_to=out_path,
        )

    print(f"Wrote script to {path}")

    if args.submit:
        mgr.submit_job(path)


if __name__ == "__main__":
    main()
