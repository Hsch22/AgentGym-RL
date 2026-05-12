#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import wandb


AVG_RE = re.compile(r"Avg@(\d+):\s*([0-9.]+)")
PASS_RE = re.compile(r"Pass@(\d+):\s*([0-9.]+)")
CATEGORY_RE = re.compile(r"Category:\s*(.+)")


def has_merged_weights(hf_dir: Path) -> bool:
    if not hf_dir.exists():
        return False
    for pattern in ("*.safetensors", "*.bin"):
        if list(hf_dir.glob(pattern)):
            return True
    return False


def ensure_merged(project_root: Path, actor_dir: Path):
    hf_dir = actor_dir / "huggingface"
    if has_merged_weights(hf_dir):
        return
    # 原因：FSDP checkpoint 评测前必须先 merge 成 HuggingFace 权重，main_generation 才能加载。
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "model_merger.py"),
        "--local_dir",
        str(actor_dir),
    ]
    subprocess.run(cmd, cwd=project_root, check=True)


def parse_eval_metrics(log_path: Path):
    metrics = {}
    category = None
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = CATEGORY_RE.match(line)
            if m:
                category = m.group(1)
                continue

            m = AVG_RE.match(line)
            if m:
                metric_name = f"avg_at_{m.group(1)}"
                value = float(m.group(2))
                metrics[metric_name if category is None else f"{category}/{metric_name}"] = value
                continue

            m = PASS_RE.match(line)
            if m:
                metric_name = f"pass_at_{m.group(1)}"
                value = float(m.group(2))
                metrics[metric_name if category is None else f"{category}/{metric_name}"] = value
    return metrics


def run_generation(
    project_root: Path,
    eval_data_dir: Path,
    actor_dir: Path,
    output_dir: Path,
    env_addr: str,
    n_gpus: int,
    cuda_visible_devices: str,
    batch_size: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation.log"
    cmd = [
        sys.executable,
        "-m",
        "verl.agent_trainer.main_generation",
        f"data.path={eval_data_dir}",
        "data.max_prompt_length=750",
        "data.max_response_length=14098",
        "data.n_samples=1",
        f"data.batch_size={batch_size}",
        "agentgym.task_name=searchqa",
        f"agentgym.env_addr={env_addr}",
        "agentgym.max_rounds=30",
        "agentgym.timeout=500",
        f"model.path={actor_dir / 'huggingface'}",
        "rollout.gpu_memory_utilization=0.95",
        "rollout.temperature=1",
        "rollout.max_model_len=32768",
        "rollout.max_tokens=512",
        "rollout.tensor_model_parallel_size=1",
        f"rollout.rollout_log_dir={output_dir / 'executer_logs'}",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={n_gpus}",
    ]

    env = os.environ.copy()
    # 注意：评测连接本机 env server，清掉代理避免 localhost 请求绕出本机。
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["VLLM_USE_MODELSCOPE"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    env["HYDRA_FULL_ERROR"] = "1"

    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=project_root, env=env, check=False, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        metrics = parse_eval_metrics(log_path)
        # 注意：generation 可能在收尾阶段报错；已有完整指标时仍保留该 checkpoint 结果。
        if "avg_at_1" not in metrics or "pass_at_1" not in metrics:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-data-dir", required=True)
    parser.add_argument("--env-addr", default="http://127.0.0.1:36015")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=int)
    parser.add_argument("--n-gpus", type=int, default=6)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wandb-project", default="agentgym-rl-eval")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    eval_data_dir = Path(args.eval_data_dir).resolve()
    results_dir = run_dir / "eval_searchqa_ckpt_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    cuda_visible_devices = args.cuda_visible_devices or ",".join(str(i) for i in range(args.n_gpus))

    wandb_run = None
    if args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = args.wandb_mode
        # 原因：checkpoint sweep 的结果按 step 记录到同一个 W&B run，便于横向比较。
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"searchqa_eval_{run_dir.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            config={
                "run_dir": str(run_dir),
                "eval_data_dir": str(eval_data_dir),
                "env_addr": args.env_addr,
                "checkpoints": args.checkpoints,
                "n_gpus": args.n_gpus,
                "cuda_visible_devices": cuda_visible_devices,
                "batch_size": args.batch_size,
            },
        )

    results = []
    try:
        for step in args.checkpoints:
            ckpt_dir = run_dir / f"global_step_{step}" / "actor"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}")

            ensure_merged(project_root, ckpt_dir)
            ckpt_output_dir = results_dir / f"global_step_{step}"
            log_path = run_generation(
                project_root=project_root,
                eval_data_dir=eval_data_dir,
                actor_dir=ckpt_dir,
                output_dir=ckpt_output_dir,
                env_addr=args.env_addr,
                n_gpus=args.n_gpus,
                cuda_visible_devices=cuda_visible_devices,
                batch_size=args.batch_size,
            )
            metrics = parse_eval_metrics(log_path)
            row = {"global_step": step, **metrics}
            results.append(row)

            if wandb_run is not None:
                payload = {f"eval/{k}": v for k, v in metrics.items()}
                payload["global_step"] = step
                wandb.log(payload, step=step)

            with (results_dir / "results.json").open("w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

            fieldnames = sorted({k for r in results for k in r.keys()})
            with (results_dir / "results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
