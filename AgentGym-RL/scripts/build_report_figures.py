#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TEXTCRAFT_TRAIN_RUNS = [
    {
        "label": "Standard rollout",
        "project": "agentgym-rl",
        "run_id": "va3qfoz6",
        "run_name": "textcraft_scalinginter_baseline_qwen25_1p5b_2xh200",
        "path": "results/wandb_projects/agentgym-rl/runs/09_va3qfoz6_textcraft_scalinginter_baseline_qwen25_1p5b_2xh200/history.csv",
    },
    {
        "label": "Response-feature G2RL",
        "project": "agentgym-rl",
        "run_id": "njnmkizg",
        "run_name": "textcraft_g2rl_response_qwen25_1p5b_2xh200",
        "path": "results/wandb_projects/agentgym-rl/runs/10_njnmkizg_textcraft_g2rl_response_qwen25_1p5b_2xh200/history.csv",
    },
]

TEXTCRAFT_EVAL_RUNS = [
    {
        "label": "Standard rollout",
        "project": "agentgym-rl-eval",
        "run_id": "j69xtiwp",
        "run_name": "textcraft_eval_qwen25_1p5b_baseline_2xh200",
        "path": "results/wandb_projects/agentgym-rl-eval/runs/09_j69xtiwp_textcraft_eval_qwen25_1p5b_baseline_2xh200/history.csv",
    },
    {
        "label": "Response-feature G2RL",
        "project": "agentgym-rl-eval",
        "run_id": "lf2duvnj",
        "run_name": "textcraft_eval_qwen25_1p5b_g2rl_response_2xh200",
        "path": "results/wandb_projects/agentgym-rl-eval/runs/10_lf2duvnj_textcraft_eval_qwen25_1p5b_g2rl_response_2xh200/history.csv",
    },
]

SCIWORLD_EVAL_RUNS = [
    {
        "method": "NAGC",
        "seed": 1,
        "source": "continuation",
        "path": "results/sciworld/sciworld_A3_g2rl_normalized_action_gradient_3b_100step_continue_from25_20260530_long100_continue/eval_sciworld_ckpt_sweep/results.csv",
    },
    {
        "method": "Standard rollout",
        "seed": 1,
        "source": "continuation",
        "path": "results/sciworld/sciworld_B0_strict_no_cluster_3b_100step_continue_from25_20260530_long100_continue/eval_sciworld_ckpt_sweep/results.csv",
    },
    {
        "method": "NAGC",
        "seed": 2,
        "source": "scratch",
        "path": "results/sciworld_multiseed/sciworld_A3_g2rl_normalized_action_gradient_3b_seed2_100step_20260530_multiseed/eval_sciworld_ckpt_sweep/results.csv",
    },
    {
        "method": "Standard rollout",
        "seed": 2,
        "source": "scratch",
        "path": "results/sciworld_multiseed/sciworld_B0_strict_no_cluster_3b_seed2_100step_20260530_multiseed/eval_sciworld_ckpt_sweep/results.csv",
    },
    {
        "method": "NAGC",
        "seed": 3,
        "source": "scratch",
        "path": "results/sciworld_multiseed/sciworld_A3_g2rl_normalized_action_gradient_3b_seed3_100step_20260530_multiseed/eval_sciworld_ckpt_sweep/results.csv",
    },
    {
        "method": "Standard rollout",
        "seed": 3,
        "source": "scratch",
        "path": "results/sciworld_multiseed/sciworld_B0_strict_no_cluster_3b_seed3_100step_20260530_multiseed/eval_sciworld_ckpt_sweep/results.csv",
    },
]


COLORS = {
    "Standard rollout": "#4C78A8",
    "Response-feature G2RL": "#F58518",
    "NAGC": "#54A24B",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relpath(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_textcraft_response_feature_figure(root: Path, fig_dir: Path, data_dir: Path) -> dict:
    train_rows = []
    train_frames = []
    for spec in TEXTCRAFT_TRAIN_RUNS:
        path = root / spec["path"]
        require_file(path)
        df = pd.read_csv(path)
        part = pd.DataFrame(
            {
                "step": clean_numeric(df["_step"]),
                "value": clean_numeric(df["critic/task_score/mean"]),
            }
        ).dropna()
        part["label"] = spec["label"]
        part["metric"] = "critic/task_score/mean"
        train_frames.append(part)
        train_rows.extend(part.to_dict("records"))

    eval_rows = []
    eval_frames = []
    for spec in TEXTCRAFT_EVAL_RUNS:
        path = root / spec["path"]
        require_file(path)
        df = pd.read_csv(path)
        step_col = "global_step" if "global_step" in df.columns else "_step"
        part = pd.DataFrame(
            {
                "step": clean_numeric(df[step_col]),
                "value": clean_numeric(df["eval/avg_at_1"]),
            }
        ).dropna()
        part["label"] = spec["label"]
        part["metric"] = "eval/avg_at_1"
        eval_frames.append(part)
        eval_rows.extend(part.to_dict("records"))

    write_csv(data_dir / "textcraft_response_feature_train.csv", train_rows)
    write_csv(data_dir / "textcraft_response_feature_eval.csv", eval_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    for frame in train_frames:
        label = frame["label"].iloc[0]
        frame = frame.sort_values("step")
        axes[0].plot(
            frame["step"],
            moving_average(frame["value"]),
            color=COLORS[label],
            linewidth=1.7,
            label=label,
        )
    axes[0].set_title("Training task score")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean score")
    axes[0].set_xlim(0, 330)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, linewidth=0.35, alpha=0.35)

    for frame in eval_frames:
        label = frame["label"].iloc[0]
        frame = frame.sort_values("step")
        axes[1].plot(
            frame["step"],
            frame["value"],
            marker="o",
            markersize=3.0,
            color=COLORS[label],
            linewidth=1.5,
            label=label,
        )
    axes[1].set_title("Checkpoint Eval Avg@1")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Avg@1")
    axes[1].set_xlim(20, 335)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, linewidth=0.35, alpha=0.35)
    axes[1].legend(frameon=False, loc="lower right")

    save_figure(fig, fig_dir / "textcraft_response_feature_curves")
    return {
        "figure": "textcraft_response_feature_curves",
        "sources": [
            {"label": spec["label"], "split": "training"}
            for spec in TEXTCRAFT_TRAIN_RUNS
        ]
        + [
            {"label": spec["label"], "split": "checkpoint evaluation"}
            for spec in TEXTCRAFT_EVAL_RUNS
        ],
        "data": [
            relpath(root, data_dir / "textcraft_response_feature_train.csv"),
            relpath(root, data_dir / "textcraft_response_feature_eval.csv"),
        ],
    }


def build_sciworld_multiseed_figure(root: Path, fig_dir: Path, data_dir: Path) -> dict:
    rows = []
    for spec in SCIWORLD_EVAL_RUNS:
        path = root / spec["path"]
        require_file(path)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "method": spec["method"],
                    "seed": spec["seed"],
                    "source": spec["source"],
                    "global_step": int(row["global_step"]),
                    "avg_at_1": float(row["avg_at_1"]),
                    "pass_at_1": float(row["pass_at_1"]),
                }
            )
    long_df = pd.DataFrame(rows)
    write_csv(data_dir / "sciworld_multiseed_nagc_standard_eval.csv", rows)

    summary_rows = []
    grouped = (
        long_df.groupby(["method", "global_step"])["avg_at_1"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["method", "global_step"])
    )
    for _, row in grouped.iterrows():
        summary_rows.append(
            {
                "method": row["method"],
                "global_step": int(row["global_step"]),
                "mean_avg_at_1": float(row["mean"]),
                "std_avg_at_1": 0.0 if math.isnan(row["std"]) else float(row["std"]),
            }
        )

    pivot = long_df.pivot_table(
        index=["seed", "source", "global_step"], columns="method", values="avg_at_1"
    ).reset_index()
    pivot["diff_nagc_minus_standard"] = pivot["NAGC"] - pivot["Standard rollout"]
    for _, row in pivot.iterrows():
        summary_rows.append(
            {
                "method": "NAGC - Standard rollout",
                "seed": int(row["seed"]),
                "source": row["source"],
                "global_step": int(row["global_step"]),
                "diff_avg_at_1": float(row["diff_nagc_minus_standard"]),
            }
        )
    write_csv(data_dir / "sciworld_multiseed_nagc_standard_summary.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    for method in ["Standard rollout", "NAGC"]:
        part = grouped[grouped["method"] == method].sort_values("global_step")
        x = part["global_step"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        std = part["std"].fillna(0).to_numpy(dtype=float)
        axes[0].plot(
            x,
            y,
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            color=COLORS[method],
            label=method,
        )
        axes[0].fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.14, linewidth=0)
    axes[0].set_title("Mean SciWorld Avg@1")
    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_ylabel("Avg@1")
    axes[0].set_xticks([50, 75, 100])
    axes[0].grid(True, linewidth=0.35, alpha=0.35)
    axes[0].legend(frameon=False, loc="upper left")

    for (seed, _source), part in pivot.groupby(["seed", "source"]):
        label = f"seed {seed}"
        axes[1].plot(
            part["global_step"],
            part["diff_nagc_minus_standard"],
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            alpha=0.75,
            color="#6F6F6F",
            label=label,
        )
    mean_diff = (
        pivot.groupby("global_step")["diff_nagc_minus_standard"]
        .mean()
        .reset_index()
        .sort_values("global_step")
    )
    axes[1].plot(
        mean_diff["global_step"],
        mean_diff["diff_nagc_minus_standard"],
        marker="o",
        markersize=3.4,
        linewidth=1.9,
        color="#111111",
        label="mean",
    )
    axes[1].axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    axes[1].set_title("Paired Avg@1 gain")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("NAGC - standard")
    axes[1].set_xticks([50, 75, 100])
    axes[1].grid(True, linewidth=0.35, alpha=0.35)
    axes[1].legend(frameon=False, loc="upper left", fontsize=7)

    save_figure(fig, fig_dir / "sciworld_multiseed_nagc_standard_curves")
    return {
        "figure": "sciworld_multiseed_nagc_standard_curves",
        "sources": [
            {key: spec[key] for key in ("method", "seed", "source")}
            for spec in SCIWORLD_EVAL_RUNS
        ],
        "data": [
            relpath(root, data_dir / "sciworld_multiseed_nagc_standard_eval.csv"),
            relpath(root, data_dir / "sciworld_multiseed_nagc_standard_summary.csv"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fig-dir", default="docs/final/technical_report/figures")
    parser.add_argument("--data-dir", default="docs/final/technical_report/figures/data")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fig_dir = (root / args.fig_dir).resolve()
    data_dir = (root / args.data_dir).resolve()
    setup_style()

    manifest = {
        "textcraft_response_feature": build_textcraft_response_feature_figure(root, fig_dir, data_dir),
        "sciworld_multiseed_nagc_standard": build_sciworld_multiseed_figure(root, fig_dir, data_dir),
    }
    (data_dir / "figure_sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for key, value in manifest.items():
        print(f"{key}: {value['figure']}.pdf")


if __name__ == "__main__":
    main()
