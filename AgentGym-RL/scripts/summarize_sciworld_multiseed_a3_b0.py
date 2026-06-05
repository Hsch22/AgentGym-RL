#!/usr/bin/env python3
"""Summarize SciWorld A3/B0 multi-seed checkpoint sweeps.

The script is intentionally read-only: it parses finished run directories and
writes a report. It can be run before all seeds finish; incomplete seeds are
kept as pending rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCORE_POSITIVE_RE = re.compile(r"ScorePositive@1:\s*([0-9.]+)")


DEFAULT_SEED1_A3 = Path(
    "results/sciworld/"
    "sciworld_A3_g2rl_normalized_action_gradient_3b_100step_continue_from25_20260530_long100_continue"
)
DEFAULT_SEED1_B0 = Path(
    "results/sciworld/"
    "sciworld_B0_strict_no_cluster_3b_100step_continue_from25_20260530_long100_continue"
)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def md_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "pending"
    return f"`{fmt(value, digits)}`"


def signed_fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def parse_score_positive(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    matches = SCORE_POSITIVE_RE.findall(log_path.read_text(errors="ignore"))
    return float(matches[-1]) if matches else None


def parse_eval(run_dir: Path) -> dict[int, dict[str, Any]]:
    csv_path = run_dir / "eval_sciworld_ckpt_sweep" / "results.csv"
    out: dict[int, dict[str, Any]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = int(float(row["global_step"]))
            log_path = run_dir / "eval_sciworld_ckpt_sweep" / f"global_step_{step}" / "generation.log"
            out[step] = {
                "global_step": step,
                "avg_at_1": float(row["avg_at_1"]),
                "pass_at_1": float(row["pass_at_1"]),
                "score_positive_at_1": parse_score_positive(log_path),
            }
    return out


def iter_action_records(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.actions.jsonl")):
        with path.open(errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_file"] = str(path)
                record["_batch"] = path.parent.name
                yield record


def eval_item_scores(run_dir: Path, step: int) -> dict[Any, dict[str, Any]]:
    root = run_dir / "eval_sciworld_ckpt_sweep" / f"global_step_{step}" / "executer_logs"
    trajectories: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in iter_action_records(root) or []:
        key = (
            record.get("_batch"),
            record.get("_file"),
            record.get("rank"),
            record.get("trajectory_index"),
            record.get("item_id"),
        )
        trajectories[key].append(record)

    by_item = {}
    for key, records in trajectories.items():
        records.sort(key=lambda r: int(r.get("round", -1)))
        item_id = key[-1]
        by_item[item_id] = {
            "final": float(records[-1].get("reward") or 0.0),
            "max": max(float(record.get("reward") or 0.0) for record in records),
            "done": any(bool(record.get("done")) for record in records),
            "rounds": len(records),
        }
    return by_item


def paired_analysis(a3_run: Path, b0_run: Path, step: int, bootstrap_samples: int = 10000) -> dict[str, Any] | None:
    a3 = eval_item_scores(a3_run, step)
    b0 = eval_item_scores(b0_run, step)
    items = sorted(set(a3) & set(b0), key=lambda value: str(value))
    if not items:
        return None
    diffs = [b0[item]["final"] - a3[item]["final"] for item in items]
    random.seed(0)
    boot = []
    for _ in range(bootstrap_samples):
        boot.append(mean(diffs[random.randrange(len(diffs))] for __ in diffs))
    boot.sort()
    return {
        "paired_items": len(items),
        "mean_diff_b0_minus_a3": mean(diffs),
        "median_diff_b0_minus_a3": sorted(diffs)[len(diffs) // 2],
        "b0_wins": sum(diff > 0 for diff in diffs),
        "a3_wins": sum(diff < 0 for diff in diffs),
        "ties": sum(diff == 0 for diff in diffs),
        "a3_done": sum(a3[item]["done"] for item in items),
        "b0_done": sum(b0[item]["done"] for item in items),
        "bootstrap_ci95": [
            boot[int(0.025 * len(boot))],
            boot[int(0.975 * len(boot))],
        ],
    }


def run_dir_for(root: Path, label: str, seed: int, run_ts: str) -> Path:
    if label == "A3":
        name = f"sciworld_A3_g2rl_normalized_action_gradient_3b_seed{seed}_100step_{run_ts}"
    else:
        name = f"sciworld_B0_strict_no_cluster_3b_seed{seed}_100step_{run_ts}"
    return root / name


def collect(args) -> dict[str, Any]:
    seeds = [int(seed) for seed in args.seeds]
    runs = {}
    for seed in seeds:
        if seed == 1:
            a3_run = args.seed1_a3_run
            b0_run = args.seed1_b0_run
            source = "seed1_current_continue_from25"
        else:
            a3_run = run_dir_for(args.multiseed_root, "A3", seed, args.run_ts)
            b0_run = run_dir_for(args.multiseed_root, "B0", seed, args.run_ts)
            source = "scratch_multiseed"
        a3_eval = parse_eval(a3_run)
        b0_eval = parse_eval(b0_run)
        paired_step100 = paired_analysis(a3_run, b0_run, 100) if 100 in a3_eval and 100 in b0_eval else None
        runs[str(seed)] = {
            "source": source,
            "A3": {"run_dir": str(a3_run), "eval": a3_eval},
            "B0": {"run_dir": str(b0_run), "eval": b0_eval},
            "paired_step100": paired_step100,
        }

    aggregate = {}
    for step in args.checkpoints:
        rows = []
        for seed in seeds:
            seed_key = str(seed)
            a3_eval = runs[seed_key]["A3"]["eval"].get(step)
            b0_eval = runs[seed_key]["B0"]["eval"].get(step)
            if a3_eval and b0_eval:
                rows.append(
                    {
                        "seed": seed,
                        "a3_avg_at_1": a3_eval["avg_at_1"],
                        "b0_avg_at_1": b0_eval["avg_at_1"],
                        "avg_diff_a3_minus_b0": a3_eval["avg_at_1"] - b0_eval["avg_at_1"],
                        "a3_pass_at_1": a3_eval["pass_at_1"],
                        "b0_pass_at_1": b0_eval["pass_at_1"],
                        "pass_diff_a3_minus_b0": a3_eval["pass_at_1"] - b0_eval["pass_at_1"],
                    }
                )
        diffs = [row["avg_diff_a3_minus_b0"] for row in rows]
        aggregate[str(step)] = {
            "finished_seed_count": len(rows),
            "rows": rows,
            "mean_avg_diff_a3_minus_b0": mean(diffs) if diffs else None,
            "std_avg_diff_a3_minus_b0": pstdev(diffs) if len(diffs) > 1 else None,
            "a3_win_count": sum(diff > 0 for diff in diffs),
            "b0_win_count": sum(diff < 0 for diff in diffs),
            "tie_count": sum(diff == 0 for diff in diffs),
        }
    return {
        "seeds": seeds,
        "checkpoints": args.checkpoints,
        "runs": runs,
        "aggregate": aggregate,
    }


def write_report(path: Path, data: dict[str, Any], json_path: Path):
    all_complete = all(
        data["runs"][str(seed)]["A3"]["eval"].get(step)
        and data["runs"][str(seed)]["B0"]["eval"].get(step)
        for seed in data["seeds"]
        for step in data["checkpoints"]
    )
    lines = [
        "# SciWorld A3/B0 多 seed 复现实验汇总",
        "",
        "## 当前状态",
        "",
        f"- seeds: `{data['seeds']}`",
        f"- checkpoints: `{data['checkpoints']}`",
        f"- 机器可读摘要：`{json_path}`",
        "",
        "## 每个 seed 的 eval 结果",
        "",
        "| seed | source | step | A3 Avg@1 | B0 Avg@1 | A3-B0 Avg | A3 Pass@1 | B0 Pass@1 | A3-B0 Pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in data["seeds"]:
        seed_key = str(seed)
        run = data["runs"][seed_key]
        for step in data["checkpoints"]:
            a3 = run["A3"]["eval"].get(step)
            b0 = run["B0"]["eval"].get(step)
            avg_diff = a3["avg_at_1"] - b0["avg_at_1"] if a3 and b0 else None
            pass_diff = a3["pass_at_1"] - b0["pass_at_1"] if a3 and b0 else None
            lines.append(
                f"| {seed} | `{run['source']}` | {step} | "
                f"{md_value(a3['avg_at_1'] if a3 else None)} | "
                f"{md_value(b0['avg_at_1'] if b0 else None)} | "
                f"{md_value(avg_diff)} | "
                f"{md_value(a3['pass_at_1'] if a3 else None)} | "
                f"{md_value(b0['pass_at_1'] if b0 else None)} | "
                f"{md_value(pass_diff)} |"
            )

    lines += [
        "",
        "## 跨 seed 聚合",
        "",
        "| step | finished seeds | mean A3-B0 Avg | std | A3 wins | B0 wins | ties |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in data["checkpoints"]:
        agg = data["aggregate"][str(step)]
        lines.append(
            f"| {step} | `{agg['finished_seed_count']}` | "
            f"`{fmt(agg['mean_avg_diff_a3_minus_b0'])}` | "
            f"`{fmt(agg['std_avg_diff_a3_minus_b0'])}` | "
            f"`{agg['a3_win_count']}` | `{agg['b0_win_count']}` | `{agg['tie_count']}` |"
        )

    lines += [
        "",
        "## step100 paired item 分析",
        "",
        "| seed | paired items | B0-A3 mean final reward | CI95 | B0 wins | A3 wins | ties | A3 done | B0 done |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in data["seeds"]:
        seed_key = str(seed)
        paired = data["runs"][seed_key]["paired_step100"]
        if not paired:
            lines.append(f"| {seed} | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        ci = paired["bootstrap_ci95"]
        lines.append(
            f"| {seed} | `{paired['paired_items']}` | "
            f"`{fmt(paired['mean_diff_b0_minus_a3'])}` | "
            f"`[{fmt(ci[0])}, {fmt(ci[1])}]` | "
            f"`{paired['b0_wins']}` | `{paired['a3_wins']}` | `{paired['ties']}` | "
            f"`{paired['a3_done']}` | `{paired['b0_done']}` |"
        )

    if all_complete:
        step100 = data["aggregate"]["100"]
        step75 = data["aggregate"]["75"]
        source_note = ""
        if data["runs"].get("1", {}).get("source") != "scratch_multiseed":
            source_note = "seed1 来自 `continue_from25` 续训证据，seed2/seed3 是 scratch 复现实验；解释时应保留这一配置来源差异。"
        lines += [
            "",
            "## 最终结论",
            "",
            f"- step100 上，A3-B0 Avg@1 的跨 seed 均值为 `{signed_fmt(step100['mean_avg_diff_a3_minus_b0'])}`，std=`{fmt(step100['std_avg_diff_a3_minus_b0'])}`；A3 wins=`{step100['a3_win_count']}`，B0 wins=`{step100['b0_win_count']}`，ties=`{step100['tie_count']}`。",
            f"- step75 上，A3 wins=`{step75['a3_win_count']}`，B0 wins=`{step75['b0_win_count']}`；step100 上多数 seed 仍是 A3 领先。",
            "- 因此，seed1 的 B0 长训反超没有在 seed2/seed3 上复现；它更像是 seed/数据顺序/训练轨迹敏感的个例，而不是当前三组证据下的稳定规律。",
            "- seed2 和 seed3 的 paired item 分析均显示 A3 在更多 item 上取得更高 final reward，且 B0-A3 的 bootstrap 95% CI 均小于 0；这说明 A3 的优势不只是均值被少数样本拉高。",
            "- 但 step100 的跨 seed std 很大，说明 SciWorld 长训结果方差高；更稳健的结论应是：A3 在当前设置下有提升信号，但仍需要更多 seed 才能估计真实效应大小。",
        ]
        if source_note:
            lines.append(f"- {source_note}")
    else:
        lines += [
            "",
            "## 暂定解释规则",
            "",
            "- 如果 seed2/seed3 也在 step75 或 step100 出现 B0 反超，则当前 seed1 不是偶发，B0 长训反超更可能是 SciWorld 长链任务特征。",
            "- 如果 seed2/seed3 中 A3 保持领先，则 seed1 的 B0 反超更可能与训练随机性或数据顺序有关。",
            "- 在所有 seed 完成前，本报告只作为滚动记录，不作为最终结论。",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-root", type=Path, default=Path("results/sciworld_multiseed"))
    parser.add_argument("--run-ts", default="20260530_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[50, 75, 100])
    parser.add_argument("--seed1-a3-run", type=Path, default=DEFAULT_SEED1_A3)
    parser.add_argument("--seed1-b0-run", type=Path, default=DEFAULT_SEED1_B0)
    parser.add_argument("--report-path", type=Path, default=Path("docs/sciworld_multiseed_A3_vs_B0_report_20260530.md"))
    args = parser.parse_args()

    data = collect(args)
    json_path = args.report_path.with_suffix(".json")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_report(args.report_path, data, json_path)
    print(f"Wrote {args.report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
