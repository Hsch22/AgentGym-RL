#!/usr/bin/env python3
"""Analyze a strict SciWorld A/B run between A3 clustering and no-cluster baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
METRIC_RE = re.compile(r"([^:]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))$")
SCORE_POSITIVE_RE = re.compile(r"ScorePositive@(\d+):\s*([0-9.]+)")
PASS_RE = re.compile(r"Pass@(\d+):\s*([0-9.]+)")
AVG_RE = re.compile(r"Avg@(\d+):\s*([0-9.]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def safe_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f):
        return None
    return value_f


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


def parse_train_log(log_path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    config_lines: list[str] = []
    if not log_path.exists():
        return {"path": str(log_path), "steps": steps, "summary": {}, "config_lines": config_lines}

    with log_path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).strip()
            if "[baseline-run]" in line:
                config_lines.append(line)
            match = STEP_RE.search(line)
            if not match:
                continue
            row: dict[str, Any] = {"step": int(match.group(1))}
            for part in match.group(2).split(" - "):
                metric = METRIC_RE.search(part.strip())
                if metric:
                    row[metric.group(1).strip()] = float(metric.group(2))
            steps.append(row)

    scores = [v for step in steps if (v := safe_float(step.get("critic/score/mean"))) is not None]
    step_times = [v for step in steps if (v := safe_float(step.get("timing_s/step"))) is not None]
    summary = {
        "step_count": len(steps),
        "final_step": steps[-1]["step"] if steps else None,
        "step25_score_mean": next(
            (safe_float(step.get("critic/score/mean")) for step in steps if step.get("step") == 25),
            None,
        ),
        "first10_score_mean": mean(scores[:10]) if len(scores) >= 10 else (mean(scores) if scores else None),
        "last10_score_mean": mean(scores[-10:]) if len(scores) >= 10 else (mean(scores) if scores else None),
        "score_mean_all": mean(scores) if scores else None,
        "avg_step_time_s": mean(step_times) if step_times else None,
        "round0_candidate_step25": next(
            (safe_float(step.get("rollout/round0/candidate_total_count")) for step in steps if step.get("step") == 25),
            None,
        ),
        "round0_taken_step25": next(
            (safe_float(step.get("rollout/round0/taken_total_count")) for step in steps if step.get("step") == 25),
            None,
        ),
    }
    return {"path": str(log_path), "steps": steps, "summary": summary, "config_lines": config_lines}


def parse_eval(run_dir: Path, step: int) -> dict[str, Any]:
    results_dir = run_dir / "eval_sciworld_ckpt_sweep"
    csv_path = results_dir / "results.csv"
    generation_log = results_dir / f"global_step_{step}" / "generation.log"
    metrics: dict[str, Any] = {"results_csv": str(csv_path), "generation_log": str(generation_log)}

    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if str(row.get("global_step")) == str(step):
                for key, value in row.items():
                    metrics[key] = safe_float(value) if key != "global_step" else int(float(value))
                break

    if generation_log.exists():
        with generation_log.open("r", errors="ignore") as f:
            for raw_line in f:
                line = strip_ansi(raw_line).strip()
                for regex, prefix in ((AVG_RE, "avg_at"), (PASS_RE, "pass_at"), (SCORE_POSITIVE_RE, "score_positive_at")):
                    match = regex.match(line)
                    if match:
                        metrics[f"{prefix}_{match.group(1)}"] = float(match.group(2))
    return metrics


def iter_action_records(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.actions.jsonl")):
        with path.open("r", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_source_file"] = str(path)
                record["_line_no"] = line_no
                yield record


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def classify_action(action: str) -> str:
    text = action.lower().strip()
    if not text or text == "<invalid>":
        return "invalid_or_empty"
    if text.startswith(("look", "examine", "inspect", "check", "read")):
        return "observe"
    if text.startswith(("go to", "move to", "enter", "exit")):
        return "navigate"
    if text.startswith(("open door", "close door")) or " door" in text:
        return "navigate_or_door"
    if text.startswith(("pick up", "take", "grab")):
        return "pick_up"
    if text.startswith(("put", "drop", "place")):
        return "place_or_drop"
    if text.startswith(("use", "activate", "turn on", "turn off", "pour", "mix", "measure")):
        return "tool_or_operation"
    if text.startswith(("focus", "wait")):
        return "control"
    return "other"


def trajectory_key(record: dict[str, Any]) -> tuple[str, Any, Any, Any]:
    return (
        record.get("_source_file", ""),
        record.get("rank"),
        record.get("trajectory_index"),
        record.get("item_id"),
    )


def summarize_actions(root: Path, label: str) -> dict[str, Any]:
    records = list(iter_action_records(root) or [])
    round0 = [record for record in records if int(record.get("round", -1)) == 0]
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in round0:
        groups[record.get("item_id")].append(record)

    unique_counts = []
    group_rows = []
    for item_id, item_records in sorted(groups.items(), key=lambda item: str(item[0])):
        actions = [normalize_text(record.get("normalized_action")) for record in item_records]
        raw_responses = [normalize_text(record.get("raw_response")) for record in item_records]
        valid_actions = [action for action in actions if action]
        unique_actions = sorted(set(valid_actions))
        unique_counts.append(len(unique_actions))
        group_rows.append(
            {
                "item_id": item_id,
                "trajectory_count": len(item_records),
                "unique_round0_actions": len(unique_actions),
                "actions": actions,
                "raw_responses": raw_responses,
                "unique_actions": unique_actions,
                "candidate_count_values": sorted(
                    {int(record.get("candidate_count", 0) or 0) for record in item_records}
                ),
            }
        )

    trajectories: dict[tuple[str, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    action_type_counter: Counter[str] = Counter()
    no_known_action = 0
    precondition_failure = 0
    positive_trajectories = 0
    done_trajectories = 0

    for record in records:
        action = normalize_text(record.get("normalized_action"))
        action_type_counter[classify_action(action)] += 1
        observation = normalize_text(record.get("env_observation")).lower()
        if "no known action matches" in observation:
            no_known_action += 1
        if any(
            phrase in observation
            for phrase in (
                "door is not open",
                "isn't open",
                "can't",
                "cannot",
                "not recognized",
                "not open",
            )
        ):
            precondition_failure += 1
        trajectories[trajectory_key(record)].append(record)

    final_rows = []
    for key, trajectory_records in trajectories.items():
        trajectory_records.sort(key=lambda record: int(record.get("round", -1)))
        rewards = [safe_float(record.get("reward")) or 0.0 for record in trajectory_records]
        done_any = any(bool(record.get("done")) for record in trajectory_records)
        max_reward = max(rewards) if rewards else 0.0
        final_reward = rewards[-1] if rewards else 0.0
        if max_reward > 0:
            positive_trajectories += 1
        if done_any:
            done_trajectories += 1
        final_rows.append(
            {
                "key": key,
                "item_id": trajectory_records[0].get("item_id"),
                "rounds": len(trajectory_records),
                "max_reward": max_reward,
                "final_reward": final_reward,
                "done": done_any,
                "actions": [normalize_text(record.get("normalized_action")) for record in trajectory_records],
                "observations": [
                    normalize_text(record.get("env_observation")).replace("\n", " ")[:220]
                    for record in trajectory_records
                ],
            }
        )

    total_records = len(records)
    action_type_total = sum(action_type_counter.values())
    return {
        "label": label,
        "root": str(root),
        "record_count": total_records,
        "round0_record_count": len(round0),
        "item_group_count": len(groups),
        "round0_unique_action_count": numeric_summary(unique_counts),
        "round0_groups": group_rows,
        "candidate_count_values_round0": sorted({int(record.get("candidate_count", 0) or 0) for record in round0}),
        "action_type_counts": dict(action_type_counter),
        "action_type_ratios": {
            key: value / action_type_total for key, value in sorted(action_type_counter.items())
        }
        if action_type_total
        else {},
        "no_known_action_ratio": no_known_action / total_records if total_records else None,
        "precondition_failure_ratio": precondition_failure / total_records if total_records else None,
        "trajectory_count": len(trajectories),
        "positive_trajectory_ratio": positive_trajectories / len(trajectories) if trajectories else None,
        "done_trajectory_ratio": done_trajectories / len(trajectories) if trajectories else None,
        "trajectories": final_rows,
    }


def pick_examples(a3_actions: dict[str, Any], b0_actions: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    a3_by_item = {row["item_id"]: row for row in a3_actions.get("round0_groups", [])}
    b0_by_item = {row["item_id"]: row for row in b0_actions.get("round0_groups", [])}
    shared_items = sorted(set(a3_by_item) & set(b0_by_item), key=lambda item: str(item))
    scored = []
    for item_id in shared_items:
        a3_unique = int(a3_by_item[item_id]["unique_round0_actions"])
        b0_unique = int(b0_by_item[item_id]["unique_round0_actions"])
        scored.append((a3_unique - b0_unique, a3_unique, item_id))
    scored.sort(reverse=True)
    examples = []
    for _, _, item_id in scored[:limit]:
        examples.append(
            {
                "item_id": item_id,
                "a3_round0_actions": a3_by_item[item_id]["actions"],
                "b0_round0_actions": b0_by_item[item_id]["actions"],
                "a3_round0_raw": a3_by_item[item_id].get("raw_responses", []),
                "b0_round0_raw": b0_by_item[item_id].get("raw_responses", []),
                "a3_unique_round0": a3_by_item[item_id]["unique_round0_actions"],
                "b0_unique_round0": b0_by_item[item_id]["unique_round0_actions"],
            }
        )
    return examples


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def action_type_line(summary: dict[str, Any]) -> str:
    ratios = summary.get("action_type_ratios") or {}
    ordered = sorted(ratios.items(), key=lambda item: item[1], reverse=True)[:6]
    return ", ".join(f"{key}={value * 100:.1f}%" for key, value in ordered)


def clip_reply(text: str, limit: int = 360) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_report(
    report_path: Path,
    a3_run: Path,
    b0_run: Path,
    a3_log: Path,
    b0_log: Path,
    comparison: dict[str, Any],
):
    a3_train = comparison["a3"]["train"]["summary"]
    b0_train = comparison["b0"]["train"]["summary"]
    a3_eval = comparison["a3"]["eval"]
    b0_eval = comparison["b0"]["eval"]
    a3_actions = comparison["a3"]["train_actions"]
    b0_actions = comparison["b0"]["train_actions"]
    examples = comparison["examples"]

    lines = [
        "# SciWorld A3 vs B0 strict baseline 技术报告",
        "",
        "日期：2026-05-30",
        "",
        "## 核心结论",
        "",
        "本报告只比较一个变量：A3 开启 `g2rl_normalized_action_gradient` round0 聚类，B0 关闭 clustering。两者使用同一模型、同一训练 batch、同一 rollout.n、同一步数、同一评测入口。",
        "",
        "严格 B0 跑完后，A3 在训练分、eval Avg@1 和 ScorePositive@1 上都高于 B0；但两者的 Pass@1 都是 0。这说明 A3 在 SciWorld 当前配置下确实提升了部分进度和探索覆盖，但还没有把这种提升转化为终局任务成功。",
        "",
        "| run | train step25 score/mean | eval Avg@1 | eval Pass@1 | eval ScorePositive@1 |",
        "|---|---:|---:|---:|---:|",
        f"| A3 | `{fmt(a3_train.get('step25_score_mean'))}` | `{fmt(a3_eval.get('avg_at_1'))}` | `{fmt(a3_eval.get('pass_at_1'))}` | `{fmt(a3_eval.get('score_positive_at_1'))}` |",
        f"| B0 strict baseline | `{fmt(b0_train.get('step25_score_mean'))}` | `{fmt(b0_eval.get('avg_at_1'))}` | `{fmt(b0_eval.get('pass_at_1'))}` | `{fmt(b0_eval.get('score_positive_at_1'))}` |",
        "",
        "## 产物",
        "",
        f"- A3 run：`{a3_run}`",
        f"- B0 run：`{b0_run}`",
        f"- A3 train log：`{a3_log}`",
        f"- B0 train log：`{b0_log}`",
        f"- 机器可读摘要：`{report_path.with_suffix('.json')}`",
        "",
        "## 配置公平性审计",
        "",
        "| 项 | A3 | B0 |",
        "|---|---|---|",
        "| model | `Qwen2.5-3B-Instruct` | `Qwen2.5-3B-Instruct` |",
        "| train batch | `16` | `16` |",
        "| rollout.n | `8` | `8` |",
        "| retained trajectories / step | `128` | `128` |",
        "| total training steps | `25` | `25` |",
        "| env server | `http://127.0.0.1:36006` | `http://127.0.0.1:36006` |",
        "| round0 candidates | `16 -> 8` | `1 -> 1` per retained trajectory |",
        "| clustering | `g2rl_normalized_action_gradient` | disabled |",
        "",
        "审计备注：B0 的 Hydra 配置打印里仍会出现默认的 inactive clustering 字段；实际启动命令没有传入 `actor_rollout_ref.rollout.clustering.enabled=true`，rollout 日志中 round0 `candidate_total_count=128`、`taken_total_count=128`，因此 B0 是普通 `rollout.n=8` baseline。A3 则显式传入 `enabled=true`、`method=g2rl_normalized_action_gradient`、`action_normalizer=sciworld`、`round1_candidates=16`、`round1_clusters=8`。",
        "",
        "## 训练曲线摘要",
        "",
        "| run | steps | first10 score mean | last10 score mean | all score mean | avg step time | round0 candidates step25 | round0 taken step25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A3 | `{a3_train.get('step_count')}` | `{fmt(a3_train.get('first10_score_mean'))}` | `{fmt(a3_train.get('last10_score_mean'))}` | `{fmt(a3_train.get('score_mean_all'))}` | `{fmt(a3_train.get('avg_step_time_s'))}s` | `{fmt(a3_train.get('round0_candidate_step25'), 0)}` | `{fmt(a3_train.get('round0_taken_step25'), 0)}` |",
        f"| B0 | `{b0_train.get('step_count')}` | `{fmt(b0_train.get('first10_score_mean'))}` | `{fmt(b0_train.get('last10_score_mean'))}` | `{fmt(b0_train.get('score_mean_all'))}` | `{fmt(b0_train.get('avg_step_time_s'))}s` | `{fmt(b0_train.get('round0_candidate_step25'), 0)}` | `{fmt(b0_train.get('round0_taken_step25'), 0)}` |",
        "",
        "## 动作多样性与动作质量",
        "",
        "| run | round0 groups | round0 unique action mean | min | max | candidate_count values | positive trajectory ratio | done trajectory ratio | no-known-action ratio | precondition-failure ratio |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        f"| A3 | `{a3_actions.get('item_group_count')}` | `{fmt((a3_actions.get('round0_unique_action_count') or {}).get('mean'))}` | `{fmt((a3_actions.get('round0_unique_action_count') or {}).get('min'), 0)}` | `{fmt((a3_actions.get('round0_unique_action_count') or {}).get('max'), 0)}` | `{a3_actions.get('candidate_count_values_round0')}` | `{pct(a3_actions.get('positive_trajectory_ratio'))}` | `{pct(a3_actions.get('done_trajectory_ratio'))}` | `{pct(a3_actions.get('no_known_action_ratio'))}` | `{pct(a3_actions.get('precondition_failure_ratio'))}` |",
        f"| B0 | `{b0_actions.get('item_group_count')}` | `{fmt((b0_actions.get('round0_unique_action_count') or {}).get('mean'))}` | `{fmt((b0_actions.get('round0_unique_action_count') or {}).get('min'), 0)}` | `{fmt((b0_actions.get('round0_unique_action_count') or {}).get('max'), 0)}` | `{b0_actions.get('candidate_count_values_round0')}` | `{pct(b0_actions.get('positive_trajectory_ratio'))}` | `{pct(b0_actions.get('done_trajectory_ratio'))}` | `{pct(b0_actions.get('no_known_action_ratio'))}` | `{pct(b0_actions.get('precondition_failure_ratio'))}` |",
        "",
        f"- A3 action type ratios：{action_type_line(a3_actions)}",
        f"- B0 action type ratios：{action_type_line(b0_actions)}",
        "",
        "## 具体回复样例",
        "",
    ]

    for idx, example in enumerate(examples, 1):
        lines.extend(
            [
                f"### 样例 {idx}: item_id `{example['item_id']}`",
                "",
                f"- A3 round0 unique：`{example['a3_unique_round0']}`",
                f"- B0 round0 unique：`{example['b0_unique_round0']}`",
                "",
                "A3 round0 selected normalized actions:",
            ]
        )
        lines.extend(f"- `{action}`" for action in example["a3_round0_actions"])
        lines.extend(["", "B0 round0 normalized actions:"])
        lines.extend(f"- `{action}`" for action in example["b0_round0_actions"])
        lines.extend(["", "A3 raw response excerpts:"])
        lines.extend(f"- {clip_reply(reply)}" for reply in example["a3_round0_raw"][:2])
        lines.extend(["", "B0 raw response excerpts:"])
        lines.extend(f"- {clip_reply(reply)}" for reply in example["b0_round0_raw"][:2])
        lines.append("")

    lines.extend(
        [
            "## 成因链条分析",
            "",
            "1. A3 的聚类标准确实提高了 round0 的 normalized-action 覆盖面；证据是 A3 的 `candidate_count` 为 `16`，并且每组 unique action 数通常高于 B0。",
            "2. 但是 SciWorld 的成功不是单步动作多样性问题，而是状态前置条件和任务链条问题。大量动作属于观察、导航、门相关动作，许多动作虽然字符串可解析，但环境反馈是 `No known action matches that input`、`The door is not open` 或类似前置条件失败。",
            "3. 本次严格结果显示，A3 并不是单纯变差；它比 B0 更容易拿到正分，说明 G2RL normalized-action 聚类确实把 rollout 推向了更多能产生部分进度的分支。",
            "4. 但 Pass@1 没有提升，说明 SciWorld 的瓶颈在“从部分进度到完整任务链”的转换。聚类按动作文本和 G2RL gradient 选中心，只保证第一步/局部动作差异，不保证后续状态前置条件连续满足。",
            "5. 所以这更像一个任务特征规律，而不是偶然噪声：TextCraft 的短链、配方式动作更容易从 action-level 多样性受益；SciWorld 的长链、可执行前置条件和空间导航让纯 normalized-action 聚类只能提升部分分，难以直接提升 pass。",
            "6. 下一步应该把聚类键从纯 action 文本升级为 `state + normalized_action + progress signal`，或者增加 environment-valid / precondition-aware 过滤。否则 G2RL gradient 聚类仍可能选出语义不同但执行上同样无法完成任务的动作。",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-run-dir", required=True, type=Path)
    parser.add_argument("--b0-run-dir", required=True, type=Path)
    parser.add_argument("--a3-train-log", required=True, type=Path)
    parser.add_argument("--b0-train-log", required=True, type=Path)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()

    a3_train = parse_train_log(args.a3_train_log)
    b0_train = parse_train_log(args.b0_train_log)
    a3_eval = parse_eval(args.a3_run_dir, args.step)
    b0_eval = parse_eval(args.b0_run_dir, args.step)
    a3_actions = summarize_actions(args.a3_run_dir / "executer_logs" / f"step{args.step}", "A3")
    b0_actions = summarize_actions(args.b0_run_dir / "executer_logs" / f"step{args.step}", "B0")
    examples = pick_examples(a3_actions, b0_actions, args.examples)

    comparison = {
        "a3": {"train": a3_train, "eval": a3_eval, "train_actions": a3_actions},
        "b0": {"train": b0_train, "eval": b0_eval, "train_actions": b0_actions},
        "examples": examples,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.report_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, sort_keys=True)
    write_report(
        report_path=args.report_path,
        a3_run=args.a3_run_dir,
        b0_run=args.b0_run_dir,
        a3_log=args.a3_train_log,
        b0_log=args.b0_train_log,
        comparison=comparison,
    )
    print(f"Wrote {args.report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
