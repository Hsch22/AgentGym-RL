#!/usr/bin/env python3
"""Audit TextCraft rollout replies with action-level similarity metrics.

The script reads ``executer_logs/step*/{rank}.actions.jsonl`` files produced by
the local rollout instrumentation.  It does not require model weights; it uses
the parser-normalized TextCraft actions already stored in the action logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Iterable


INVALID = "<INVALID>"
END = "<END>"
EMPTY = "<EMPTY>"


@dataclass
class Trajectory:
    step: str
    rank: int
    trajectory_index: int
    item_id: str
    records: list[dict]
    goal: str = ""

    @property
    def key(self) -> str:
        return f"rank{self.rank}:traj{self.trajectory_index}"

    @property
    def reward(self) -> float:
        rewards = []
        for record in self.records:
            try:
                rewards.append(float(record.get("reward", 0.0)))
            except (TypeError, ValueError):
                rewards.append(0.0)
        return max(rewards) if rewards else 0.0

    @property
    def success(self) -> bool:
        return self.reward > 0.0

    @property
    def action_sequence(self) -> list[str]:
        return [record_action(record) for record in sorted(self.records, key=lambda r: int(r.get("round", 0)))]

    @property
    def valid_actions(self) -> list[str]:
        return [action for action in self.action_sequence if action != INVALID]

    @property
    def string_valid_ratio(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for record in self.records if bool(record.get("string_valid", False))) / len(self.records)

    @property
    def empty_raw_ratio(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for record in self.records if not str(record.get("raw_response", ""))) / len(self.records)

    @property
    def max_raw_chars(self) -> int:
        return max((len(str(record.get("raw_response", ""))) for record in self.records), default=0)


def record_action(record: dict) -> str:
    normalized = str(record.get("normalized_action") or "").strip()
    if bool(record.get("string_valid", False)) and normalized:
        return normalized
    return INVALID


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return mean(values) if values else default


def action_set_jaccard(left: list[str], right: list[str]) -> float:
    left_set = {x for x in left if x != INVALID}
    right_set = {x for x in right if x != INVALID}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def roundwise_match(left: list[str], right: list[str]) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 1.0
    left_pad = left + [END] * (width - len(left))
    right_pad = right + [END] * (width - len(right))
    return sum(1 for lval, rval in zip(left_pad, right_pad) if lval == rval) / width


def pair_metrics(left: Trajectory, right: Trajectory) -> dict:
    left_seq = left.action_sequence
    right_seq = right.action_sequence
    return {
        "step": left.step,
        "item_id": left.item_id,
        "left": left.key,
        "right": right.key,
        "left_reward": left.reward,
        "right_reward": right.reward,
        "left_success": int(left.success),
        "right_success": int(right.success),
        "action_seq_exact": float(left_seq == right_seq),
        "action_set_jaccard": action_set_jaccard(left_seq, right_seq),
        "roundwise_action_match": roundwise_match(left_seq, right_seq),
        "left_action_seq": " | ".join(left_seq),
        "right_action_seq": " | ".join(right_seq),
    }


def extract_goal(conversations: list[dict]) -> str:
    for message in conversations:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        match = re.search(r"Goal:\s*(.+?)(?:\n|$)", content)
        if match:
            return match.group(1).strip()
    return ""


def load_goals(step_dir: Path) -> dict[str, str]:
    goals: dict[str, str] = {}
    for json_path in sorted(step_dir.glob("*.json")):
        if json_path.name.endswith(".actions.jsonl"):
            continue
        try:
            payload = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            item_id = str(row.get("item_id", ""))
            if not item_id or item_id in goals:
                continue
            goal = extract_goal(row.get("conversations") or [])
            if goal:
                goals[item_id] = goal
    return goals


def iter_step_dirs(log_dir: Path, steps: list[str] | None) -> list[Path]:
    if steps:
        names = [step if step.startswith("step") else f"step{step}" for step in steps]
        return [log_dir / name for name in names]
    return sorted(
        [path for path in log_dir.iterdir() if path.is_dir() and path.name.startswith("step")],
        key=lambda path: int(re.sub(r"\D+", "", path.name) or 0),
    )


def load_trajectories(log_dir: Path, steps: list[str] | None) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for step_dir in iter_step_dirs(log_dir, steps):
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing step directory: {step_dir}")
        goals = load_goals(step_dir)
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for action_path in sorted(step_dir.glob("*.actions.jsonl")):
            rank_match = re.match(r"(\d+)\.actions\.jsonl$", action_path.name)
            if not rank_match:
                continue
            rank = int(rank_match.group(1))
            with action_path.open("r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record.setdefault("rank", rank)
                    trajectory_index = int(record.get("trajectory_index", 0))
                    grouped[(rank, trajectory_index)].append(record)

        step_name = step_dir.name
        for (rank, trajectory_index), records in sorted(grouped.items()):
            if not records:
                continue
            records = sorted(records, key=lambda r: int(r.get("round", 0)))
            item_id = str(records[0].get("item_id", ""))
            trajectories.append(
                Trajectory(
                    step=step_name,
                    rank=rank,
                    trajectory_index=trajectory_index,
                    item_id=item_id,
                    records=records,
                    goal=goals.get(item_id, ""),
                )
            )
    return trajectories


def build_tables(trajectories: list[Trajectory]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[(trajectory.step, trajectory.item_id)].append(trajectory)

    pair_rows: list[dict] = []
    group_rows: list[dict] = []
    trajectory_rows: list[dict] = []

    for trajectory in trajectories:
        trajectory_rows.append(
            {
                "step": trajectory.step,
                "rank": trajectory.rank,
                "trajectory_index": trajectory.trajectory_index,
                "trajectory_key": trajectory.key,
                "item_id": trajectory.item_id,
                "goal": trajectory.goal,
                "reward": trajectory.reward,
                "success": int(trajectory.success),
                "rounds": len(trajectory.records),
                "valid_action_count": len(trajectory.valid_actions),
                "unique_valid_action_count": len(set(trajectory.valid_actions)),
                "string_valid_ratio": trajectory.string_valid_ratio,
                "empty_raw_ratio": trajectory.empty_raw_ratio,
                "max_raw_chars": trajectory.max_raw_chars,
                "action_sequence": " | ".join(trajectory.action_sequence),
            }
        )

    for (step, item_id), group in sorted(groups.items(), key=lambda kv: (int(re.sub(r"\D+", "", kv[0][0]) or 0), kv[0][1])):
        pairs = [pair_metrics(left, right) for left, right in combinations(group, 2)]
        pair_rows.extend(pairs)
        action_sequences = [tuple(trajectory.action_sequence) for trajectory in group]
        all_records = [record for trajectory in group for record in trajectory.records]
        later_records = [record for record in all_records if int(record.get("round", 0)) > 0]
        valid_actions = [action for trajectory in group for action in trajectory.valid_actions]
        group_rows.append(
            {
                "step": step,
                "item_id": item_id,
                "goal": group[0].goal if group else "",
                "trajectory_count": len(group),
                "success_count": sum(1 for trajectory in group if trajectory.success),
                "unique_action_sequence_count": len(set(action_sequences)),
                "unique_valid_action_count": len(set(valid_actions)),
                "mean_action_seq_exact": safe_mean(row["action_seq_exact"] for row in pairs),
                "mean_action_set_jaccard": safe_mean(row["action_set_jaccard"] for row in pairs),
                "mean_roundwise_action_match": safe_mean(row["roundwise_action_match"] for row in pairs),
                "string_valid_ratio": safe_mean(bool(record.get("string_valid", False)) for record in all_records),
                "later_string_valid_ratio": safe_mean(bool(record.get("string_valid", False)) for record in later_records),
                "empty_raw_ratio": safe_mean(not str(record.get("raw_response", "")) for record in all_records),
                "max_raw_chars": max((len(str(record.get("raw_response", ""))) for record in all_records), default=0),
                "dominant_sequence_count": Counter(action_sequences).most_common(1)[0][1] if action_sequences else 0,
            }
        )

    return trajectory_rows, pair_rows, group_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.3f}"


def clip_text(text: str, max_chars: int) -> str:
    if text == "":
        return EMPTY
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def select_examples(group_rows: list[dict], max_examples: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, candidates: list[dict]) -> None:
        for row in candidates:
            key = (row["step"], row["item_id"])
            if key in seen:
                continue
            selected.append({"label": label, **row})
            seen.add(key)
            break

    add(
        "后续轮动作有效性坍缩",
        sorted(
            [row for row in group_rows if int(row["trajectory_count"]) >= 4],
            key=lambda row: (float(row["later_string_valid_ratio"]), -float(row["success_count"]), row["step"], row["item_id"]),
        ),
    )
    add(
        "近重复动作计划",
        sorted(
            [row for row in group_rows if int(row["trajectory_count"]) >= 4],
            key=lambda row: (-float(row["mean_action_seq_exact"]), int(row["unique_action_sequence_count"]), row["step"], row["item_id"]),
        ),
    )
    add(
        "成功但多样性较低",
        sorted(
            [row for row in group_rows if int(row["success_count"]) > 0 and int(row["trajectory_count"]) >= 4],
            key=lambda row: (-float(row["mean_action_set_jaccard"]), -float(row["success_count"]), row["step"], row["item_id"]),
        ),
    )
    add(
        "多样化动作计划",
        sorted(
            [row for row in group_rows if int(row["trajectory_count"]) >= 4],
            key=lambda row: (-int(row["unique_action_sequence_count"]), float(row["mean_action_set_jaccard"]), row["step"], row["item_id"]),
        ),
    )
    add(
        "超长原始回复离群样例",
        sorted(group_rows, key=lambda row: (-int(row["max_raw_chars"]), row["step"], row["item_id"])),
    )

    if len(selected) < max_examples:
        for row in sorted(group_rows, key=lambda row: (row["step"], row["item_id"])):
            key = (row["step"], row["item_id"])
            if key not in seen:
                selected.append({"label": "补充覆盖样例", **row})
                seen.add(key)
            if len(selected) >= max_examples:
                break
    return selected[:max_examples]


def write_report(
    path: Path,
    *,
    log_dir: Path,
    trajectories: list[Trajectory],
    trajectory_rows: list[dict],
    pair_rows: list[dict],
    group_rows: list[dict],
    max_examples: int,
    max_trajectories_per_group: int,
    max_rounds_per_trajectory: int,
    max_raw_chars: int,
) -> None:
    groups_by_key: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups_by_key[(trajectory.step, trajectory.item_id)].append(trajectory)

    group_count = len(group_rows)
    pair_count = len(pair_rows)
    traj_count = len(trajectory_rows)
    examples = select_examples(group_rows, max_examples=max_examples)

    lines: list[str] = []
    lines.append("# TextCraft rollout 原始回复与动作相似度审计")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(f"- 源 rollout 日志目录：`{log_dir}`")
    lines.append(f"- 组数：`{group_count}` 个 prompt group")
    lines.append(f"- 轨迹数：`{traj_count}`")
    lines.append(f"- 两两比较数：`{pair_count}`")
    lines.append("")
    lines.append("本审计使用 `*.actions.jsonl` 中的 action-level 证据。它刻意独立于模型空间的 G2RL tensor，因为运行时 G2RL feature 没有持久化到 rollout 日志中。")
    lines.append("")
    lines.append("## 指标")
    lines.append("")
    lines.append("- `action_seq_exact`：只有当两个完整 normalized action 序列完全相同时才为 1.0，包括无效轮次。")
    lines.append("- `action_set_jaccard`：对有效 normalized actions 计算 Jaccard；两个全无效轨迹记为 1.0，因为它们共享相同的“无有效动作”行为。")
    lines.append("- `roundwise_action_match`：较短轨迹用 `<END>` padding 后，逐轮比较 normalized action 是否匹配。")
    lines.append("- `later_string_valid_ratio`：round 0 之后的 string-valid 比例，用于隔离多步退化。")
    lines.append("")
    lines.append("## 汇总发现")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 平均组 `action_seq_exact` | `{fmt_float(safe_mean(float(row['mean_action_seq_exact']) for row in group_rows))}` |")
    lines.append(f"| 平均组 `action_set_jaccard` | `{fmt_float(safe_mean(float(row['mean_action_set_jaccard']) for row in group_rows))}` |")
    lines.append(f"| 平均组 `roundwise_action_match` | `{fmt_float(safe_mean(float(row['mean_roundwise_action_match']) for row in group_rows))}` |")
    lines.append(f"| 平均组 string-valid 比例 | `{fmt_float(safe_mean(float(row['string_valid_ratio']) for row in group_rows))}` |")
    lines.append(f"| 平均组后续轮 string-valid 比例 | `{fmt_float(safe_mean(float(row['later_string_valid_ratio']) for row in group_rows))}` |")
    lines.append(f"| 后续轮有效率 < 0.20 的组数 | `{sum(1 for row in group_rows if float(row['later_string_valid_ratio']) < 0.20)} / {group_count}` |")
    lines.append(f"| 单一主导序列覆盖全部轨迹的组数 | `{sum(1 for row in group_rows if int(row['dominant_sequence_count']) == int(row['trajectory_count']))} / {group_count}` |")
    lines.append(f"| 成功轨迹数 | `{sum(int(row['success']) for row in trajectory_rows)} / {traj_count}` |")
    lines.append("")
    lines.append("## 典型样例")
    lines.append("")

    for example_index, example in enumerate(examples, start=1):
        key = (example["step"], example["item_id"])
        group = sorted(groups_by_key[key], key=lambda trajectory: (trajectory.rank, trajectory.trajectory_index))
        lines.append(f"### 样例 {example_index}：{example['label']}")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| step | `{example['step']}` |")
        lines.append(f"| item_id | `{example['item_id']}` |")
        lines.append(f"| 目标 | `{example.get('goal', '')}` |")
        lines.append(f"| 轨迹数 | `{example['trajectory_count']}` |")
        lines.append(f"| success_count | `{example['success_count']}` |")
        lines.append(f"| 唯一 action 序列数 | `{example['unique_action_sequence_count']}` |")
        lines.append(f"| 唯一有效 action 数 | `{example['unique_valid_action_count']}` |")
        lines.append(f"| 平均 action_seq_exact | `{fmt_float(example['mean_action_seq_exact'])}` |")
        lines.append(f"| 平均 action_set_jaccard | `{fmt_float(example['mean_action_set_jaccard'])}` |")
        lines.append(f"| 平均 roundwise_action_match | `{fmt_float(example['mean_roundwise_action_match'])}` |")
        lines.append(f"| string_valid_ratio | `{fmt_float(example['string_valid_ratio'])}` |")
        lines.append(f"| later_string_valid_ratio | `{fmt_float(example['later_string_valid_ratio'])}` |")
        lines.append(f"| empty_raw_ratio | `{fmt_float(example['empty_raw_ratio'])}` |")
        lines.append(f"| max_raw_chars | `{example['max_raw_chars']}` |")
        lines.append("")
        lines.append("#### 轨迹摘要")
        lines.append("")
        lines.append("| 轨迹 | reward | 轮数 | 有效 actions | string_valid | empty_raw | 最大 raw chars | action 序列 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for trajectory in group[:max_trajectories_per_group]:
            lines.append(
                "| {key} | {reward:.1f} | {rounds} | {valid_count} | {sv} | {empty} | {max_chars} | {seq} |".format(
                    key=trajectory.key,
                    reward=trajectory.reward,
                    rounds=len(trajectory.records),
                    valid_count=len(trajectory.valid_actions),
                    sv=fmt_float(trajectory.string_valid_ratio),
                    empty=fmt_float(trajectory.empty_raw_ratio),
                    max_chars=trajectory.max_raw_chars,
                    seq=" -> ".join(trajectory.action_sequence),
                )
            )
        if len(group) > max_trajectories_per_group:
            lines.append(f"| ... | | | | | | | 省略 `{len(group) - max_trajectories_per_group}` 条更多轨迹 |")
        lines.append("")
        lines.append("#### 原始回复")
        lines.append("")
        for trajectory in group[:max_trajectories_per_group]:
            lines.append(f"##### {trajectory.key} reward={trajectory.reward:.1f}")
            lines.append("")
            for record in trajectory.records[:max_rounds_per_trajectory]:
                round_idx = int(record.get("round", 0))
                raw_response = str(record.get("raw_response", ""))
                normalized = record_action(record)
                env_observation = str(record.get("env_observation", ""))
                lines.append(
                    f"- 第 `{round_idx}` 轮 string_valid=`{bool(record.get('string_valid', False))}` "
                    f"env_valid=`{bool(record.get('env_valid', False))}` normalized_action=`{normalized}`"
                )
                if env_observation:
                    lines.append(f"  env_observation: `{clip_text(env_observation, 240)}`")
                lines.append("")
                lines.append("```text")
                lines.append(clip_text(raw_response, max_raw_chars))
                lines.append("```")
                lines.append("")
            if len(trajectory.records) > max_rounds_per_trajectory:
                lines.append(f"... 此轨迹省略 `{len(trajectory.records) - max_rounds_per_trajectory}` 个更多轮次。")
                lines.append("")
        lines.append("#### 解读")
        lines.append("")
        if float(example["later_string_valid_ratio"]) < 0.20:
            lines.append("后续轮 parser-valid 比例低于 0.20，因此多数 round 0 之后的回复为空或无效。这是策略坍缩信号，不是有用探索。")
        elif int(example["unique_action_sequence_count"]) == 1:
            lines.append("所有轨迹共享相同的 normalized action 序列。任何把这些轨迹判为多样的模型空间指标，都应当在 TextCraft 命令多样性角度被谨慎看待。")
        elif int(example["unique_action_sequence_count"]) > 4:
            lines.append("该组包含许多不同的 normalized action 序列，适合用于检查聚类选择器是否能在不选择无效命令的前提下保留动作多样性。")
        else:
            lines.append("该样例作为审计语料的补充覆盖。")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", nargs="*", default=None, help="Step numbers or step directory names. Defaults to all step* directories.")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-trajectories-per-group", type=int, default=8)
    parser.add_argument("--max-rounds-per-trajectory", type=int, default=12)
    parser.add_argument("--max-raw-chars", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.log_dir, args.steps)
    if not trajectories:
        raise RuntimeError(f"No trajectories found under {args.log_dir}")

    trajectory_rows, pair_rows, group_rows = build_tables(trajectories)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory_rows.csv", trajectory_rows)
    write_csv(args.output_dir / "pairwise_action_similarity.csv", pair_rows)
    write_csv(args.output_dir / "group_summary.csv", group_rows)
    write_report(
        args.output_dir / "audit_report.md",
        log_dir=args.log_dir,
        trajectories=trajectories,
        trajectory_rows=trajectory_rows,
        pair_rows=pair_rows,
        group_rows=group_rows,
        max_examples=args.max_examples,
        max_trajectories_per_group=args.max_trajectories_per_group,
        max_rounds_per_trajectory=args.max_rounds_per_trajectory,
        max_raw_chars=args.max_raw_chars,
    )
    print(f"Wrote {args.output_dir}")
    print(f"trajectories={len(trajectory_rows)} groups={len(group_rows)} pairs={len(pair_rows)}")


if __name__ == "__main__":
    main()
