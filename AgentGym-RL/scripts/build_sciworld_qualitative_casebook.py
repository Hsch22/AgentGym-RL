#!/usr/bin/env python3
"""Build a qualitative SciWorld A3/B0 casebook from finished step100 eval logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_RUNS = {
    1: {
        "source": "seed1_current_continue_from25",
        "A3": Path(
            "results/sciworld/"
            "sciworld_A3_g2rl_normalized_action_gradient_3b_100step_continue_from25_20260530_long100_continue"
        ),
        "B0": Path(
            "results/sciworld/"
            "sciworld_B0_strict_no_cluster_3b_100step_continue_from25_20260530_long100_continue"
        ),
    },
    2: {
        "source": "scratch_multiseed",
        "A3": Path(
            "results/sciworld_multiseed/"
            "sciworld_A3_g2rl_normalized_action_gradient_3b_seed2_100step_20260530_multiseed"
        ),
        "B0": Path(
            "results/sciworld_multiseed/"
            "sciworld_B0_strict_no_cluster_3b_seed2_100step_20260530_multiseed"
        ),
    },
    3: {
        "source": "scratch_multiseed",
        "A3": Path(
            "results/sciworld_multiseed/"
            "sciworld_A3_g2rl_normalized_action_gradient_3b_seed3_100step_20260530_multiseed"
        ),
        "B0": Path(
            "results/sciworld_multiseed/"
            "sciworld_B0_strict_no_cluster_3b_seed3_100step_20260530_multiseed"
        ),
    },
}


ACTION_RE = re.compile(r"action\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
ROOM_RE = re.compile(
    r"\b(?:bedroom|bathroom|kitchen|living room|workshop|greenhouse|art studio|hallway|outside)\b",
    re.IGNORECASE,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def clip(text: Any, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def md_escape(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def extract_parsed_action(raw_response: Any) -> str:
    raw = str(raw_response or "").strip()
    if not raw:
        return ""
    matches = list(ACTION_RE.finditer(raw))
    if not matches:
        return ""
    tail = matches[-1].group(1).strip()
    for line in tail.splitlines():
        line = line.strip()
        if line:
            return line
    return tail


def iter_action_records(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.actions.jsonl")):
        with path.open(errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_file"] = str(path)
                record["_batch"] = path.parent.name
                record["_line_no"] = line_no
                yield record


def extract_task_prompt(conversations: list[dict[str, Any]]) -> str:
    for message in conversations:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "Your task is" in content:
            return content.strip()
    for message in conversations:
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def load_task_prompts(root: Path) -> dict[Any, dict[str, Any]]:
    prompts: dict[Any, dict[str, Any]] = {}
    if not root.exists():
        return prompts
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict) or "item_id" not in row:
                continue
            conversations = row.get("conversations") or []
            if not isinstance(conversations, list):
                conversations = []
            item_id = row.get("item_id")
            prompts[item_id] = {
                "task_prompt": extract_task_prompt(conversations),
                "conversation_count": len(conversations),
                "json_file": str(path),
                "final_conversation_reward": row.get("reward"),
            }
    return prompts


def rooms_in_text(text: str) -> list[str]:
    seen = []
    for match in ROOM_RE.finditer(text):
        room = match.group(0).lower()
        if room not in seen:
            seen.append(room)
    return seen


def instruction_part(task_prompt: str) -> str:
    text = task_prompt.lower()
    marker = "this room is called"
    if marker in text:
        return text.split(marker, 1)[0]
    return text


def task_labels(task_prompt: str, trajectories: list[dict[str, Any]] | None = None) -> list[str]:
    text = task_prompt.lower()
    instruction = instruction_part(task_prompt)
    labels = set()
    if any(token in instruction for token in ("conductive", "conductivity", "electric", "battery", "wire", "light bulb")):
        labels.add("conductivity")
    if any(token in instruction for token in ("temperature", "thermometer", "melting", "melt", "freezing", "freeze")):
        labels.add("temperature")
    if "living thing" in instruction or "non-living" in instruction or "nonliving" in instruction:
        labels.add("living_nonliving")
    if any(token in instruction for token in ("move it to", "move the", "place it", "put it")):
        labels.add("object_relocation")
    if any(token in instruction for token in (" box", "container", "jar", "cup", "closet", "drawer")):
        labels.add("container_manipulation")

    rooms = rooms_in_text(instruction)
    current_match = re.search(r"this room is called the ([a-z ]+?)\.", text)
    current_room = current_match.group(1).strip() if current_match else ""
    target_rooms = [room for room in rooms if room != current_room]
    if target_rooms:
        labels.add("room_navigation")

    if not labels:
        labels.add("other")
    return sorted(labels)


def action_family(action: Any) -> str:
    text = str(action or "").lower().strip()
    if not text:
        return "empty"
    if text.startswith(("go to", "open door", "close door")) or " door" in text:
        return "room_navigation"
    if text.startswith(("look", "examine", "read", "inventory", "task")):
        return "observe"
    if text.startswith("focus on"):
        return "focus"
    if text.startswith(("pick up", "take", "grab")):
        return "pick_up"
    if text.startswith(("move ", "put down", "drop", "place")):
        return "container_manipulation"
    if text.startswith(("use ", "connect ", "disconnect ", "activate", "deactivate", "pour", "mix", "dunk")):
        return "tool_or_science_operation"
    if text.startswith("wait"):
        return "wait"
    return "other"


def make_round(record: dict[str, Any]) -> dict[str, Any]:
    raw_response = str(record.get("raw_response") or "")
    parsed = extract_parsed_action(raw_response)
    normalized = str(record.get("normalized_action") or "").strip()
    return {
        "round": int(record.get("round", -1)),
        "raw_response": raw_response,
        "parsed_action": parsed,
        "normalized_action": normalized,
        "action_family": action_family(normalized),
        "reward": safe_float(record.get("reward")),
        "done": bool(record.get("done")),
        "string_valid": bool(record.get("string_valid")),
        "env_valid": bool(record.get("env_valid")),
        "env_action_type": record.get("env_action_type"),
        "env_action_error": record.get("env_action_error"),
        "env_observation": str(record.get("env_observation") or ""),
        "candidate_count": int(record.get("candidate_count") or 0),
        "candidate_string_valid_count": int(record.get("candidate_string_valid_count") or 0),
        "raw_candidate_index": record.get("raw_candidate_index"),
        "rank": record.get("rank"),
        "trajectory_index": record.get("trajectory_index"),
    }


def load_eval_trajectories(run_dir: Path, step: int) -> dict[Any, dict[str, Any]]:
    root = run_dir / "eval_sciworld_ckpt_sweep" / f"global_step_{step}" / "executer_logs"
    prompts = load_task_prompts(root)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in iter_action_records(root) or []:
        key = (
            record.get("_batch"),
            record.get("_file"),
            record.get("rank"),
            record.get("trajectory_index"),
            record.get("item_id"),
        )
        grouped[key].append(record)

    by_item: dict[Any, dict[str, Any]] = {}
    duplicates: Counter[Any] = Counter()
    for key, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        records.sort(key=lambda row: int(row.get("round", -1)))
        item_id = key[-1]
        rounds = [make_round(record) for record in records]
        rewards = [row["reward"] for row in rounds]
        task_prompt = prompts.get(item_id, {}).get("task_prompt", "")
        trajectory = {
            "item_id": item_id,
            "task_prompt": task_prompt,
            "task_labels": task_labels(task_prompt),
            "rounds": rounds,
            "reward_progression": rewards,
            "final_reward": rewards[-1] if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "done": any(row["done"] for row in rounds),
            "round_count": len(rounds),
            "source_key": key,
            "task_prompt_meta": prompts.get(item_id, {}),
        }
        if item_id in by_item:
            duplicates[item_id] += 1
            if trajectory["final_reward"] <= by_item[item_id]["final_reward"]:
                continue
        by_item[item_id] = trajectory
    if duplicates:
        for item_id, count in duplicates.items():
            by_item[item_id]["duplicate_trajectory_count"] = count + 1
    return by_item


def load_a3_train_clusters(run_dir: Path, step: int) -> dict[Any, dict[str, Any]]:
    root = run_dir / "executer_logs" / f"step{step}"
    prompts = load_task_prompts(root)
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_action_records(root) or []:
        if int(record.get("round", -1)) == 0:
            groups[record.get("item_id")].append(record)

    out: dict[Any, dict[str, Any]] = {}
    for item_id, records in groups.items():
        records.sort(key=lambda row: (int(row.get("trajectory_index", 0) or 0), int(row.get("raw_candidate_index", 0) or 0)))
        selected = [make_round(record) for record in records]
        task_prompt = prompts.get(item_id, {}).get("task_prompt", "")
        unique_actions = sorted({row["normalized_action"] for row in selected if row["normalized_action"]})
        out[item_id] = {
            "item_id": item_id,
            "task_prompt": task_prompt,
            "task_labels": task_labels(task_prompt),
            "match_type": "training_item",
            "candidate_count_values": sorted({row["candidate_count"] for row in selected}),
            "candidate_string_valid_count_values": sorted({row["candidate_string_valid_count"] for row in selected}),
            "selected_count": len(selected),
            "unique_selected_action_count": len(unique_actions),
            "unique_selected_actions": unique_actions,
            "selected_round0_records": selected,
        }
    return out


def choose_cluster_snapshot(
    item_id: Any,
    labels: list[str],
    clusters: dict[Any, dict[str, Any]],
) -> dict[str, Any] | None:
    if item_id in clusters:
        snapshot = dict(clusters[item_id])
        snapshot["match_type"] = "exact_eval_item_overlap"
        return snapshot
    label_set = set(labels)
    scored = []
    for cluster_item_id, snapshot in clusters.items():
        overlap = len(label_set & set(snapshot.get("task_labels") or []))
        scored.append((overlap, snapshot.get("unique_selected_action_count", 0), str(cluster_item_id), snapshot))
    scored.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))
    if not scored:
        return None
    match_type = "same_task_type_representative" if scored[0][0] > 0 else "same_seed_representative"
    snapshot = dict(scored[0][3])
    snapshot["match_type"] = match_type
    snapshot["matched_label_count"] = scored[0][0]
    return snapshot


def pair_seed(seed: int, source: str, a3: dict[Any, dict[str, Any]], b0: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for item_id in sorted(set(a3) & set(b0), key=lambda value: str(value)):
        a3_traj = a3[item_id]
        b0_traj = b0[item_id]
        prompt = a3_traj.get("task_prompt") or b0_traj.get("task_prompt") or ""
        labels = task_labels(prompt, [a3_traj, b0_traj])
        diff = a3_traj["final_reward"] - b0_traj["final_reward"]
        a3_pass = a3_traj["final_reward"] >= 100.0
        b0_pass = b0_traj["final_reward"] >= 100.0
        pairs.append(
            {
                "seed": seed,
                "source": source,
                "item_id": item_id,
                "task_prompt": prompt,
                "task_labels": labels,
                "a3_final_reward": a3_traj["final_reward"],
                "b0_final_reward": b0_traj["final_reward"],
                "a3_max_reward": a3_traj["max_reward"],
                "b0_max_reward": b0_traj["max_reward"],
                "a3_done": a3_traj["done"],
                "b0_done": b0_traj["done"],
                "a3_pass": a3_pass,
                "b0_pass": b0_pass,
                "diff_a3_minus_b0": diff,
                "a3": a3_traj,
                "b0": b0_traj,
            }
        )
    return pairs


def add_case(selected: dict[tuple[int, Any], dict[str, Any]], pair: dict[str, Any], category: str) -> None:
    key = (int(pair["seed"]), pair["item_id"])
    if key in selected:
        if category not in selected[key]["case_categories"]:
            selected[key]["case_categories"].append(category)
        return
    case = dict(pair)
    case["case_categories"] = [category]
    selected[key] = case


def pick_cases(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[int, Any], dict[str, Any]] = {}

    def pick(category: str, candidates: list[dict[str, Any]], limit: int) -> None:
        added = 0
        for pair in candidates:
            key = (int(pair["seed"]), pair["item_id"])
            was_new = key not in selected
            add_case(selected, pair, category)
            if was_new:
                added += 1
            if added >= limit:
                break
        if added == 0 and candidates:
            add_case(selected, candidates[0], category)

    seed1_b0_big = sorted(
        [row for row in pairs if row["seed"] == 1 and row["diff_a3_minus_b0"] < 0],
        key=lambda row: (row["diff_a3_minus_b0"], row["a3_final_reward"]),
    )
    pick("seed1_B0_big_win_A3_loss", seed1_b0_big, 2)

    for seed in (2, 3):
        a3_big = sorted(
            [row for row in pairs if row["seed"] == seed and row["diff_a3_minus_b0"] > 0],
            key=lambda row: (-row["diff_a3_minus_b0"], row["b0_final_reward"]),
        )
        pick(f"seed{seed}_A3_big_win_B0_loss", a3_big, 2)

    both_success = sorted(
        [row for row in pairs if row["a3_pass"] and row["b0_pass"]],
        key=lambda row: (abs(row["diff_a3_minus_b0"]), -row["a3_final_reward"] - row["b0_final_reward"]),
    )
    pick("A3_B0_both_success", both_success, 2)

    both_fail = sorted(
        [row for row in pairs if not row["a3_pass"] and not row["b0_pass"]],
        key=lambda row: (max(row["a3_final_reward"], row["b0_final_reward"]), abs(row["diff_a3_minus_b0"])),
    )
    pick("A3_B0_both_fail", both_fail, 2)

    a3_pass_only = sorted(
        [row for row in pairs if row["a3_pass"] and not row["b0_pass"]],
        key=lambda row: (-row["diff_a3_minus_b0"], row["b0_final_reward"]),
    )
    pick("A3_pass_B0_not_pass", a3_pass_only, 2)

    b0_pass_only = sorted(
        [row for row in pairs if row["b0_pass"] and not row["a3_pass"]],
        key=lambda row: (row["diff_a3_minus_b0"], row["a3_final_reward"]),
    )
    pick("B0_pass_A3_not_pass", b0_pass_only, 2)

    return sorted(selected.values(), key=lambda row: (min(row["case_categories"]), row["seed"], str(row["item_id"])))


def category_availability(pairs: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {
        "seed1_B0_big_win_A3_loss": lambda row: row["seed"] == 1 and row["diff_a3_minus_b0"] < 0,
        "seed2_A3_big_win_B0_loss": lambda row: row["seed"] == 2 and row["diff_a3_minus_b0"] > 0,
        "seed3_A3_big_win_B0_loss": lambda row: row["seed"] == 3 and row["diff_a3_minus_b0"] > 0,
        "A3_B0_both_success": lambda row: row["a3_pass"] and row["b0_pass"],
        "A3_B0_both_fail": lambda row: not row["a3_pass"] and not row["b0_pass"],
        "A3_pass_B0_not_pass": lambda row: row["a3_pass"] and not row["b0_pass"],
        "B0_pass_A3_not_pass": lambda row: row["b0_pass"] and not row["a3_pass"],
    }
    sampled = Counter(category for case in cases for category in case["case_categories"])
    out = []
    for category, predicate in categories.items():
        available = [row for row in pairs if predicate(row)]
        out.append(
            {
                "category": category,
                "available_items": len(available),
                "sampled_cases": sampled.get(category, 0),
                "note": "no paired eval item satisfies this category" if not available else "",
            }
        )
    return out


def summarize_task_types(pairs: list[dict[str, Any]], seed_filter: set[int] | None = None) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if seed_filter is not None and row["seed"] not in seed_filter:
            continue
        for label in row["task_labels"]:
            buckets[label].append(row)

    out = []
    for label, rows in buckets.items():
        diffs = [row["diff_a3_minus_b0"] for row in rows]
        out.append(
            {
                "task_type": label,
                "count": len(rows),
                "mean_a3_minus_b0": mean(diffs) if diffs else None,
                "std_a3_minus_b0": pstdev(diffs) if len(diffs) > 1 else None,
                "a3_wins": sum(diff > 0 for diff in diffs),
                "b0_wins": sum(diff < 0 for diff in diffs),
                "ties": sum(diff == 0 for diff in diffs),
                "a3_done": sum(row["a3_done"] for row in rows),
                "b0_done": sum(row["b0_done"] for row in rows),
                "a3_pass": sum(row["a3_pass"] for row in rows),
                "b0_pass": sum(row["b0_pass"] for row in rows),
                "mean_a3_final": mean(row["a3_final_reward"] for row in rows),
                "mean_b0_final": mean(row["b0_final_reward"] for row in rows),
            }
        )
    out.sort(key=lambda row: (-row["count"], row["task_type"]))
    return out


def case_summary_row(case: dict[str, Any]) -> str:
    return (
        f"| {', '.join(case['case_categories'])} | `{case['seed']}` | `{case['item_id']}` | "
        f"`{fmt(case['a3_final_reward'])}` | `{fmt(case['b0_final_reward'])}` | "
        f"`{signed(case['diff_a3_minus_b0'])}` | `{case['a3_done']}` | `{case['b0_done']}` | "
        f"`{case['a3_pass']}` | `{case['b0_pass']}` | {', '.join(f'`{label}`' for label in case['task_labels'])} |"
    )


def write_task_type_table(lines: list[str], title: str, rows: list[dict[str, Any]], limit: int = 12) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| task type | items | mean A3-B0 | std | A3 wins | B0 wins | ties | A3 pass | B0 pass | A3 done | B0 done | mean A3 final | mean B0 final |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows[:limit]:
        lines.append(
            f"| `{row['task_type']}` | `{row['count']}` | `{signed(row['mean_a3_minus_b0'])}` | "
            f"`{fmt(row['std_a3_minus_b0'])}` | `{row['a3_wins']}` | `{row['b0_wins']}` | "
            f"`{row['ties']}` | `{row['a3_pass']}` | `{row['b0_pass']}` | `{row['a3_done']}` | `{row['b0_done']}` | "
            f"`{fmt(row['mean_a3_final'])}` | `{fmt(row['mean_b0_final'])}` |"
        )
    lines.append("")


def write_availability_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "## 覆盖审计",
            "",
            "| requested slice | available paired items | sampled cases | note |",
            "|---|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['category']}` | `{row['available_items']}` | `{row['sampled_cases']}` | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "说明：`A3_B0_both_success` 使用 `final_reward >= 100` 定义 pass，而不是 `done=True`。SciWorld 日志中 `done=True` 也可能对应 `-100` 的失败终止，因此不能把 done 直接当成成功。",
            "",
        ]
    )


def write_trajectory_table(lines: list[str], title: str, trajectory: dict[str, Any], max_raw_chars: int) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            f"- reward progression: `{[round(value, 3) for value in trajectory['reward_progression']]}`",
            f"- done: `{trajectory['done']}`; final reward: `{fmt(trajectory['final_reward'])}`; max reward: `{fmt(trajectory['max_reward'])}`",
            "",
            "| r | reward | done | parsed action | normalized action | raw reply excerpt | observation excerpt |",
            "|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in trajectory["rounds"]:
        lines.append(
            f"| {row['round']} | `{fmt(row['reward'])}` | `{row['done']}` | "
            f"`{md_escape(row['parsed_action'])}` | `{md_escape(row['normalized_action'])}` | "
            f"{md_escape(clip(row['raw_response'], max_raw_chars))} | "
            f"{md_escape(clip(row['env_observation'], 180))} |"
        )
    lines.append("")


def write_cluster_snapshot(lines: list[str], snapshot: dict[str, Any] | None, max_raw_chars: int) -> None:
    lines.extend(["### A3 第一轮聚类选择结果", ""])
    if not snapshot:
        lines.extend(
            [
                "未找到对应 seed 的 A3 训练 step100 聚类日志；本 case 只能展示 eval 单候选轨迹。",
                "",
            ]
        )
        return
    lines.extend(
        [
            f"- match type: `{snapshot.get('match_type')}`",
            f"- source training item_id: `{snapshot.get('item_id')}`",
            f"- source task labels: {', '.join(f'`{label}`' for label in snapshot.get('task_labels', []))}",
            f"- candidate_count values: `{snapshot.get('candidate_count_values')}`; selected centers: `{snapshot.get('selected_count')}`; unique selected normalized actions: `{snapshot.get('unique_selected_action_count')}`",
            "",
            "Selected round0 center actions:",
            "",
            "| center | raw candidate idx | reward | normalized action | raw reply excerpt |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for idx, row in enumerate(snapshot.get("selected_round0_records", [])[:8], 1):
        lines.append(
            f"| {idx} | `{row.get('raw_candidate_index')}` | `{fmt(row.get('reward'))}` | "
            f"`{md_escape(row.get('normalized_action'))}` | {md_escape(clip(row.get('raw_response'), max_raw_chars))} |"
        )
    lines.append("")


def write_report(report_path: Path, data: dict[str, Any], max_raw_chars: int) -> None:
    cases = data["cases"]
    all_task_types = data["task_type_summary"]["all_seeds"]
    scratch_task_types = data["task_type_summary"]["scratch_seed2_3"]
    seed1_task_types = data["task_type_summary"]["seed1_only"]
    step100 = data["step100_summary"]

    lines = [
        "# SciWorld A3/B0 qualitative casebook",
        "",
        "日期：2026-05-31",
        "",
        "## 结论先行",
        "",
        f"- step100 paired items: `{step100['paired_items']}`；A3 wins=`{step100['a3_wins']}`，B0 wins=`{step100['b0_wins']}`，ties=`{step100['ties']}`。",
        f"- mean A3-B0 final reward=`{signed(step100['mean_a3_minus_b0'])}`，std=`{fmt(step100['std_a3_minus_b0'])}`。",
        "- 定性结论：A3 的优势主要来自能在多 seed 中更稳定地产生“可推进状态”的轨迹，尤其是需要先导航、开门、寻找物体、再移动到容器的任务；B0 在 seed1 的反超集中在少数长链高 partial reward 轨迹，seed2/seed3 没有复现。",
        "- 解释边界：eval 阶段是单候选生成，action log 中 `candidate_count=1`，因此 eval 轨迹本身不记录聚类。每个 case 的“A3 第一轮聚类选择结果”来自同 seed 的 A3 训练 step100 round0 聚类中心；若 eval item 与训练 item 不重合，则使用同 seed、同任务类型的代表性聚类快照，并在 `match type` 中标明。",
        "- `parsed action` 是从 raw response 的 `Action:` 后缀派生；原始日志没有单独保存未归一化 parsed action 字段。机器可读 JSON 保留完整 raw response，Markdown 表格只显示截断摘录。",
        "",
        "## 抽样覆盖",
        "",
        "| case categories | seed | item_id | A3 final | B0 final | A3-B0 | A3 done | B0 done | A3 pass | B0 pass | task labels |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(case_summary_row(case))
    lines.append("")

    write_availability_table(lines, data["category_availability"])

    lines.extend(
        [
            "## 字段完整性",
            "",
            "每个 case 在机器可读 JSON 中都保存以下字段：",
            "",
            "- `task_prompt`：原始任务描述与初始 observation。",
            "- `a3.rounds[]` / `b0.rounds[]`：逐轮 `raw_response`、派生 `parsed_action`、`normalized_action`、`reward`、`done`、`env_observation`、`string_valid`、`env_valid`。",
            "- `reward_progression`：逐轮 reward 序列。",
            "- `a3_round0_cluster_selection`：同 seed 的 A3 训练 step100 round0 聚类中心选择；含 `candidate_count_values`、`selected_count`、`unique_selected_actions`、每个中心的 raw reply 与 normalized action。",
            "",
        ]
    )

    write_task_type_table(lines, "任务类型分解：全部 seed", all_task_types)
    write_task_type_table(lines, "A3 收益集中区：scratch seed2/3", scratch_task_types)

    b0_focus = sorted(seed1_task_types, key=lambda row: (row["mean_a3_minus_b0"], -row["count"]))
    write_task_type_table(lines, "B0 seed1 反超集中区", b0_focus)

    lines.extend(
        [
            "## 成因链条",
            "",
            "1. A3 只在第一轮多采样，但第一轮动作会改变后续 observation 分布。SciWorld 任务常见链条是先 `open door`/`go to room`，再 `look around`/`look in container`，最后 `pick up`/`move ... to box`。首轮如果卡在错误房间、错误对象或不可执行动作，后续 20 轮会沿着错误状态展开。",
            "2. G2RL normalized-action gradient 聚类不是按表面回复聚类，而是对归一化动作对应的梯度特征选中心。它的作用是让首轮训练样本覆盖不同动作方向，例如导航、观察、聚焦、拾取/移动。当前 eval item 几乎都涉及跨房间和容器目标，因此 `room_navigation` 与 `container_manipulation` 是全局瓶颈；真正区分收益大小的是 conductivity、temperature、living/nonliving 等任务族。",
            "3. 但 SciWorld 的难点不是“动作不够多”本身，而是动作必须满足前置条件。casebook 中大量失败轨迹都出现了门未打开、对象不在当前房间、容器不可见、把科学判断直接写进动作等问题。A3 能改善首轮分布，但如果聚类中心仍是不可执行或只会保守观察，优势会停在 partial reward。",
            "4. seed1 的 B0 反超更像训练轨迹敏感性：普通采样偶然保留了少数能深入长链的高回报分支，PPO 后续放大这些分支；seed2/seed3 中这种分支没有稳定复现，A3 的首轮多样性反而更稳。",
            "5. 因此下一版 SciWorld A3 不应只看 `normalized_action + gradient`，还应加入 `state + normalized_action + progress/precondition signal`。尤其要单独处理 room navigation 和 container manipulation：先保证门/房间/容器可达，再鼓励对象操作多样性。",
            "",
        ]
    )

    for idx, case in enumerate(cases, 1):
        lines.extend(
            [
                f"## Case {idx}: seed `{case['seed']}` item `{case['item_id']}`",
                "",
                f"- categories: {', '.join(f'`{cat}`' for cat in case['case_categories'])}",
                f"- task labels: {', '.join(f'`{label}`' for label in case['task_labels'])}",
                f"- outcome: A3 final=`{fmt(case['a3_final_reward'])}`, B0 final=`{fmt(case['b0_final_reward'])}`, A3-B0=`{signed(case['diff_a3_minus_b0'])}`, A3 done=`{case['a3_done']}`, B0 done=`{case['b0_done']}`, A3 pass=`{case['a3_pass']}`, B0 pass=`{case['b0_pass']}`",
                "",
                "Task prompt:",
                "",
                f"> {md_escape(clip(case['task_prompt'], 700))}",
                "",
            ]
        )
        write_cluster_snapshot(lines, case.get("a3_round0_cluster_selection"), max_raw_chars)
        write_trajectory_table(lines, "A3 eval trajectory", case["a3"], max_raw_chars)
        write_trajectory_table(lines, "B0 eval trajectory", case["b0"], max_raw_chars)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    all_pairs: list[dict[str, Any]] = []
    evals: dict[int, dict[str, Any]] = {}
    train_clusters: dict[int, dict[Any, dict[str, Any]]] = {}
    for seed in args.seeds:
        run_cfg = DEFAULT_RUNS[seed]
        a3_eval = load_eval_trajectories(run_cfg["A3"], args.step)
        b0_eval = load_eval_trajectories(run_cfg["B0"], args.step)
        evals[seed] = {"A3": a3_eval, "B0": b0_eval}
        seed_pairs = pair_seed(seed, run_cfg["source"], a3_eval, b0_eval)
        all_pairs.extend(seed_pairs)
        train_clusters[seed] = load_a3_train_clusters(run_cfg["A3"], args.step)

    cases = pick_cases(all_pairs)
    for case in cases:
        snapshot = choose_cluster_snapshot(case["item_id"], case["task_labels"], train_clusters[case["seed"]])
        case["a3_round0_cluster_selection"] = snapshot

    diffs = [row["diff_a3_minus_b0"] for row in all_pairs]
    summary = {
        "paired_items": len(all_pairs),
        "mean_a3_minus_b0": mean(diffs) if diffs else None,
        "std_a3_minus_b0": pstdev(diffs) if len(diffs) > 1 else None,
        "a3_wins": sum(diff > 0 for diff in diffs),
        "b0_wins": sum(diff < 0 for diff in diffs),
        "ties": sum(diff == 0 for diff in diffs),
        "a3_done": sum(row["a3_done"] for row in all_pairs),
        "b0_done": sum(row["b0_done"] for row in all_pairs),
        "a3_pass": sum(row["a3_pass"] for row in all_pairs),
        "b0_pass": sum(row["b0_pass"] for row in all_pairs),
    }

    return {
        "step": args.step,
        "seeds": args.seeds,
        "runs": {
            str(seed): {
                "source": DEFAULT_RUNS[seed]["source"],
                "A3": str(DEFAULT_RUNS[seed]["A3"]),
                "B0": str(DEFAULT_RUNS[seed]["B0"]),
            }
            for seed in args.seeds
        },
        "step100_summary": summary,
        "task_type_summary": {
            "all_seeds": summarize_task_types(all_pairs),
            "scratch_seed2_3": summarize_task_types(all_pairs, {2, 3}),
            "seed1_only": summarize_task_types(all_pairs, {1}),
        },
        "cases": cases,
        "category_availability": category_availability(all_pairs, cases),
        "cluster_snapshot_note": (
            "Eval logs use candidate_count=1, so cluster selections are taken from A3 training step100 "
            "round0 logs. match_type tells whether the training item is an exact item overlap or a same-task-type representative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--report-path", type=Path, default=Path("docs/sciworld_qualitative_casebook_20260531.md"))
    parser.add_argument("--json-path", type=Path, default=Path("docs/sciworld_qualitative_casebook_20260531.json"))
    parser.add_argument("--max-raw-chars", type=int, default=220)
    args = parser.parse_args()

    data = build(args)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_report(args.report_path, data, args.max_raw_chars)
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.json_path}")


if __name__ == "__main__":
    main()
