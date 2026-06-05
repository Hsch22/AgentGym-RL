#!/usr/bin/env python3
"""Summarize TextCraft 1.5B clustering ablation logs and audit CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
TRAIN_START_RE = re.compile(r"\[cluster-100\]\s+train\s+(A\d+)\s+start")
TRAIN_DONE_RE = re.compile(r"\[cluster-100\]\s+train\s+(A\d+)\s+done")

VARIANT_LABELS = {
    "A0": "no_cluster",
    "A1": "random_valid",
    "A2": "gradient_multiview",
    "A3": "g2rl_normalized_action_gradient",
    "A4": "quality_unique_action",
}


@dataclass
class StepMetrics:
    step: int
    values: dict[str, float]
    raw_line: str

    @property
    def is_eval(self) -> bool:
        return any(key.startswith("eval/") for key in self.values)


@dataclass
class VariantLog:
    variant: str
    started: bool = False
    done: bool = False
    rows: list[StepMetrics] = field(default_factory=list)

    @property
    def train_rows(self) -> list[StepMetrics]:
        return [row for row in self.rows if not row.is_eval]

    @property
    def eval_rows(self) -> list[StepMetrics]:
        return [row for row in self.rows if row.is_eval]

    @property
    def last_train(self) -> StepMetrics | None:
        return max(self.train_rows, key=lambda row: row.step, default=None)

    @property
    def last_eval(self) -> StepMetrics | None:
        return max(self.eval_rows, key=lambda row: row.step, default=None)

    @property
    def max_step(self) -> int:
        return max((row.step for row in self.train_rows), default=0)


def strip_console_prefix(line: str) -> str:
    line = ANSI_RE.sub("", line).strip()
    if ") " in line and line.startswith("("):
        line = line.split(") ", 1)[1]
    return line


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_step_line(line: str) -> StepMetrics | None:
    match = STEP_RE.search(line)
    if not match:
        return None
    step = int(match.group(1))
    values: dict[str, float] = {}
    for chunk in match.group(2).split(" - "):
        if ":" not in chunk:
            continue
        key, value = chunk.rsplit(":", 1)
        parsed = parse_float(value)
        if parsed is not None:
            values[key] = parsed
    if not values:
        return None
    return StepMetrics(step=step, values=values, raw_line=line)


def parse_training_log(log_file: Path) -> dict[str, VariantLog]:
    variants: dict[str, VariantLog] = {key: VariantLog(variant=key) for key in VARIANT_LABELS}
    current: str | None = None
    if not log_file.exists():
        return variants

    with log_file.open("r", errors="ignore") as f:
        for raw_line in f:
            line = strip_console_prefix(raw_line)
            start = TRAIN_START_RE.search(line)
            if start:
                current = start.group(1)
                variants.setdefault(current, VariantLog(variant=current)).started = True
                continue
            done = TRAIN_DONE_RE.search(line)
            if done:
                variant = done.group(1)
                variants.setdefault(variant, VariantLog(variant=variant)).done = True
                current = None
                continue
            row = parse_step_line(line)
            if row and current:
                variants.setdefault(current, VariantLog(variant=current)).rows.append(row)

    return variants


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = defaultdict(dict)
    if not path.exists():
        return rows
    with path.open("r", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            variant, key, value = parts
            rows[variant][key] = value
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", errors="ignore") as f:
        return list(csv.DictReader(f))


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def safe_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else float("nan")


def step_number(step: str) -> int:
    return int(re.sub(r"\D+", "", str(step)) or 0)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def metric(row: StepMetrics | None, key: str) -> float | None:
    if row is None:
        return None
    return row.values.get(key)


def best_eval_value(rows: list[StepMetrics], key: str) -> float | None:
    values = [row.values[key] for row in rows if key in row.values and math.isfinite(row.values[key])]
    return max(values) if values else None


def audit_dirs(audit_root: Path, manifest: dict[str, dict[str, str]]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for variant, entries in manifest.items():
        audit_dir = entries.get("audit_dir")
        if audit_dir:
            found[variant] = Path(audit_dir)
    if audit_root.exists():
        for path in sorted(audit_root.iterdir()):
            if not path.is_dir():
                continue
            variant = path.name.split("_", 1)[0]
            if variant.startswith("A"):
                found.setdefault(variant, path)
    return found


def summarize_audit(path: Path) -> dict[str, float | int | str]:
    group_rows = read_csv(path / "group_summary.csv")
    trajectory_rows = read_csv(path / "trajectory_rows.csv")
    pair_rows = read_csv(path / "pairwise_action_similarity.csv")
    if not group_rows and not trajectory_rows:
        return {"status": "missing"}

    latest_step = max((step_number(row.get("step", "")) for row in group_rows), default=0)
    latest_groups = [row for row in group_rows if step_number(row.get("step", "")) == latest_step]
    latest_traj = [row for row in trajectory_rows if step_number(row.get("step", "")) == latest_step]

    def group_avg(key: str, rows: list[dict[str, str]] = group_rows) -> float:
        return safe_mean(safe_float(row, key, float("nan")) for row in rows)

    def traj_avg(key: str, rows: list[dict[str, str]] = trajectory_rows) -> float:
        return safe_mean(safe_float(row, key, float("nan")) for row in rows)

    success_count = sum(int(safe_float(row, "success", 0.0)) for row in trajectory_rows)
    latest_success_count = sum(int(safe_float(row, "success", 0.0)) for row in latest_traj)
    latest_traj_count = len(latest_traj)

    return {
        "status": "ok",
        "groups": len(group_rows),
        "trajectories": len(trajectory_rows),
        "pairs": len(pair_rows),
        "latest_step": latest_step,
        "mean_action_seq_exact": group_avg("mean_action_seq_exact"),
        "mean_action_set_jaccard": group_avg("mean_action_set_jaccard"),
        "mean_roundwise_action_match": group_avg("mean_roundwise_action_match"),
        "mean_later_string_valid_ratio": group_avg("later_string_valid_ratio"),
        "mean_string_valid_ratio": group_avg("string_valid_ratio"),
        "mean_unique_action_sequence_count": group_avg("unique_action_sequence_count"),
        "mean_unique_valid_action_count": group_avg("unique_valid_action_count"),
        "success_trajectories": success_count,
        "success_rate": success_count / len(trajectory_rows) if trajectory_rows else float("nan"),
        "mean_reward": traj_avg("reward"),
        "latest_mean_action_seq_exact": group_avg("mean_action_seq_exact", latest_groups),
        "latest_mean_action_set_jaccard": group_avg("mean_action_set_jaccard", latest_groups),
        "latest_later_string_valid_ratio": group_avg("later_string_valid_ratio", latest_groups),
        "latest_success_trajectories": latest_success_count,
        "latest_success_rate": latest_success_count / latest_traj_count if latest_traj_count else float("nan"),
    }


def compare(audit: dict[str, dict[str, float | int | str]]) -> list[str]:
    lines: list[str] = []
    required = {"A0", "A1", "A2"}
    if not required.issubset(audit) or any(audit[key].get("status") != "ok" for key in required):
        lines.append("- Final comparison is pending until A0/A1/A2 all have audit CSVs.")
        return lines

    a0 = audit["A0"]
    a1 = audit["A1"]
    a2 = audit["A2"]

    def val(variant: dict[str, float | int | str], key: str) -> float:
        raw = variant.get(key, float("nan"))
        return float(raw) if isinstance(raw, (float, int)) else float("nan")

    a0_seq = val(a0, "mean_action_seq_exact")
    a1_seq = val(a1, "mean_action_seq_exact")
    a2_seq = val(a2, "mean_action_seq_exact")
    a0_jac = val(a0, "mean_action_set_jaccard")
    a1_jac = val(a1, "mean_action_set_jaccard")
    a2_jac = val(a2, "mean_action_set_jaccard")
    a1_success = val(a1, "success_rate")
    a2_success = val(a2, "success_rate")

    if a1_seq < a0_seq and a1_jac < a0_jac:
        lines.append("- A1 lowers exact-sequence and action-set similarity versus A0, so extra candidate sampling plus random valid selection increases command diversity.")
    else:
        lines.append("- A1 does not cleanly lower both diversity similarity metrics versus A0; random valid selection is not sufficient evidence by itself.")

    if a2_seq < a1_seq and a2_jac <= a1_jac:
        lines.append("- A2 is more diverse than A1 on the main action-similarity metrics.")
    else:
        lines.append("- A2 does not dominate A1 on action-similarity diversity metrics.")

    if a2_success >= a1_success:
        lines.append("- A2 does not trade away trajectory success rate relative to A1 in the audited sample.")
    else:
        lines.append("- A2 has lower audited trajectory success rate than A1; any diversity gain should be treated as exploratory rather than quality-improving.")
    return lines


def write_summary(
    *,
    output: Path,
    run_ts: str,
    log_file: Path,
    manifest_path: Path,
    audit_root: Path,
    variants: dict[str, VariantLog],
    manifest: dict[str, dict[str, str]],
    audit: dict[str, dict[str, float | int | str]],
) -> None:
    lines: list[str] = []
    lines.append("# TextCraft 1.5B clustering 100-step summary")
    lines.append("")
    lines.append(f"Run timestamp: `{run_ts}`")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- Queue log: `{log_file}`")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Audit root: `{audit_root}`")
    lines.append("")
    lines.append("## Training status")
    lines.append("")
    lines.append(
        "| Variant | Label | Started | Done | Max step | Final train score | Best eval score | KL | Response len | Later clustered | Save dir |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for variant in sorted(VARIANT_LABELS):
        row = variants.get(variant, VariantLog(variant=variant))
        last = row.last_train
        save_dir = manifest.get(variant, {}).get("save_dir", "-")
        lines.append(
            "| {variant} | `{label}` | `{started}` | `{done}` | `{max_step}` | `{score}` | `{eval_score}` | `{kl}` | `{resp}` | `{clustered}` | `{save}` |".format(
                variant=variant,
                label=VARIANT_LABELS[variant],
                started=str(row.started).lower(),
                done=str(row.done).lower(),
                max_step=row.max_step,
                score=fmt(metric(last, "critic/task_score/mean")),
                eval_score=fmt(best_eval_value(row.eval_rows, "eval/task_score/mean")),
                kl=fmt(metric(last, "actor/kl_loss")),
                resp=fmt(metric(last, "response_length/mean")),
                clustered=fmt(metric(last, "rollout/later_clustered_action_ratio")),
                save=save_dir,
            )
        )

    lines.append("")
    lines.append("## Rollout audit")
    lines.append("")
    lines.append(
        "| Variant | Audit status | Groups | Traj | Latest step | Seq exact | Set Jaccard | Later valid | Unique seq | Success rate | Latest success | Audit dir |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    dirs = audit_dirs(audit_root, manifest)
    for variant in sorted(VARIANT_LABELS):
        row = audit.get(variant, {"status": "missing"})
        audit_dir = dirs.get(variant, Path("-"))
        lines.append(
            "| {variant} | `{status}` | `{groups}` | `{traj}` | `{latest}` | `{seq}` | `{jac}` | `{later}` | `{unique}` | `{success}` | `{latest_success}` | `{audit_dir}` |".format(
                variant=variant,
                status=row.get("status", "missing"),
                groups=row.get("groups", "-"),
                traj=row.get("trajectories", "-"),
                latest=row.get("latest_step", "-"),
                seq=fmt(row.get("mean_action_seq_exact") if isinstance(row.get("mean_action_seq_exact"), (int, float)) else None),
                jac=fmt(row.get("mean_action_set_jaccard") if isinstance(row.get("mean_action_set_jaccard"), (int, float)) else None),
                later=fmt(row.get("mean_later_string_valid_ratio") if isinstance(row.get("mean_later_string_valid_ratio"), (int, float)) else None),
                unique=fmt(row.get("mean_unique_action_sequence_count") if isinstance(row.get("mean_unique_action_sequence_count"), (int, float)) else None),
                success=fmt(row.get("success_rate") if isinstance(row.get("success_rate"), (int, float)) else None),
                latest_success=fmt(row.get("latest_success_rate") if isinstance(row.get("latest_success_rate"), (int, float)) else None),
                audit_dir=audit_dir,
            )
        )

    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.extend(compare(audit))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Lower `Seq exact` and `Set Jaccard` mean more action-level diversity.")
    lines.append("- `Later valid` is the post-round-0 parser-valid reply ratio; low values indicate multi-step collapse or invalid command drift.")
    lines.append("- This summary uses rollout-time clustering only. G2RL reward shaping is disabled for A0/A1/A2.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    variants = parse_training_log(args.log_file)
    dirs = audit_dirs(args.audit_root, manifest)
    audit = {variant: summarize_audit(path) for variant, path in dirs.items()}
    write_summary(
        output=args.output,
        run_ts=args.run_ts,
        log_file=args.log_file,
        manifest_path=args.manifest,
        audit_root=args.audit_root,
        variants=variants,
        manifest=manifest,
        audit=audit,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
