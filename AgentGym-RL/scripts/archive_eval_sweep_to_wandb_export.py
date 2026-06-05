#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def maybe_number(value):
    if value is None or value == "":
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def prefix_metric(key: str) -> str:
    if key in {"global_step", "_step", "_runtime", "_timestamp"}:
        return key
    if key.startswith("eval/"):
        return key
    return f"eval/{key}"


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_indexes(archive_root: Path):
    run_dirs = sorted((archive_root / "runs").glob("*"))
    index_rows = []
    summary_rows = []
    config_rows = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        metadata = json.loads((run_dir / "metadata.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
        history_rows = sum(1 for _ in (run_dir / "history.jsonl").open())
        files_manifest = json.loads((run_dir / "files_manifest.json").read_text())

        index_rows.append(
            {
                "ordinal": idx,
                "id": metadata.get("id"),
                "name": metadata.get("name"),
                "state": metadata.get("state", "finished"),
                "created_at": metadata.get("created_at"),
                "url": metadata.get("url", ""),
                "run_dir": str(run_dir),
                "history_rows": history_rows,
                "files_downloaded": len(files_manifest),
                "files_total": len(files_manifest),
                "summary_eval/avg_at_1": summary.get("eval/avg_at_1"),
                "summary_eval/pass_at_1": summary.get("eval/pass_at_1"),
                "summary_global_step": summary.get("global_step"),
            }
        )
        summary_rows.append({"id": metadata.get("id"), "name": metadata.get("name"), "state": metadata.get("state"), "created_at": metadata.get("created_at"), **summary})
        config_rows.append({"id": metadata.get("id"), "name": metadata.get("name"), "state": metadata.get("state"), "created_at": metadata.get("created_at"), **config})

    (archive_root / "runs_index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True))
    if index_rows:
        write_csv(archive_root / "runs_index.csv", index_rows, list(index_rows[0].keys()))

    if summary_rows:
        fields = sorted({key for row in summary_rows for key in row})
        write_csv(archive_root / "summaries_wide.csv", summary_rows, fields)
    if config_rows:
        fields = sorted({key for row in config_rows for key in row})
        write_csv(archive_root / "configs_wide.csv", config_rows, fields)

    (archive_root / "README.md").write_text(
        f"# W&B-style local export: agentgym-rl-eval\n\nRun count: {len(run_dirs)}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--ordinal", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-data-dir", required=True)
    parser.add_argument("--env-addr", required=True)
    parser.add_argument("--n-gpus", required=True, type=int)
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--results-json", required=True)
    args = parser.parse_args()

    archive_root = Path(args.archive_root).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    runs_root = archive_root / "runs"
    runs_root.mkdir(exist_ok=True)

    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.run_name)
    out_dir = runs_root / f"{args.ordinal:02d}_{args.run_id}_{safe_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "files"
    files_dir.mkdir(exist_ok=True)

    results_csv = Path(args.results_csv).resolve()
    results_json = Path(args.results_json).resolve()
    rows = read_rows(results_csv)
    history_rows = []
    for row in rows:
        history_rows.append({prefix_metric(key): maybe_number(value) for key, value in row.items()})

    fields = sorted({key for row in history_rows for key in row})
    write_csv(out_dir / "history.csv", history_rows, fields)
    with (out_dir / "history.jsonl").open("w") as f:
        for row in history_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = history_rows[-1] if history_rows else {}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    config = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "eval_data_dir": str(Path(args.eval_data_dir).resolve()),
        "env_addr": args.env_addr,
        "n_gpus": args.n_gpus,
        "offline_run_dir": str(Path(args.offline_run_dir).resolve()),
        "checkpoints": [maybe_number(row.get("global_step")) for row in rows],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    metadata = {
        "id": args.run_id,
        "name": args.run_name,
        "state": "finished",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "url": "",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    copied = []
    for source, target_name in ((results_csv, "results.csv"), (results_json, "results.json")):
        target = files_dir / target_name
        shutil.copy2(source, target)
        copied.append({"name": target_name, "path": str(target), "size": target.stat().st_size})
    (out_dir / "files_manifest.json").write_text(json.dumps(copied, indent=2, sort_keys=True))

    rebuild_indexes(archive_root)
    print(out_dir)


if __name__ == "__main__":
    main()
