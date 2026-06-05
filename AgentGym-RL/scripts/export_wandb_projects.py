#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


def clean_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unnamed"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: jsonable(v) for k, v in row.items()} for row in rows])


def run_metadata(run) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "path": list(run.path),
        "project": run.project,
        "entity": run.entity,
        "state": run.state,
        "created_at": str(run.created_at),
        "updated_at": str(getattr(run, "updated_at", "")),
        "url": run.url,
        "group": run.group,
        "job_type": run.job_type,
        "tags": list(run.tags or []),
    }


def export_run(run, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metadata.json", run_metadata(run))
    write_json(out_dir / "config.json", dict(run.config or {}))
    write_json(out_dir / "summary.json", dict(run.summary._json_dict or {}))

    rows = []
    for row in run.scan_history():
        rows.append(dict(row))

    with (out_dir / "history.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    if rows:
        write_rows(out_dir / "history.csv", rows)
    else:
        (out_dir / "history.csv").write_text("")

    file_rows = []
    try:
        for file_obj in run.files():
            file_rows.append(
                {
                    "name": file_obj.name,
                    "size": getattr(file_obj, "size", None),
                    "md5": getattr(file_obj, "md5", None),
                    "url": getattr(file_obj, "url", None),
                }
            )
    except Exception as exc:
        file_rows.append({"error": f"{type(exc).__name__}: {exc}"})
    write_json(out_dir / "files_manifest.json", file_rows)

    metadata = run_metadata(run)
    return {
        "id": metadata["id"],
        "name": metadata["name"],
        "state": metadata["state"],
        "created_at": metadata["created_at"],
        "url": metadata["url"],
        "run_dir": str(out_dir),
        "history_rows": len(rows),
        "file_count": len(file_rows),
    }


def export_project(api, entity: str, project: str, out_root: Path) -> None:
    project_dir = out_root / clean_name(project)
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    runs = list(api.runs(f"{entity}/{project}", per_page=100))
    runs = sorted(runs, key=lambda r: (str(r.created_at), r.id))
    for idx, run in enumerate(runs, start=1):
        run_dir = runs_dir / f"{idx:02d}_{run.id}_{clean_name(run.name)}"
        print(f"[{project}] export {idx}/{len(runs)} {run.id} {run.name}", flush=True)
        index_rows.append(export_run(run, run_dir))

    write_json(project_dir / "runs_index.json", index_rows)
    if index_rows:
        write_rows(project_dir / "runs_index.csv", index_rows)
    readme = [
        f"# W&B export: {entity}/{project}",
        "",
        f"Exported at: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"Run count: {len(index_rows)}",
        "",
        "Each run directory contains metadata.json, config.json, summary.json, history.csv, history.jsonl, and files_manifest.json.",
        "This export is intended for report figures and local reproducibility of scalar curves.",
    ]
    (project_dir / "README.md").write_text("\n".join(readme) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hsch224-peking-university")
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--out-root", default="results/wandb_projects")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    api = wandb.Api(timeout=args.timeout)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for project in args.project:
        export_project(api, args.entity, project, out_root)


if __name__ == "__main__":
    main()
