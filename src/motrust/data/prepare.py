"""Convert processed R matrices and metadata into portable integration task inputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from motrust.paths import resource_path, workdir
from .integration import TaskData


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "metadata"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--scenario")
    args = parser.parse_args(argv)
    if args.stage == "metadata" and not args.task_id:
        parser.error("metadata requires --task-id")
    if args.stage == "metadata" and args.scenario:
        parser.error("--scenario applies to cache preparation; metadata uses --task-id")
    rscript = os.environ.get("RSCRIPT", "Rscript")
    executable = shutil.which(rscript) or (rscript if Path(rscript).is_file() else None)
    if executable is None:
        parser.error("Rscript is unavailable; set RSCRIPT or add it to PATH")
    script = resource_path("data", "build_cache.R" if args.stage == "cache" else "prepare_metadata.R")
    command = [executable, "--vanilla", str(script), "--workdir", str(workdir()),
               "--protocol", str(args.protocol or resource_path("integration", "protocol.json"))]
    if args.data_dir:
        command.extend(["--data-dir", str(args.data_dir.resolve())])
    if args.stage == "metadata":
        command.extend(["--tasks", str(args.tasks or TaskData().definition_path), "--task-id", args.task_id])
    if args.scenario:
        command.extend(["--scenario", args.scenario])
    subprocess.run(command, check=True)
    if args.stage == "cache" and args.scenario:
        cache = workdir() / "data/cache"
        partial = json.loads((cache / f"cache_index_{args.scenario}.json").read_text(encoding="utf-8"))
        full_path = cache / "cache_index.json"
        combined = json.loads(full_path.read_text(encoding="utf-8")) if full_path.is_file() else partial
        combined["scenarios"].update(partial["scenarios"])
        full_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
