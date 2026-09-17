"""Command-line interface for MoTRUST workflows."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

from . import __version__


MODULE_COMMANDS = {
    "prepare-data": ("motrust.data.prepare", "Convert processed inputs and prepare task metadata"),
    "prepare-semantic": ("motrust.workflows.integration.semantic", "Build the shared-feature semantic anchor"),
    "train-integration": ("motrust.workflows.integration.trainer", "Train the multimodal representation model"),
    "build-spectral": ("motrust.workflows.integration.spectral", "Build the spectral representation"),
    "finalize-candidate": ("motrust.workflows.integration.candidate", "Route and finalize candidate coordinates"),
    "compose": ("motrust.workflows.integration.compose", "Compose integration inputs and reference coordinates"),
    "evaluate-integration": ("motrust.benchmark.evaluate", "Evaluate an integrated representation"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="motrust", description="Mosaic multi-omics integration and molecular recovery.")
    result.add_argument("--version", action="version", version=f"MoTRUST {__version__}")
    result.add_argument("--workdir", type=Path, help="Data/output directory (default: MOTRUST_WORKDIR or current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    integration = commands.add_parser("integrate", help="Integrate a prepared multimodal task")
    integration.add_argument("--task-id", required=True)
    integration.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    integration.add_argument("--no-evaluate", action="store_true", help="Produce coordinates without evaluation labels")
    integration.add_argument("--force", action="store_true", help="Replace an existing integration result when --run is used")
    diffusion = commands.add_parser("rna-diffusion", help="Prepare inputs, fit point models, or evaluate RNA samples")
    diffusion.add_argument("--stage", choices=("prepare", "points", "evaluate"), required=True)
    recover = commands.add_parser("recover", help="Fit the three-member ATAC-to-RNA point recovery ensemble")
    for command in (integration, diffusion, recover):
        command.add_argument("--workdir", type=Path, default=argparse.SUPPRESS)
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--check", action="store_true", help="Check required inputs only (default)")
        mode.add_argument("--run", action="store_true", help="Execute after the input check succeeds")
    for name, (_, description) in MODULE_COMMANDS.items():
        command = commands.add_parser(name, help=description, add_help=False)
        command.add_argument("--workdir", type=Path, default=argparse.SUPPRESS)
    return result


def _module_main(name: str, args: list[str]) -> int:
    module = importlib.import_module(name)
    previous = sys.argv
    try:
        sys.argv = ["motrust " + name.rsplit(".", 1)[-1]] + args
        result = module.main()
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous


def main(argv: list[str] | None = None) -> int:
    command_parser = parser()
    args, remaining = command_parser.parse_known_args(argv)
    if args.workdir is not None:
        os.environ["MOTRUST_WORKDIR"] = str(args.workdir.expanduser().resolve())
    if args.command in MODULE_COMMANDS:
        return _module_main(MODULE_COMMANDS[args.command][0], remaining)
    if remaining:
        command_parser.error("unrecognized arguments: " + " ".join(remaining))
    from .preflight import integration_check, rna_check
    if args.command == "integrate":
        issues = integration_check(args.task_id, args.domain, evaluate=not args.no_evaluate)
    else:
        stage = "points" if args.command == "recover" else args.stage
        issues = rna_check(stage)
    if issues:
        print("Required inputs are not ready:", file=sys.stderr)
        for issue in issues[:25]:
            print("- " + issue, file=sys.stderr)
        if len(issues) > 25:
            print(f"- {len(issues) - 25} additional missing inputs.", file=sys.stderr)
        return 2
    if not args.run:
        print("Input check passed. Use --run to execute.")
        return 0
    if args.command == "integrate":
        from .workflows.integration.integrate import run_integration
        output = run_integration(args.task_id, domain=args.domain, evaluate=not args.no_evaluate, force=args.force)
        print(output)
        return 0
    return _module_main("motrust.workflows.rna_diffusion.pipeline", ["--stage", stage])


if __name__ == "__main__":
    raise SystemExit(main())
