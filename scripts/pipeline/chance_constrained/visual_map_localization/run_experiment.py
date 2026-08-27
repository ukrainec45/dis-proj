"""Command-line entry point for the synthetic VML campaign."""

from __future__ import annotations

import argparse
import json

from .experiment import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a spatially sampled synthetic visual-map-localization experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="run two nominal trials before the full campaign")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = run_experiment(args.config, args.output_dir,
                             smoke=args.smoke, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
