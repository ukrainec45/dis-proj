"""Rebuild descriptive analysis from a retained VML trials CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import analyse_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyse retained VML trials")
    parser.add_argument("--trials", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.trials).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trials CSV is empty")
    analysis = analyse_rows(rows)
    Path(args.output).write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
