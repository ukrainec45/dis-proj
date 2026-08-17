"""CLI entry point for aggregate thesis tables and paired tests."""

import argparse

from .analysis import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate a benchmark result directory")
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    result = analyze(args.run_dir)
    print(f"Analysed {result['rows']} records in {result['analysis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
