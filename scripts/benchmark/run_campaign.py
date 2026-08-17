"""Run the thesis synthetic/real campaign and produce raw + aggregate evidence."""

import argparse
import platform
import shutil
import sys
from pathlib import Path

from .analysis import analyze, write_csv, write_scaling_figure
from .campaign import (VEHICLE_PROFILES, WEATHER_PROFILES, real_aoi_campaign,
                       scale_case, synthetic_campaign)
from .oracle import validate_oracle_suite
from .report import TOPSIS_PROFILES, write_json
from .run_benchmark import _case_manifest, run_cases


def _parse_scales(value):
    try:
        scales = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must be comma-separated integers") from exc
    if not scales or any(scale < 1 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be positive")
    return scales


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the thesis MOA* evaluation campaign")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-variants", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--aoi-manifest", help="five-AOI JSON manifest")
    parser.add_argument("--include-navigation-ablations", action="store_true")
    parser.add_argument("--scales", type=_parse_scales, default=(1,),
                        help="grid scales for the separate scaling study, e.g. 1,2,3")
    args = parser.parse_args(argv)
    if args.synthetic_variants < 0:
        parser.error("--synthetic-variants must be non-negative")

    cases = synthetic_campaign(args.synthetic_variants, args.seed) if args.synthetic_variants else []
    if args.aoi_manifest:
        ablations = ("full", "visual_only", "terrain_only") if args.include_navigation_ablations else ("full",)
        cases.extend(real_aoi_campaign(args.aoi_manifest, ablations=ablations))
    if not cases:
        parser.error("select synthetic variants and/or provide --aoi-manifest")

    output_dir = Path(args.output_dir) / args.run_id
    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"result directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest = {
        "run_id": args.run_id, "python": sys.version, "platform": platform.platform(),
        "seed": args.seed, "synthetic_variants_per_family": args.synthetic_variants,
        "weather_profiles": WEATHER_PROFILES, "vehicle_profiles": VEHICLE_PROFILES,
        "topsis_profiles": TOPSIS_PROFILES, "scales_requested": args.scales,
        "cases": [_case_manifest(case) for case in cases],
    }
    write_json(output_dir / "manifest.json", manifest)
    oracle_rows = validate_oracle_suite()
    write_csv(output_dir / "correctness_oracle.csv", oracle_rows)
    if not all(row["exact_match"] for row in oracle_rows):
        raise RuntimeError("EMOA* failed the exhaustive oracle validation")
    rows = run_cases(cases, output_dir, measure_memory=True)
    analyze(output_dir)
    # Scaling is intentionally separate: it reuses a compact representative subset
    # instead of multiplying the full 100-instance real study.
    if args.scales != (1,):
        representatives = cases[:min(4, len(cases))]
        scaling_dir = output_dir / "scaling"
        scaled = [scale_case(case, factor) for case in representatives for factor in args.scales]
        scaling_rows = run_cases(scaled, scaling_dir, measure_memory=True)
        write_csv(scaling_dir / "scaling_summary.csv", scaling_rows)
        write_scaling_figure(scaling_rows, scaling_dir / "scaling.png")
    print(f"Wrote {len(rows)} method records for {len(cases)} campaign instances to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
