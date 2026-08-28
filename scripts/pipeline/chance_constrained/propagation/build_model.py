"""CLI for constructing the empirical INS/OpenVINS baseline artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from .artifacts import atomic_json, atomic_npz, build_manifest, validate_model_arrays
from ..artifacts import source_fingerprint
from .config import load_build_config
from .fitting import fit_records, validate_records
from .run_log import load_run_log
from .segments import classify_modes, extract_segments, write_segments_csv


OUTPUT_NAMES = ("ins_openvins_model.npz", "segments.csv", "summary.json", "manifest.json")


def _quality_checks(records, run_logs, config):
    nominal_coverage = {}
    blackout_classification = {}
    recovery_observed = {}
    passed = True
    for run in config.runs:
        log = run_logs[run.run_id]
        if run.visual_regime == "nominal_20hz":
            coverage = float(log.post_initialization_coverage)
            nominal_coverage[run.run_id] = coverage
            passed = passed and coverage >= .95
        modes = classify_modes(log, config.mode_policy)
        unavailable = np.asarray(log.initialized, dtype=bool) & ~np.asarray(
            log.forced_camera_available, dtype=bool)
        if np.any(unavailable):
            correct = bool(np.all(modes[unavailable] == "ins_only"))
            blackout_classification[run.run_id] = correct
            passed = passed and correct
            after_blackout = np.flatnonzero(
                np.asarray(log.forced_camera_available, dtype=bool)
                & (np.arange(len(modes)) > int(np.flatnonzero(unavailable)[-1])))
            recovery_observed[run.run_id] = bool(
                len(after_blackout)
                and np.any(np.isin(modes[after_blackout], [
                    "reinitialized", "vio_operational", "vio_degraded"])))

    drift_comparisons = []
    for duration in config.durations_s:
        operational = [
            record.horizontal_position_error_m for record in records
            if record.duration_s == duration
            and record.visual_regime == "nominal_20hz"
            and record.start_mode == "vio_operational"
            and record.end_mode == "vio_operational"
        ]
        ins_only = [
            record.horizontal_position_error_m for record in records
            if record.duration_s == duration
            and record.visual_regime == "blackout"
            and record.end_mode == "ins_only"
        ]
        if not operational or not ins_only:
            continue
        operational_median = float(np.median(operational))
        ins_only_median = float(np.median(ins_only))
        comparison_passed = operational_median < ins_only_median
        drift_comparisons.append({
            "duration_s": duration,
            "vio_operational_sample_count": len(operational),
            "ins_only_sample_count": len(ins_only),
            "vio_operational_median_position_error_m": operational_median,
            "ins_only_median_position_error_m": ins_only_median,
            "passed": comparison_passed,
        })
        passed = passed and comparison_passed
    if not drift_comparisons:
        passed = False
    return {
        "passed": bool(passed),
        "nominal_post_initialization_coverage": nominal_coverage,
        "blackout_classified_as_ins_only": blackout_classification,
        "recovery_observed_after_final_blackout": recovery_observed,
        "drift_comparisons": drift_comparisons,
        "reason": (None if passed else
                   "coverage, blackout classification, or measured drift ordering failed"),
    }


def build_propagation_model(config_path, output_dir, *, force=False):
    config = load_build_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing outputs: " + ", ".join(path.name for path in existing))

    records = []
    run_logs = {}
    for run in config.runs:
        log = load_run_log(run.run_log_path)
        run_logs[run.run_id] = log
        records.extend(extract_segments(log, run, config))
    if not records:
        raise ValueError("no valid propagation segments were extracted")

    regimes = tuple(dict.fromkeys(
        list(config.visual_regimes) + [record.visual_regime for record in records]))
    fit_records_only = [record for record in records if record.split == "fit"]
    validation_records = [record for record in records if record.split == "validation"]
    training_model = fit_records(fit_records_only, config, regimes=regimes)
    validation = validate_records(training_model, validation_records, config)
    validation["statistical_passed"] = validation["passed"]
    validation["quality_checks"] = _quality_checks(records, run_logs, config)
    validation["passed"] = bool(
        validation["statistical_passed"] and validation["quality_checks"]["passed"])
    if not validation["passed"]:
        validation["reason"] = (
            "held-out calibration or mandatory propagation quality checks failed")
    final_model = fit_records(records, config, regimes=regimes)
    final_model["model_id"] = np.asarray(config.model_id)
    provenance = config.raw.get("provenance", {})
    final_model["openvins_version"] = np.asarray(
        str(provenance.get("openvins_version", "unspecified")))
    final_model["openvins_commit"] = np.asarray(
        str(provenance.get("openvins_commit", "unspecified")))
    final_model["dataset_ids"] = np.asarray(tuple(dict.fromkeys(
        run.dataset_id for run in config.runs)))
    final_model["configuration_sha256"] = np.asarray(
        source_fingerprint(config.config_path)["sha256"])
    final_model["degradation_mode_ids"] = np.asarray(regimes)
    final_model["baseline_only"] = np.asarray(True)
    final_model["target_platform_calibrated"] = np.asarray(False)
    final_model["validation_passed"] = np.asarray(validation["passed"], dtype=bool)
    validate_model_arrays(final_model)

    summary = {
        "model_id": config.model_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_only": True,
        "target_platform_calibrated": False,
        "segment_count": len(records),
        "segments_by_split": dict(Counter(record.split for record in records)),
        "segments_by_dataset": dict(Counter(record.dataset_id for record in records)),
        "segments_by_regime": dict(Counter(record.visual_regime for record in records)),
        "segments_by_start_mode": dict(Counter(record.start_mode for record in records)),
        "supported_group_count": int(np.count_nonzero(final_model["supported"])),
        "validation": validation,
    }
    manifest = build_manifest(config, validation)

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary_segments = Path(temporary) / "segments.csv"
        write_segments_csv(temporary_segments, records)
        temporary_segments.replace(output / "segments.csv")
    atomic_npz(output / "ins_openvins_model.npz", final_model)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "manifest.json", manifest)
    return final_model, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a versioned empirical INS/OpenVINS propagation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _, summary = build_propagation_model(args.config, args.output_dir,
                                         force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
