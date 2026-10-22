"""Resumable 20-frame AnyVisLoc smoke campaign (run real data in Colab only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .anyvisloc import load_reference, load_sample
from .planar import covariance_extent
from .smoke_localizer import localize


SCHEMA_VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_trial(path, expected=None):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if fingerprint(envelope["trial"]) != envelope["sha256"]:
        raise ValueError(f"Trial checksum mismatch: {path}")
    if expected is not None and envelope["trial"]["identity"] != expected:
        raise ValueError(f"Trial identity mismatch: {path}")
    return envelope["trial"]


def validate_config(config):
    if config["schema_version"] != SCHEMA_VERSION or config["campaign"] != "smoke":
        raise ValueError("This runner implements schema 1 / smoke only")
    if config["frame_count"] != 20:
        raise ValueError("The smoke protocol requires exactly 20 distinct frames")
    if config["reference_modes"] != ["aerial", "satellite"]:
        raise ValueError("Smoke checks aerial and satellite independently")
    if config["prior"]["source"] != "synthetic_gaussian_from_gt":
        raise ValueError("Unknown prior protocol")
    covariance_extent(config["prior"]["covariance_xy_m2"], config["search_probability_mass"])
    for key in ("rectified_gsd_m", "max_image_pixels", "maximum_projection_condition",
                "geometric_R_floor_m"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < config["minimum_map_coverage"] <= 1:
        raise ValueError("minimum_map_coverage must be in (0,1]")
    if config["pose_world_frame"] not in ("dataset", "enu", "anyvisloc_ned_attitude"):
        raise ValueError("Declare pose_world_frame")
    p = config["sift"]
    if not 0 < p["ratio_threshold"] < 1 or not 0 < p["minimum_inlier_ratio"] <= 1:
        raise ValueError("Invalid SIFT ratios")
    if p["minimum_inliers"] < 4 or p["minimum_good_matches"] < 4:
        raise ValueError("Homography requires at least four correspondences")


def run_smoke(dataset_root, output_dir, config):
    validate_config(config)
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Keep dataset and output directories separate")
    sources = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    run_identity = fingerprint(dict(config=config, sources=sources,
                                    numpy=np.__version__, opencv=cv2.__version__))
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8"))["identity"] != run_identity:
            raise ValueError("Code/config/environment changed; choose a new output directory")
    else:
        write_json(run_path, dict(schema_version=1, identity=run_identity, config=config,
                                 source_fingerprints=sources, numpy=np.__version__,
                                 opencv=cv2.__version__))
    inventory, selected, all_trials, dataset_files = [], [], [], {}
    cov = np.asarray(config["prior"]["covariance_xy_m2"], float)
    references = sorted(root.rglob("L??_reference.json"))
    if not references:
        raise FileNotFoundError("No Lxx_reference.json found under dataset root")
    for ref_path in references:
        paths = sorted(ref_path.parent.glob("L??_????.npz"))
        if not paths:
            continue
        aerial = load_reference(ref_path, "aerial")
        scene_selected = []
        for path in paths:
            sample = load_sample(path, pose_world_frame=config["pose_world_frame"])
            try:
                agl = float(sample["xyz"][2] - aerial.elevation(sample["xyz"][:2]))
                eligible = 20 <= agl <= 120
                reason = "eligible" if eligible else "agl_outside_20_120_m"
            except ValueError as error:
                agl, eligible, reason = None, False, str(error)
            record = dict(sample=path.relative_to(root).as_posix(), agl_m=agl,
                          altitude_eligible=eligible, reason=reason)
            inventory.append(record)
            if eligible:
                selected.append(record)
                scene_selected.append((path, sample, agl))
            if len(selected) == 20:
                break
        if scene_selected:
            for mode in config["reference_modes"]:
                reference = aerial if mode == "aerial" else load_reference(ref_path, mode)
                files = set(reference.files)
                for _, sample, _ in scene_selected:
                    files.update(sample["files"])
                for p in sorted(files):
                    name = p.relative_to(root).as_posix()
                    if name not in dataset_files:
                        dataset_files[name] = digest(p)
                unit_id = ref_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
                unit = output / "shards" / unit_id / mode / "sift" / "smoke"
                unit.mkdir(parents=True, exist_ok=True)
                write_json(unit / "status.json", dict(status="running", run_identity=run_identity))
                rows = []
                for path, sample, agl in scene_selected:
                    sample_id = path.relative_to(root).as_posix()
                    seed = int(fingerprint(dict(seed=config["seed"], sample=sample_id))[:16], 16)
                    prior = sample["xyz"][:2] + np.random.default_rng(seed).multivariate_normal([0, 0], cov)
                    file_names = [p.relative_to(root).as_posix()
                                  for p in (*reference.files, *sample["files"])]
                    identity = fingerprint(dict(run=run_identity, sample=sample_id, mode=mode,
                                                files={k: dataset_files[k] for k in file_names}))
                    trial_path = unit / f"{path.stem}.json"
                    if trial_path.exists():
                        row = checked_trial(trial_path, identity)
                    else:
                        result = localize(sample["image"], sample["K"], sample["dist"],
                                          sample["rotation"], agl, prior, cov,
                                          config["search_probability_mass"], reference, config)
                        error = (None if result["z"] is None else
                                 (np.asarray(result["z"][:2])-sample["xyz"][:2]).tolist())
                        row = dict(identity=identity, sample=sample_id, mode=mode, agl_m=agl,
                                   prior_source=config["prior"]["source"], prior_mean_xy=prior.tolist(),
                                   prior_covariance_xy_m2=cov.tolist(), result=result,
                                   residual_xy_m=error, outcome_component=None)
                        write_json(trial_path, dict(trial=row, sha256=fingerprint(row)))
                    rows.append(row)
                shard = unit / "trials.json"
                write_json(shard, rows)
                write_json(unit / "status.json", dict(status="completed", run_identity=run_identity,
                                                       trial_count=len(rows), sha256=digest(shard)))
                all_trials.extend(rows)
        if len(selected) == 20:
            break
    write_json(output / "inventory.json", dict(scope="smoke_prefix_until_20_altitude_eligible",
                                               records=inventory, selected=selected))
    summary = {}
    for mode in config["reference_modes"]:
        rows = [r for r in all_trials if r["mode"] == mode]
        accepted = [r for r in rows if r["result"]["status"] == "accepted"]
        errors = [float(np.linalg.norm(r["residual_xy_m"])) for r in accepted]
        summary[mode] = dict(trial_count=len(rows), accepted_count=len(accepted),
                             reject_count=len(rows)-len(accepted),
                             acc_at_m={str(t): sum(e <= t for e in errors)/len(rows) if rows else None
                                       for t in (1, 3, 5, 10, 20)},
                             validation_passed=False)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", dict(
        schema_version=1, campaign="smoke", run_identity=run_identity,
        status="completed" if len(selected) == 20 else "insufficient_frames",
        dataset_fingerprint=fingerprint(dataset_files), dataset_files=dataset_files,
        matcher="sift", matcher_fingerprint=fingerprint(config["sift"]),
        config_fingerprint=fingerprint(config), validation_passed={m: False for m in summary},
        deployment_status="not_fitted_smoke_only", supported_domain=None,
        license_marker=config["license_marker"], target_platform_calibrated=False,
        R_calibrated=False, prior_source=config["prior"]["source"],
        limitations=["No outcome model fitted", "No local/gross assignment",
                     "No cross-scene validation", "Camera attitude is dataset ground truth",
                     "AGL uses aerial DSM and GT position; no altitude uncertainty simulated",
                     "Yaw unavailable; XY geometric R is uncalibrated"]))
    if len(selected) != 20:
        raise ValueError(f"Smoke incomplete: only {len(selected)} altitude-eligible frames")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    print(json.dumps(run_smoke(args.dataset_root, args.output_dir, config), indent=2))


if __name__ == "__main__":
    main()
