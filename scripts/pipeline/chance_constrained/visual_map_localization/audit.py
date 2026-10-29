"""Full aerial data audit. Real datasets are processed only in hosted Colab.

No matching is performed by audit_scene. run_control is a separate, opt-in
experiment on the retained Scene_05 coverage rejects. Synthetic fixtures may
exercise these functions locally.
"""
import csv
import hashlib
import json
import math
import os
import tempfile
import zlib
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from shapely.geometry import Polygon

from .anyvisloc import load_reference, load_sample
from .anyvisloc_smoke import (canonical, checked_trial, digest, fingerprint,
                             validate_config, write_json)
from .planar import rectify, search_bounds, search_crop
from .smoke_localizer import localize


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def separate_output(output, *sources):
    output = Path(output).resolve()
    for source in sources:
        if source is None:
            continue
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output and source directories must be disjoint")
    return output


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def inventory_archives(folder, output, budget_bytes=5_000_000_000):
    """Read ZIP metadata and hash sources; do not extract or delete anything."""
    folder = Path(folder).resolve()
    output = separate_output(output, folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows = []
    for path in sorted(folder.glob("*.zip")):
        row = dict(archive=str(path), archive_bytes=path.stat().st_size,
                   source_available=True, fully_verified=False)
        try:
            row["sha256"] = digest(path)
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                row.update(unpacked_bytes=sum(x.file_size for x in members),
                           scene_ids=sorted({PurePosixPath(x.filename).parts[-2]
                               for x in members if x.filename.endswith("_reference.json")
                               and len(PurePosixPath(x.filename).parts) > 1}),
                           status="discovered_not_audited")
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            row.update(status="archive_error", error=str(error))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, archives=rows, budget_bytes=budget_bytes,
                  archive_bytes=sum(x["archive_bytes"] for x in rows),
                  free_bytes=shutil.disk_usage(folder).free,
                  note="Budget applies to archive storage; results also need free space.")
    result["within_archive_budget"] = result["archive_bytes"] <= budget_bytes
    write_json(output / "archives.json", result)
    write_csv(output / "archives.csv", rows)
    return result


def stage_archive(path, destination, receipts, expected_sha256, reserve_bytes=512*1024**2):
    """Extract missing files or reuse verified staging. Never overwrite source data."""
    path, destination = Path(path).resolve(), Path(destination).resolve()
    receipts = separate_output(receipts, destination, path.parent)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Staging destination is not a directory")
    if digest(path) != expected_sha256:
        raise ValueError("Archive changed since inventory")
    missing, verified = [], 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = set()
        for member in members:
            name = PurePosixPath(member.filename)
            raw_target = destination / member.filename
            target = raw_target.resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or ":" in member.filename or not target.is_relative_to(destination)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                    or member.filename in names):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            names.add(member.filename)
            if member.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"Staging path conflicts with ZIP directory: {target}")
                continue
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.file_size:
                    raise ValueError(f"Staged file differs from archive: {target}. Use a new STAGING_TAG; existing files are unchanged.")
                crc = 0
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b""):
                        crc = zlib.crc32(block, crc)
                if crc != member.CRC:
                    raise ValueError(f"Staged file checksum mismatch: {target}. Use a new STAGING_TAG; existing files are unchanged.")
                verified += 1
            else:
                missing.append(member)
        required = sum(m.file_size for m in missing)
        ancestor = destination if destination.exists() else destination.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if missing and shutil.disk_usage(ancestor).free < required + reserve_bytes:
            raise RuntimeError("Insufficient runtime disk space for missing files plus reserve")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            if member.is_dir():
                (destination / member.filename).mkdir(parents=True, exist_ok=True)
        for member in missing:
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".audit-extract-", delete=False) as stream:
                    temporary = Path(stream.name)
                    with archive.open(member) as source:
                        shutil.copyfileobj(source, stream, length=4*1024*1024)
                if target.exists():
                    raise ValueError(f"Staging target appeared during extraction: {target}")
                os.replace(temporary, target)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    receipt = dict(archive=str(path), sha256=expected_sha256,
                   archive_bytes=path.stat().st_size,
                   unpacked_bytes=sum(m.file_size for m in members),
                   destination=str(destination), verified_existing_files=verified,
                   extracted_files=len(missing),
                   status="reused" if not missing else "resumed" if verified else "extracted")
    receipts.mkdir(parents=True, exist_ok=True)
    write_json(receipts / f"{path.stem}-{expected_sha256[:12]}.json", receipt)
    return receipt


def load_baseline(folder, config):
    """Validate retained metadata, aggregate shards and per-trial envelopes.

    Data hashes are verified separately when that scene is staged. Old source
    fingerprints identify the old run; current source code need not equal it.
    """
    folder = Path(folder)
    run, manifest = read_json(folder / "run.json"), read_json(folder / "manifest.json")
    if run["config"] != config or manifest["config_fingerprint"] != fingerprint(config):
        raise ValueError("Baseline configuration mismatch")
    identity = fingerprint(dict(config=run["config"], sources=run["source_fingerprints"],
                                numpy=run["numpy"], opencv=run["opencv"]))
    if run["identity"] != identity or manifest["run_identity"] != identity:
        raise ValueError("Baseline run identity mismatch")
    if (manifest.get("dataset_fingerprint") != fingerprint(manifest["dataset_files"])
            or manifest.get("matcher") != "sift"
            or manifest.get("matcher_fingerprint") != fingerprint(config["sift"])):
        raise ValueError("Baseline dataset/matcher fingerprint mismatch")
    if manifest["status"] != "completed":
        raise ValueError("Baseline is incomplete")
    trials = {}
    for aggregate in sorted(folder.glob("shards/**/aerial/sift/smoke/trials.json")):
        status = read_json(aggregate.with_name("status.json"))
        rows = read_json(aggregate)
        if (status["status"] != "completed" or status["run_identity"] != identity
                or status["sha256"] != digest(aggregate) or status["trial_count"] != len(rows)):
            raise ValueError("Baseline shard checksum/status mismatch")
        for row in rows:
            saved = checked_trial(aggregate.with_name(Path(row["sample"]).stem + ".json"), row["identity"])
            if saved != row or row["mode"] != "aerial" or row["sample"] in trials:
                raise ValueError("Baseline aggregate/envelope mismatch or duplicate")
            if (row["prior_source"] != config["prior"]["source"]
                    or row["prior_covariance_xy_m2"] != config["prior"]["covariance_xy_m2"]
                    or np.asarray(row["prior_mean_xy"]).shape != (2,)
                    or not np.isfinite(row["prior_mean_xy"]).all()):
                raise ValueError("Baseline prior protocol mismatch")
            trials[row["sample"]] = row
    if not trials:
        raise ValueError("No retained aerial trials found")
    return dict(run_identity=identity, trials=trials, files=manifest["dataset_files"],
                manifest_sha256=digest(folder / "manifest.json"))


def map_polygon(reference):
    h, w = reference.image.shape[:2]
    # Transforms use pixel-centre coordinates. Raster support extends half a pixel.
    return Polygon([reference.transform * p for p in
                    [(-.5, -.5), (w-.5, -.5), (w-.5, h-.5), (-.5, h-.5)]])


def geometry(reference, sample, agl, prior, config):
    _, _, offsets, _, condition = rectify(
        sample["image"], sample["K"], sample["dist"], sample["rotation"], agl,
        config["rectified_gsd_m"], config["max_image_pixels"],
        config["maximum_projection_condition"])
    footprint = offsets + sample["xyz"][:2]
    polygon = Polygon(footprint)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("invalid_footprint")
    cov, mass = config["prior"]["covariance_xy_m2"], config["search_probability_mass"]
    gt = search_bounds(reference, offsets, sample["xyz"][:2], cov, mass)
    actual = search_bounds(reference, offsets, prior, cov, mass)
    def rectangle(bounds):
        start, end = bounds[:2]
        return [list(reference.transform * p) for p in
                [(start[0], start[1]), (end[0], start[1]),
                 (end[0], end[1]), (start[0], end[1])]]
    result = dict(footprint_coverage=float(polygon.intersection(map_polygon(reference)).area/polygon.area),
                  gt_search_coverage=gt[-1], prior_search_coverage=actual[-1],
                  footprint_xy=footprint.tolist(), gt_search_xy=rectangle(gt),
                  prior_search_xy=rectangle(actual), projection_condition=condition,
                  footprint_area_m2=polygon.area)
    flags = []
    if result["footprint_coverage"] < 1 - 1e-9:
        flags.append("footprint_outside_map")
    if gt[-1] < config["minimum_map_coverage"]:
        flags.append("insufficient_map_margin_at_gt")
    if actual[-1] < gt[-1] - 1e-12:
        flags.append("prior_shift_reduces_coverage")
    resource_error = None
    try:
        search_crop(reference, offsets, prior, cov, mass, config["max_image_pixels"],
                    matching_gsd_m=config["rectified_gsd_m"])
    except ValueError as error:
        resource_error = str(error)
        flags.append(resource_error)
    result.update(factors=flags, geometry_supported=resource_error is None,
                  strict_policy_pass=resource_error is None and actual[-1] >= config["minimum_map_coverage"])
    return result


def synthetic_prior(sample_id, xy, config):
    seed = int(fingerprint(dict(seed=config["seed"], sample=sample_id))[:16], 16)
    return xy + np.random.default_rng(seed).multivariate_normal(
        [0, 0], config["prior"]["covariance_xy_m2"])


def audit_scene(dataset_root, reference_path, output, config, baseline_dir):
    """Audit every frame in one scene. Does not call the matcher."""
    validate_config(config)
    root, ref_path = Path(dataset_root).resolve(), Path(reference_path).resolve()
    if not ref_path.is_relative_to(root):
        raise ValueError("Reference must be inside dataset root")
    output = separate_output(output, root, baseline_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Split already frozen; use a new audit output directory")
    baseline = load_baseline(baseline_dir, config)
    scene_id = ref_path.parent.relative_to(root).as_posix()
    scene_dir = output / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    audit_sources = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    existing = list((output / "scenes").glob("**/scene.json"))
    for path in existing:
        old = read_json(path)
        if old["config_fingerprint"] != fingerprint(config) or old["baseline_identity"] != baseline["run_identity"] or old.get("audit_sources") != audit_sources:
            raise ValueError("Audit config or baseline changed; use a new output directory")
    scene = dict(scene_id=scene_id, reference=ref_path.relative_to(root).as_posix(),
                 config_fingerprint=fingerprint(config), config=config,
                 baseline_identity=baseline["run_identity"],
                 audit_sources=audit_sources,
                 numpy=np.__version__, opencv=cv2.__version__, shared_errors=[], files={},
                 fully_verified=False, status="incomplete")
    paths = sorted(ref_path.parent.glob("L??_????.npz"))
    rows = []
    retained = {k: v for k, v in baseline["trials"].items()
                if PurePosixPath(k).parent.as_posix() == scene_id}
    try:
        reference = load_reference(ref_path, "aerial")
        scene["files"] = {p.relative_to(root).as_posix(): digest(p) for p in reference.files}
        # Suspicious pixels are warnings, NOT an inferred nodata mask.
        thumb = reference.image[::max(1, reference.image.shape[0]//512),
                                ::max(1, reference.image.shape[1]//512)]
        scene["map_warnings"] = dict(sampled_black_fraction=float(np.mean(np.all(thumb <= 2, axis=2))),
                                      sampled_white_fraction=float(np.mean(np.all(thumb >= 253, axis=2))),
                                      sampled_intensity_std=float(np.std(thumb)),
                                      interpretation="Indicators only; snow/water/fields may be valid. No nodata inferred.")
        # Verify every aerial file used by retained samples, including external images.
        if retained:
            for name in scene["files"]:
                if baseline["files"].get(name) != scene["files"][name]:
                    raise ValueError(f"Baseline source hash mismatch: {name}")
            missing = set(retained) - {p.relative_to(root).as_posix() for p in paths}
            if missing:
                raise ValueError(f"Missing retained samples: {sorted(missing)}")
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, cv2.error) as error:
        scene["shared_errors"].append(f"{type(error).__name__}: {error}")
        reference = None
    for path in paths:
        sample_id = path.relative_to(root).as_posix()
        row = dict(sample=sample_id, scene_id=scene_id, data_valid=False,
                   geometry_supported=False, strict_policy_pass=False, factors=[], error=None)
        if sample_id in retained:
            row.update(baseline_reason=retained[sample_id]["result"]["reason"], baseline_verified=False)
        try:
            scene["files"][sample_id] = digest(path)
            sample = load_sample(path, pose_world_frame=config["pose_world_frame"])
            hashes = {p.relative_to(root).as_posix(): digest(p) for p in sample["files"]}
            scene["files"].update(hashes)
            old = retained.get(sample_id)
            if old:
                if any(baseline["files"].get(k) != v for k, v in hashes.items()):
                    raise ValueError("baseline_source_hash_mismatch")
                if reference is None:
                    raise ValueError("baseline_shared_sources_unverified")
                names = [p.relative_to(root).as_posix() for p in (*reference.files, *sample["files"])]
                expected = fingerprint(dict(run=baseline["run_identity"], sample=sample_id, mode="aerial",
                                            files={k: baseline["files"][k] for k in names}))
                if old["identity"] != expected:
                    raise ValueError("baseline_trial_identity_mismatch")
            row.update(data_valid=True, gt_xy=sample["xyz"][:2].tolist())
            prior = (np.asarray(old["prior_mean_xy"], float) if old else
                     synthetic_prior(sample_id, sample["xyz"][:2], config))
            row.update(prior_xy=prior.tolist(), prior_source="retained" if old else "synthetic_reproduced")
            if reference is None:
                raise ValueError("shared_reference_unavailable")
            agl = float(sample["xyz"][2] - reference.elevation(sample["xyz"][:2]))
            row.update(agl_m=agl, altitude_eligible=20 <= agl <= 120)
            if not row["altitude_eligible"]:
                raise ValueError("agl_outside_20_120_m")
            if old and not np.isclose(old["agl_m"], agl, atol=1e-6, rtol=0):
                raise ValueError("baseline_agl_mismatch")
            row.update(geometry(reference, sample, agl, prior, config))
            if old:
                recorded = old["result"]["diagnostics"].get("map_coverage")
                row.update(baseline_reason=old["result"]["reason"], baseline_verified=True,
                           baseline_coverage=recorded)
                if recorded is not None and not np.isclose(recorded, row["prior_search_coverage"], atol=1e-12, rtol=0):
                    row["baseline_verified"] = False
                    row["factors"].append("baseline_coverage_mismatch_unresolved")
                if ((old["result"]["reason"] == "insufficient_map_coverage") !=
                        (row["prior_search_coverage"] < config["minimum_map_coverage"])):
                    row["baseline_verified"] = False
                    row["factors"].append("baseline_rejection_policy_mismatch_unresolved")
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, EOFError, zipfile.BadZipFile, cv2.error) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row["factors"].append(str(error))
        rows.append(row)
    scene.update(frame_count=len(rows), data_valid_count=sum(r["data_valid"] for r in rows),
                 geometry_supported_count=sum(r["geometry_supported"] for r in rows),
                 strict_policy_pass_count=sum(r["strict_policy_pass"] for r in rows))
    if not paths:
        scene["shared_errors"].append("no_frame_files")
    scene["fully_verified"] = not scene["shared_errors"] and bool(rows) and all(r["data_valid"] for r in rows)
    count = scene["geometry_supported_count"]
    scene["status"] = ("incomplete" if scene["shared_errors"] else
                       "unusable_current_protocol" if not count else
                       "usable" if count == len(rows) else "partially_usable")
    scene["missing_baseline_samples"] = sorted(set(retained) - {r["sample"] for r in rows})
    scene["factor_counts"] = dict(Counter(f for r in rows for f in r["factors"]))
    write_json(scene_dir / "frames.json", rows)
    write_csv(scene_dir / "frames.csv", rows)
    write_csv(scene_dir / "usable_frames.csv", [r for r in rows if r["data_valid"] and r["geometry_supported"]])
    write_json(scene_dir / "scene.json", scene)
    if reference is not None:
        draw_examples(reference, rows, scene_dir)
    return scene



def audit_dataset(dataset_root, output, config, baseline_dir):
    """Inventory scene directories, including frames whose reference is missing."""
    root = Path(dataset_root).resolve()
    candidates = {}
    for path in sorted(root.rglob("L??_reference.json")):
        if path.parent in candidates:
            raise ValueError(f"Multiple references in one scene: {path.parent}")
        candidates[path.parent] = path
    for path in sorted(root.rglob("L??_????.npz")):
        candidates.setdefault(path.parent, path.parent / (path.name[:3] + "_reference.json"))
    for directory in sorted(root.glob("Scene_*")):
        if directory.is_dir():
            candidates.setdefault(directory, directory / ("L" + directory.name[-2:] + "_reference.json"))
    if not candidates:
        raise ValueError("No scene directories, references or frames found")
    return [audit_scene(root, reference, output, config, baseline_dir)
            for _, reference in sorted(candidates.items())]

def draw_examples(reference, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = reference.image.shape[:2]
    step = max(1, max(h, w)//1000)
    preview = reference.image[::step, ::step]
    suspicious = np.all(preview <= 2, axis=2) | np.all(preview >= 253, axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(preview)
    axes[0].set_title("Map preview")
    axes[1].imshow(preview)
    overlay = np.zeros((*suspicious.shape, 4))
    overlay[suspicious] = [1, 0, 1, .7]
    axes[1].imshow(overlay)
    axes[1].set_title("Near black/white indicators (NOT nodata)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(output / "map-inspection.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    seen = set()
    for row in rows:
        if "footprint_xy" not in row:
            continue
        key = tuple(row["factors"]) or ("covered",)
        if key in seen:
            continue
        seen.add(key)
        fig, ax = plt.subplots(figsize=(8, 7))
        h, w = reference.image.shape[:2]
        step = max(1, max(h, w)//1500)
        bounds = map_polygon(reference).bounds
        ax.imshow(reference.image[::step, ::step], extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        for field, label, color in [("footprint_xy", "Actual footprint", "lime"),
                                    ("gt_search_xy", "Search at GT", "cyan"),
                                    ("prior_search_xy", "Search at prior", "orange")]:
            points = np.asarray(row[field] + [row[field][0]])
            ax.plot(points[:, 0], points[:, 1], color=color, label=label)
        ax.scatter(*row["gt_xy"], c="lime", marker="+", label="GT")
        ax.scatter(*row["prior_xy"], c="orange", marker="x", label="Prior")
        ax.set(title=row["sample"] + "\n" + ", ".join(key), xlabel="East, m", ylabel="North, m", aspect="equal")
        ax.legend(fontsize=8)
        fig.savefig(output / f"geometry-{Path(row['sample']).stem}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def finalize_audit(output):
    """Freeze roles after inventory, before any new matching. Never reshuffle."""
    output = Path(output)
    target = output / "manifest.json"
    frozen = read_json(target) if target.exists() else None
    scenes = [read_json(p) for p in sorted((output / "scenes").glob("**/scene.json"))]
    if not scenes:
        raise ValueError("Audit at least one scene first")
    if len({(s["config_fingerprint"], s["baseline_identity"], fingerprint(s["audit_sources"]),
             s["numpy"], s["opencv"]) for s in scenes}) != 1:
        raise ValueError("Mixed audit configuration/baselines")
    if frozen is not None:
        if set(frozen["roles"]) != {s["scene_id"] for s in scenes}:
            raise ValueError("Frozen scene inventory changed; reports cannot be rebuilt")
        for scene in scenes:
            scene_id = scene["scene_id"]
            folder = output / "scenes" / scene_id
            if (digest(folder / "scene.json") != frozen["scene_report_hashes"][scene_id]
                    or digest(folder / "frames.json") != frozen["frame_report_hashes"][scene_id]):
                raise ValueError(f"Frozen audit evidence changed: {scene_id}")
        manifest = frozen
        roles = frozen["roles"]
        holdout = {key for key, value in roles.items() if value == "holdout"}
    else:
        candidates = [s["scene_id"] for s in scenes if s["fully_verified"] and
                      PurePosixPath(s["scene_id"]).name != "Scene_05"]
        ranked = sorted(candidates, key=lambda x: hashlib.sha256(f"42:{x}".encode()).hexdigest())
        holdout = set(ranked[:math.ceil(len(ranked)/3)])
        roles = {s["scene_id"]: ("holdout" if s["scene_id"] in holdout else
                 "working" if s["fully_verified"] or PurePosixPath(s["scene_id"]).name == "Scene_05"
                 else "unassigned_incomplete") for s in scenes}
        manifest = dict(schema_version=1, matcher="sift", reference_mode="aerial", seed=42,
                        roles=roles, reserve_formed=bool(holdout), split_frozen=True,
                        config=scenes[0]["config"], config_fingerprint=scenes[0]["config_fingerprint"],
                        baseline_identity=scenes[0]["baseline_identity"],
                        audit_sources=scenes[0]["audit_sources"],
                        numpy=scenes[0]["numpy"], opencv=scenes[0]["opencv"],
                        frame_report_hashes={s["scene_id"]: digest(output / "scenes" / s["scene_id"] / "frames.json") for s in scenes},
                        scene_report_hashes={s["scene_id"]: digest(output / "scenes" / s["scene_id"] / "scene.json") for s in scenes},
                        limitations=["No training or flight probabilities", "No matching on holdout",
                                     "Planar geometry does not establish relief/parallax suitability",
                                     "GT attitude and DSM-derived AGL; synthetic XY prior"])
    for scene in scenes:
        scene["role"] = roles[scene["scene_id"]]
    summary_fields = ("scene_id", "status", "role", "fully_verified", "frame_count",
                      "data_valid_count", "geometry_supported_count", "strict_policy_pass_count",
                      "shared_errors", "factor_counts", "missing_baseline_samples", "map_warnings")
    write_csv(output / "scenes.csv", [{k: s.get(k) for k in summary_fields} for s in scenes])
    report = ["# AnyVisLoc audit: SIFT and aerial reference maps", "",
              "The prior is a synthetic XY position estimate, not recorded VIO output.",
              "95% is an engineering threshold on search rectangle area, not localization success probability.",
              "Footprint coverage uses polygon intersection; search rectangles retain the original pixel rounding.",
              "Near-black/white regions are inspection indicators, not established missing data.", "",
              "## Scenes", ""]
    for scene in scenes:
        report.append(f"- {scene['scene_id']}: {scene['status']}; role {scene['role']}; "
                      f"frames {scene['frame_count']}, geometry supported {scene['geometry_supported_count']}, "
                      f"strict policy passed {scene['strict_policy_pass_count']}. "
                      f"Shared errors: {scene['shared_errors']}; factors: {scene['factor_counts']}.")
    if not holdout:
        report += ["", "No independent holdout formed: no new scenes with fully verified data are available."]
    report += ["", "## Baseline frames: explanations or unresolved evidence", ""]
    for scene in scenes:
        for row in read_json(output / "scenes" / scene["scene_id"] / "frames.json"):
            if "baseline_reason" in row:
                coverage = {k: row.get(k) for k in ("footprint_coverage", "gt_search_coverage", "prior_search_coverage")}
                report.append(f"- {row['sample']}: {row['baseline_reason']}; coverage {coverage}; "
                              f"factors {row['factors']}; error {row['error']}; "
                              f"baseline comparability verified: {row.get('baseline_verified', False)}.")
        if scene["missing_baseline_samples"]:
            report.append(f"- Missing baseline frames: {scene['missing_baseline_samples']}")
    report += ["", "## Interpretation", "",
               "Footprint outside the map means part of the actual visible ground is unavailable. "
               "Insufficient margin at GT means the map cannot support the configured search uncertainty. "
               "Reduced coverage at the prior shows the contribution of its particular displacement. Factors may coexist.",
               "Read/coordinate errors retain their exception details and require inspection; "
               "they do not establish poor texture as the cause.",
               "Eligibility refers to the current protocol; it does not guarantee SIFT success or validity of the planar approximation.",
               "Per-frame details: scenes/<scene>/frames.csv; geometry figures: geometry-*.png.",
               "The control experiment is stored separately and does not change the production localization policy."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if frozen is None:
        write_json(target, manifest)
    return manifest


def run_control(dataset_root, baseline_dir, audit_output, output, config):
    """Replay only the 16 original Scene_05 coverage rejects, never holdout."""
    root = Path(dataset_root).resolve()
    output = separate_output(output, root, baseline_dir, audit_output)
    baseline = load_baseline(baseline_dir, config)
    manifest = read_json(Path(audit_output) / "manifest.json")
    if manifest["config"] != config or manifest["baseline_identity"] != baseline["run_identity"]:
        raise ValueError("Control differs from frozen audit")
    old_run = read_json(Path(baseline_dir) / "run.json")
    if old_run["numpy"] != np.__version__ or old_run["opencv"] != cv2.__version__:
        raise ValueError("Control requires baseline NumPy/OpenCV versions; see baseline run.json")
    trials = [r for r in baseline["trials"].values() if PurePosixPath(r["sample"]).parent.name == "Scene_05"]
    selected = [r for r in trials if r["result"]["reason"] == "insufficient_map_coverage"]
    if (len(trials) != 20 or len(selected) != 16
            or sum(r["result"]["status"] == "accepted" for r in trials) != 4):
        raise ValueError("Expected original 20 Scene_05 aerial trials and 16 coverage rejects")
    scene_id = PurePosixPath(trials[0]["sample"]).parent.as_posix()
    if manifest["roles"].get(scene_id) != "working":
        raise ValueError("Control is restricted to the working Scene_05")
    scene_dir = Path(audit_output) / "scenes" / scene_id
    if digest(scene_dir / "scene.json") != manifest["scene_report_hashes"][scene_id]:
        raise ValueError("Frozen scene audit changed")
    if digest(scene_dir / "frames.json") != manifest["frame_report_hashes"][scene_id]:
        raise ValueError("Frozen frame audit changed")
    audited = {r["sample"]: r for r in read_json(scene_dir / "frames.json")}
    scene = read_json(scene_dir / "scene.json")
    if not all(audited.get(r["sample"], {}).get("baseline_verified") for r in trials):
        raise ValueError("All 20 baseline frames must have verified audit evidence first")
    reference = load_reference(root / scene["reference"], "aerial")
    # Verify baseline data again immediately before matching.
    required = {p.relative_to(root).as_posix() for p in reference.files}
    for row in trials:
        sample = load_sample(root / row["sample"], pose_world_frame=config["pose_world_frame"])
        required.update(p.relative_to(root).as_posix() for p in sample["files"])
    if any(digest(root / name) != baseline["files"].get(name) for name in required):
        raise ValueError("Baseline sources changed before control")
    identity = dict(config=config, coverage_policy="audit_clipped", baseline_identity=baseline["run_identity"],
                    audit_manifest_sha256=digest(Path(audit_output) / "manifest.json"),
                    sources={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    numpy=np.__version__, opencv=cv2.__version__)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    if run_path.exists() and read_json(run_path) != identity:
        raise ValueError("Control identity changed; use a new output directory")
    write_json(run_path, identity)
    results = []
    for old in selected:
        sample = load_sample(root / old["sample"], pose_world_frame=config["pose_world_frame"])
        path = output / (Path(old["sample"]).stem + ".json")
        trial_identity = fingerprint(dict(run=identity, sample=old["sample"]))
        if path.exists():
            row = checked_trial(path, trial_identity)
        else:
            result = localize(sample["image"], sample["K"], sample["dist"], sample["rotation"],
                              old["agl_m"], old["prior_mean_xy"], old["prior_covariance_xy_m2"],
                              config["search_probability_mass"], reference, config,
                              coverage_policy="audit_clipped")
            residual = None if result["z"] is None else (np.asarray(result["z"])-sample["xyz"][:2]).tolist()
            row = dict(identity=trial_identity, sample=old["sample"], baseline_reason=old["result"]["reason"],
                       prior_xy=old["prior_mean_xy"], result=result, residual_xy_m=residual,
                       error_m=None if residual is None else float(np.linalg.norm(residual)))
            write_json(path, dict(trial=row, sha256=fingerprint(row)))
        results.append(row)
    write_csv(output / "comparison.csv", [dict(sample=r["sample"], baseline_reason=r["baseline_reason"],
              control_status=r["result"]["status"], control_reason=r["result"]["reason"],
              error_m=r["error_m"], residual_xy_m=r["residual_xy_m"],
              diagnostics=r["result"]["diagnostics"]) for r in results])
    write_json(output / "summary.json", dict(trial_count=len(results),
               reasons=dict(Counter(r["result"]["reason"] for r in results)),
               accepted_errors_m=[r["error_m"] for r in results if r["error_m"] is not None],
               production_policy_changed=False, flight_probability_estimate=False))
    lines = ["# Control experiment: clipped search region", "",
             "Only the response to coverage below 95% changed; this does not approve a new production policy.",
             "SIFT acceptance does not establish a correct fix: inspect XY error against ground truth.", ""]
    for r in results:
        lines.append(f"- {r['sample']}: insufficient_map_coverage -> {r['result']['reason']}; "
                     f"error, m: {r['error_m']}; factors: {audited[r['sample']]['factors']}.")
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return results
