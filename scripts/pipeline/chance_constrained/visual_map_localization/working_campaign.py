"""Resumable SIFT campaign on frozen working-scene geometry candidates.

The 95% raster-extent threshold is diagnostic only. Real images run in hosted
Colab; local tests use synthetic fixtures. No holdout matching is supported.
"""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .anyvisloc import load_reference, load_sample
from .anyvisloc_smoke import checked_trial, digest, fingerprint, write_json
from .audit import read_json, separate_output, write_csv
from .smoke_localizer import localize


def _manifest(audit_dir):
    manifest = read_json(Path(audit_dir) / "manifest.json")
    if fingerprint(manifest["config"]) != manifest["config_fingerprint"]:
        raise ValueError("Audit configuration fingerprint mismatch")
    if not manifest["split_frozen"]:
        raise ValueError("Freeze the audit before working-scene matching")
    if manifest["reference_mode"] != "aerial" or manifest["matcher"] != "sift":
        raise ValueError("This campaign supports aerial maps and SIFT only")
    if (manifest["numpy"] != np.__version__ or manifest["opencv"] != cv2.__version__):
        raise ValueError("Use the audited NumPy/OpenCV versions from audit manifest.json")
    return manifest


def _scene(audit_dir, manifest, scene_id):
    if manifest["roles"].get(scene_id) != "working":
        raise ValueError(f"Matching is restricted to frozen working scenes: {scene_id}")
    folder = Path(audit_dir) / "scenes" / scene_id
    for filename, key in [("scene.json", "scene_report_hashes"), ("frames.json", "frame_report_hashes")]:
        if digest(folder / filename) != manifest[key][scene_id]:
            raise ValueError(f"Frozen audit evidence changed: {scene_id}/{filename}")
    scene, frames = read_json(folder / "scene.json"), read_json(folder / "frames.json")
    if scene["config_fingerprint"] != manifest["config_fingerprint"]:
        raise ValueError("Scene/audit configuration mismatch")
    selected = [r for r in frames if r["data_valid"] and r["geometry_supported"]]
    if len(selected) != scene["geometry_supported_count"]:
        raise ValueError("Frozen candidate count mismatch")
    return scene, selected


def _run_identity(audit_dir, manifest):
    return dict(schema_version=1, campaign="working_available_crop", matcher="sift",
                reference_mode="aerial", coverage_policy="available_crop",
                config=manifest["config"],
                audit_manifest_sha256=digest(Path(audit_dir) / "manifest.json"),
                working_scenes=sorted(k for k, role in manifest["roles"].items() if role == "working"),
                holdout_matching_enabled=False,
                numpy=np.__version__, opencv=cv2.__version__,
                source_fingerprints={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def _metrics(rows, legacy=False):
    accepted = [r for r in rows if r["result"]["status"] == "accepted"
                and (not legacy or r["legacy_coverage_pass"])]
    errors = [r["error_m"] for r in accepted]
    return dict(trial_count=len(rows), accepted_count=len(accepted), reject_count=len(rows)-len(accepted),
                acc_at_m={str(t): sum(e <= t for e in errors)/len(rows) if rows else None
                          for t in (1, 3, 5, 10, 20)},
                accepted_error_m=dict(min=min(errors), median=float(np.median(errors)),
                    mean=float(np.mean(errors)), max=max(errors),
                    rmse=float(np.sqrt(np.mean(np.square(errors))))) if errors else None,
                accepted_error_over_m={str(t): sum(e > t for e in errors) for t in (1, 3, 5, 10, 20)})


def _export_scene(folder, rows, expected):
    summary = dict(status="completed" if len(rows) == expected else "running", expected_trial_count=expected,
                   available_crop=_metrics(rows), legacy_95pct_counterfactual=_metrics(rows, legacy=True),
                   reasons=dict(Counter(r["result"]["reason"] for r in rows)),
                   legacy_coverage_reject_count=sum(not r["legacy_coverage_pass"] for r in rows),
                   newly_accepted_count=sum(not r["legacy_coverage_pass"] and r["result"]["status"] == "accepted" for r in rows),
                   flight_probability_estimate=False, R_calibrated=False)
    write_json(folder / "summary.json", summary)
    write_csv(folder / "trials.csv", [dict(sample=r["sample"], status=r["result"]["status"],
              reason=r["result"]["reason"], error_m=r["error_m"], residual_xy_m=r["residual_xy_m"],
              legacy_coverage_pass=r["legacy_coverage_pass"], prior_xy=r["prior_xy"],
              footprint_coverage=r["footprint_coverage"], map_coverage=r["result"]["diagnostics"].get("map_coverage"),
              diagnostics=r["result"]["diagnostics"], runtime_s=r["result"].get("runtime_s")) for r in rows])
    return summary


def summarize_campaign(audit_dir, output):
    """Summarize verified checkpoints, including explicitly incomplete scenes."""
    output = Path(output)
    manifest = _manifest(audit_dir)
    identity = _run_identity(audit_dir, manifest)
    if read_json(output / "run.json") != identity:
        raise ValueError("Campaign identity changed; use a new output directory")
    summaries, all_rows, pending = {}, [], []
    for scene_id in identity["working_scenes"]:
        _, selected = _scene(audit_dir, manifest, scene_id)
        folder = output / "scenes" / scene_id
        rows = []
        for frame in selected:
            path = folder / "trials" / (Path(frame["sample"]).stem + ".json")
            if path.exists():
                trial_id = fingerprint(dict(run=identity, scene_id=scene_id, sample=frame["sample"]))
                row = checked_trial(path, trial_id)
                if row["sample"] != frame["sample"]:
                    raise ValueError("Trial sample mismatch")
                rows.append(row)
        if not rows and selected:
            pending.append(scene_id)
            summaries[scene_id] = dict(status="pending", expected_trial_count=len(selected), completed_trial_count=0)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        summaries[scene_id] = _export_scene(folder, rows, len(selected))
        all_rows.extend(rows)
    completed = all(s["status"] == "completed" for s in summaries.values())
    summary = dict(status="completed" if completed else "incomplete", scenes=summaries,
                   available_crop=_metrics(all_rows), legacy_95pct_counterfactual=_metrics(all_rows, legacy=True),
                   expected_trial_count=sum(s["expected_trial_count"] for s in summaries.values()),
                   completed_trial_count=len(all_rows), holdout_matching_enabled=False,
                   flight_probability_estimate=False, R_calibrated=False)
    write_json(output / "summary.json", summary)
    lines = ["# Working-scene matching without a coverage threshold", "",
             "Only the early 95% raster-extent coverage gate is disabled. Map coverage remains diagnostic.",
             "SIFT thresholds, priors, AGL domain, scale and geometry/resource limits are unchanged.",
             "The legacy policy column is a counterfactual applied to these same outcomes, not a second matcher run.",
             "Accepted errors are relative to retained dataset GT; acceptance is not proof of correctness.", "",
             f"Campaign status: {summary['status']}; completed {len(all_rows)}/{summary['expected_trial_count']} trials.", "",
             "| Scene | Status | Completed/expected | Accepted | Accepted with error >1 m | Accepted with error >5 m |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for scene_id, s in summaries.items():
        if s["status"] == "pending":
            lines.append(f"| {scene_id} | pending | 0/{s['expected_trial_count']} | - | - | - |")
        else:
            m = s["available_crop"]
            lines.append(f"| {scene_id} | {s['status']} | {m['trial_count']}/{s['expected_trial_count']} | "
                         f"{m['accepted_count']} | {m['accepted_error_over_m']['1']} | {m['accepted_error_over_m']['5']} |")
    lines += ["", "## Aggregate results (completed trials only)", "",
              f"Available crop: {summary['available_crop']}", "",
              f"Legacy 95% gate counterfactual: {summary['legacy_95pct_counterfactual']}", "",
              "Error cutoffs are descriptive accuracy levels, not fitted local/gross labels.",
              "Both denominators include all completed candidate attempts, including rejects. Incomplete results are provisional.",
              "Frame selection is fixed by the audit before matching. Excluded-domain frames are documented in the audit.",
              "Black padding inside the raster is not removed or automatically treated as nodata. This limitation is unchanged.",
              "Scene roles and original audit/control results are unchanged. No holdout matching or model fitting is performed."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run_working_scene(dataset_root, audit_dir, output, scene_id, *, progress=None):
    """Run/resume one frozen working scene. Sources are rehashed before matching."""
    manifest = _manifest(audit_dir)
    scene, selected = _scene(audit_dir, manifest, scene_id)
    root = Path(dataset_root).resolve()
    output = separate_output(output, root, audit_dir)
    config = manifest["config"]
    identity = _run_identity(audit_dir, manifest)
    if (output / "run.json").exists():
        if read_json(output / "run.json") != identity:
            raise ValueError("Campaign code/config/environment changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty campaign output directory")
    reference = load_reference(root / scene["reference"], "aerial")
    required = {p.relative_to(root).as_posix() for p in reference.files}
    if progress:
        progress(f"{scene_id}: verifying source hashes for {len(selected)} frozen candidates")
    for frame in selected:
        sample = load_sample(root / frame["sample"], pose_world_frame=config["pose_world_frame"])
        required.update(p.relative_to(root).as_posix() for p in sample["files"])
    for name in sorted(required):
        if digest(root / name) != scene["files"].get(name):
            raise ValueError(f"Audited source changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", identity)
    folder = output / "scenes" / scene_id
    write_json(folder / "status.json", dict(status="running", expected_trial_count=len(selected)))
    rows = []
    for frame in selected:
        sample_id = frame["sample"]
        trial_id = fingerprint(dict(run=identity, scene_id=scene_id, sample=sample_id))
        path = folder / "trials" / (Path(sample_id).stem + ".json")
        if path.exists():
            row = checked_trial(path, trial_id)
        else:
            sample = load_sample(root / sample_id, pose_world_frame=config["pose_world_frame"])
            agl = float(sample["xyz"][2]-reference.elevation(sample["xyz"][:2]))
            if not np.isclose(agl, frame["agl_m"], atol=1e-6, rtol=0):
                raise ValueError(f"Audited AGL changed: {sample_id}")
            result = localize(sample["image"], sample["K"], sample["dist"], sample["rotation"],
                              agl, frame["prior_xy"], config["prior"]["covariance_xy_m2"],
                              config["search_probability_mass"], reference, config,
                              coverage_policy="available_crop")
            coverage = result["diagnostics"].get("map_coverage")
            if coverage is not None and not np.isclose(coverage, frame["prior_search_coverage"], atol=1e-12, rtol=0):
                raise ValueError(f"Audited coverage changed: {sample_id}")
            residual = None if result["z"] is None else (np.asarray(result["z"][:2])-sample["xyz"][:2]).tolist()
            row = dict(identity=trial_id, sample=sample_id, result=result,
                       prior_xy=frame["prior_xy"], gt_xy=sample["xyz"][:2].tolist(), agl_m=agl,
                       legacy_coverage_pass=frame["strict_policy_pass"], footprint_coverage=frame["footprint_coverage"],
                       residual_xy_m=residual, error_m=None if residual is None else float(np.linalg.norm(residual)))
            write_json(path, dict(trial=row, sha256=fingerprint(row)))
        rows.append(row)
        if progress and (len(rows) % 20 == 0 or len(rows) == len(selected)):
            progress(f"{scene_id}: {len(rows)}/{len(selected)} trials saved or reused")
    write_json(folder / "status.json", dict(status="completed", trial_count=len(rows), expected_trial_count=len(selected)))
    return summarize_campaign(audit_dir, output)
