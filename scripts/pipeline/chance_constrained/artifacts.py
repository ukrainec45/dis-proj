"""Versioned persistence and source-integrity checks for scene artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from .config import FORMAT_VERSION, PipelineConfig


REQUIRED_ARRAYS = {
    "format_version", "crs", "extent", "grid_shape", "dem_center_m",
    "occupancy", "start_cell_xy", "delivery_center_xy", "delivery_radius_m",
    "edge_start_cell_xy", "edge_end_cell_xy", "edge_goal_entry_fraction",
    "edge_travel_time_s", "edge_feasible",
}


def source_fingerprint(path):
    """Return a reproducibility fingerprint for one input file."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def build_manifest(config: PipelineConfig, artifact_name="scene_edges.npz"):
    sources = {
        key: source_fingerprint(path)
        for key, path in config.source_paths.items()
        if path is not None
    }
    return {
        "artifact_format_version": FORMAT_VERSION,
        "configuration_schema_version": config.raw["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact_name,
        "configuration_path": str(config.config_path),
        "configuration": config.raw,
        "resolved_sources": {key: str(path) if path is not None else None
                             for key, path in config.source_paths.items()},
        "source_fingerprints": sources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _atomic_npz(path, arrays):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json(path, payload):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_scene_artifact(output_dir, arrays, manifest, *, force=False):
    """Atomically save ``scene_edges.npz`` and ``manifest.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "scene_edges.npz"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (artifact_path, manifest_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass force=True")
    _atomic_npz(artifact_path, arrays)
    _atomic_json(manifest_path, manifest)
    return artifact_path, manifest_path


def read_manifest(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported manifest artifact format: "
                         f"{manifest.get('artifact_format_version')}")
    return manifest


def verify_source_fingerprints(manifest):
    """Raise when any source file is missing or differs from the manifest."""
    for key, expected in manifest.get("source_fingerprints", {}).items():
        path = Path(expected["path"])
        if not path.is_file():
            raise FileNotFoundError(f"source {key} is missing: {path}")
        actual = source_fingerprint(path)
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"source {key} changed after artifact creation: {path}")
    return True


def _validate_arrays(arrays):
    if "format_version" in arrays:
        version = int(np.asarray(arrays["format_version"]).item())
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported scene artifact format: {version}")
    missing = REQUIRED_ARRAYS - arrays.keys()
    if missing:
        raise ValueError(f"scene artifact is missing arrays: {', '.join(sorted(missing))}")
    rows, cols = map(int, np.asarray(arrays["grid_shape"]))
    if np.asarray(arrays["dem_center_m"]).shape != (rows, cols):
        raise ValueError("dem_center_m does not match grid_shape")
    if np.asarray(arrays["occupancy"]).shape != (rows, cols):
        raise ValueError("occupancy does not match grid_shape")
    edge_count = len(np.asarray(arrays["edge_start_cell_xy"]))
    if np.asarray(arrays["edge_end_cell_xy"]).shape != (edge_count, 2):
        raise ValueError("edge endpoints have inconsistent shapes")
    if len(np.asarray(arrays["edge_goal_entry_fraction"])) != edge_count:
        raise ValueError("edge goal fractions do not match edge count")


def load_scene_edges(path, *, manifest_path=None, verify_sources=False):
    """Load and validate an artifact, optionally checking original inputs."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate_arrays(arrays)
    if verify_sources:
        selected_manifest = (Path(manifest_path) if manifest_path is not None
                             else path.with_name("manifest.json"))
        verify_source_fingerprints(read_manifest(selected_manifest))
    return arrays
