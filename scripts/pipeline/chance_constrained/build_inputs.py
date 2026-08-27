"""CLI for the first chance-constrained preprocessing increment."""

from __future__ import annotations

import argparse

from .artifacts import build_manifest, save_scene_artifact
from .config import load_config
from .scene import build_scene_edges


def build_inputs(config_path, output_dir, *, force=False):
    config = load_config(config_path)
    arrays = build_scene_edges(config)
    manifest = build_manifest(config)
    artifact_path, manifest_path = save_scene_artifact(
        output_dir, arrays, manifest, force=force)
    return arrays, artifact_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build scene and directed edges for chance-constrained planning")
    parser.add_argument("--config", required=True, help="versioned JSON configuration")
    parser.add_argument("--output-dir", required=True,
                        help="directory for scene_edges.npz and manifest.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing first-increment artifacts")
    args = parser.parse_args(argv)
    arrays, artifact_path, manifest_path = build_inputs(
        args.config, args.output_dir, force=args.force)
    rows, cols = map(int, arrays["grid_shape"])
    print(f"wrote {artifact_path}: {rows}x{cols} cells, "
          f"{int(arrays['edge_count'])} directed edges")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

