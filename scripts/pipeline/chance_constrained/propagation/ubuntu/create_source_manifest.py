#!/usr/bin/env python3
"""Record absolute dataset/config paths and deterministic SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    total = 0
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
    return {"path": str(path), "size_bytes": total, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs=3,
                        metavar=("ID", "ROS2_BAG", "GROUND_TRUTH"), required=True)
    parser.add_argument("--openvins-config-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    datasets = {
        dataset_id: {
            "ros2_bag": fingerprint(bag),
            "ground_truth": fingerprint(ground_truth),
        }
        for dataset_id, bag, ground_truth in args.dataset
    }
    payload = {
        "format_version": 1,
        "datasets": datasets,
        "openvins_config_manifests": [
            fingerprint(path) for path in args.openvins_config_manifest],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
