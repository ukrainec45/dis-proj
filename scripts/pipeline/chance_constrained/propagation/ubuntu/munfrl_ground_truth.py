#!/usr/bin/env python3
"""Export an offline MUN-FRL reference trajectory from a ROS 2 bag."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


PATH_TOPICS = {
    "frl": "/globalEstimator/frl_path",
    "ppk": "/globalEstimator/ppk_path",
}
FIX_TOPIC = "/fix"
FIELDS = (
    "timestamp_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
    "orientation_valid", "source_topic",
)


def stamp_seconds(header):
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _reader(bag):
    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def _export_path(reader, topic, message_type):
    """Return the most complete Path snapshot, deduplicated by pose timestamp."""
    message_class = get_message(message_type)
    best = {}
    best_span = (-1, -1.0)
    while reader.has_next():
        _, raw, _ = reader.read_next()
        message = deserialize_message(raw, message_class)
        if not message.poses:
            continue
        candidate = {}
        for pose in message.poses:
            timestamp = stamp_seconds(pose.header)
            if timestamp <= 0:
                continue
            p = pose.pose.position
            q = pose.pose.orientation
            candidate[timestamp] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not candidate:
            continue
        span = (len(candidate), max(candidate) - min(candidate))
        if span > best_span:
            best = candidate
            best_span = span
    if not best:
        raise ValueError(f"{topic} contains no timestamped poses")
    orientation_valid = topic == PATH_TOPICS["frl"]
    return [
        {
            "timestamp_s": timestamp,
            "x_m": values[0], "y_m": values[1], "z_m": values[2],
            "qx": values[3], "qy": values[4], "qz": values[5], "qw": values[6],
            "orientation_valid": int(orientation_valid),
            "source_topic": topic,
        }
        for timestamp, values in sorted(best.items())
    ]


def _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m):
    a = 6378137.0
    eccentricity_squared = 6.69437999014e-3
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_latitude = math.sin(latitude)
    prime_vertical = a / math.sqrt(1.0 - eccentricity_squared * sin_latitude**2)
    return (
        (prime_vertical + altitude_m) * math.cos(latitude) * math.cos(longitude),
        (prime_vertical + altitude_m) * math.cos(latitude) * math.sin(longitude),
        (prime_vertical * (1.0 - eccentricity_squared) + altitude_m) * sin_latitude,
    )


def _export_fix(reader, topic, message_type):
    message_class = get_message(message_type)
    fixes = []
    while reader.has_next():
        _, raw, _ = reader.read_next()
        message = deserialize_message(raw, message_class)
        timestamp = stamp_seconds(message.header)
        values = (message.latitude, message.longitude, message.altitude)
        if timestamp > 0 and all(math.isfinite(value) for value in values):
            fixes.append((timestamp, *values))
    if not fixes:
        raise ValueError(f"{topic} contains no finite timestamped fixes")

    latitude0, longitude0, altitude0 = fixes[0][1:]
    origin = _geodetic_to_ecef(latitude0, longitude0, altitude0)
    latitude = math.radians(latitude0)
    longitude = math.radians(longitude0)
    rows = []
    for timestamp, lat, lon, altitude in fixes:
        ecef = _geodetic_to_ecef(lat, lon, altitude)
        dx, dy, dz = (ecef[index] - origin[index] for index in range(3))
        east = -math.sin(longitude) * dx + math.cos(longitude) * dy
        north = (-math.sin(latitude) * math.cos(longitude) * dx
                 - math.sin(latitude) * math.sin(longitude) * dy
                 + math.cos(latitude) * dz)
        up = (math.cos(latitude) * math.cos(longitude) * dx
              + math.cos(latitude) * math.sin(longitude) * dy
              + math.sin(latitude) * dz)
        rows.append({
            "timestamp_s": timestamp,
            "x_m": east, "y_m": north, "z_m": up,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "orientation_valid": 0,
            "source_topic": topic,
        })
    return rows


def export_ground_truth(bag, output, source):
    bag = Path(bag).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reader = _reader(bag)
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic = FIX_TOPIC if source == "fix" else PATH_TOPICS[source]
    if topic not in types:
        raise ValueError(f"bag does not contain {topic}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    rows = (_export_fix(reader, topic, types[topic]) if source == "fix"
            else _export_path(reader, topic, types[topic]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format_version": 1,
        "bag": str(bag),
        "source": source,
        "source_topic": topic,
        "sample_count": len(rows),
        "timestamp_start_s": rows[0]["timestamp_s"],
        "timestamp_end_s": rows[-1]["timestamp_s"],
        "orientation_valid": bool(int(rows[0]["orientation_valid"])),
        "usage": "offline evaluation only; not supplied to OpenVINS",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export MUN-FRL FRL/PPK/fix ground truth to a metric CSV")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=("frl", "ppk", "fix"), default="frl")
    args = parser.parse_args(argv)
    rows = export_ground_truth(args.bag, args.output, args.source)
    print(f"wrote {len(rows)} ground-truth samples to {args.output}")


if __name__ == "__main__":
    main()
