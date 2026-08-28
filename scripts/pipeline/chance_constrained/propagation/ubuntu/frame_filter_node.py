#!/usr/bin/env python3
"""Deterministically thin or suppress EuRoC cam0 frames during ROS2 replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_seconds(message):
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


class FrameFilter(Node):
    def __init__(self):
        super().__init__("euroc_frame_filter")
        self.declare_parameter("stride", 1)
        self.declare_parameter("base_regime", "nominal_20hz")
        self.declare_parameter("blackout_starts_s", "[]")
        self.declare_parameter("blackout_durations_s", "[]")
        self.declare_parameter("schedule_csv", "/tmp/openvins_frame_schedule.csv")

        self.stride = int(self.get_parameter("stride").value)
        self.base_regime = str(self.get_parameter("base_regime").value)
        starts = json.loads(str(self.get_parameter("blackout_starts_s").value))
        durations = json.loads(str(self.get_parameter("blackout_durations_s").value))
        if self.stride < 1:
            raise ValueError("stride must be at least one")
        if len(starts) != len(durations) or any(float(value) <= 0 for value in durations):
            raise ValueError("blackout starts and positive durations must have equal lengths")
        self.blackouts = tuple((float(start), float(start) + float(duration))
                               for start, duration in zip(starts, durations))

        output = Path(str(self.get_parameter("schedule_csv").value)).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.stream,
            fieldnames=("timestamp_s", "relative_time_s", "camera_available",
                        "forwarded", "visual_regime"),
        )
        self.writer.writeheader()
        self.publisher = self.create_publisher(Image, "output", qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            Image, "input", self.on_image, qos_profile_sensor_data)
        self.first_timestamp = None
        self.frame_index = 0

    def on_image(self, message):
        timestamp = stamp_seconds(message)
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        relative = timestamp - self.first_timestamp
        camera_available = not any(start <= relative < end for start, end in self.blackouts)
        forwarded = camera_available and self.frame_index % self.stride == 0
        regime = self.base_regime if camera_available else "blackout"
        if forwarded:
            self.publisher.publish(message)
        self.writer.writerow({
            "timestamp_s": f"{timestamp:.15f}",
            "relative_time_s": f"{relative:.9f}",
            "camera_available": int(camera_available),
            "forwarded": int(forwarded),
            "visual_regime": regime,
        })
        self.stream.flush()
        self.frame_index += 1

    def destroy_node(self):
        if not self.stream.closed:
            self.stream.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = FrameFilter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
