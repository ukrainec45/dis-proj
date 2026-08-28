#!/usr/bin/env python3
"""Record the propagated OpenVINS odometry stream in a stable CSV schema."""

from __future__ import annotations

import csv
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


BASE_FIELDS = (
    "timestamp_s", "qx", "qy", "qz", "qw", "px_m", "py_m", "pz_m",
    "vx_mps", "vy_mps", "vz_mps",
)
FIELDS = (BASE_FIELDS
          + tuple(f"pose_cov_{index}" for index in range(36))
          + tuple(f"twist_cov_{index}" for index in range(36)))


class OdometryRecorder(Node):
    def __init__(self):
        super().__init__("openvins_odometry_recorder")
        self.declare_parameter("output_csv", "/tmp/openvins_odometry.csv")
        output = Path(str(self.get_parameter("output_csv").value)).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.subscription = self.create_subscription(
            Odometry, "input", self.on_odometry, qos_profile_sensor_data)

    def on_odometry(self, message):
        orientation = message.pose.pose.orientation
        position = message.pose.pose.position
        velocity = message.twist.twist.linear
        timestamp = (float(message.header.stamp.sec)
                     + float(message.header.stamp.nanosec) * 1e-9)
        row = {
            "timestamp_s": f"{timestamp:.15f}",
            "qx": orientation.x,
            "qy": orientation.y,
            "qz": orientation.z,
            "qw": orientation.w,
            "px_m": position.x,
            "py_m": position.y,
            "pz_m": position.z,
            "vx_mps": velocity.x,
            "vy_mps": velocity.y,
            "vz_mps": velocity.z,
        }
        row.update({f"pose_cov_{index}": value
                    for index, value in enumerate(message.pose.covariance)})
        row.update({f"twist_cov_{index}": value
                    for index, value in enumerate(message.twist.covariance)})
        self.writer.writerow(row)
        self.stream.flush()

    def destroy_node(self):
        if not self.stream.closed:
            self.stream.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = OdometryRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
