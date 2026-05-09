from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import time

from cv_bridge import CvBridge
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraRecorderNode(Node):
    def __init__(self):
        super().__init__('real_camera_recorder_node')
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw/image_color').value
        self.output_dir = Path(str(self.declare_parameter('output_dir', 'run_logs/camera_recordings').value))
        self.output_name = str(self.declare_parameter('output_name', '').value)
        self.fps = float(self.declare_parameter('fps', 20.0).value)
        self.max_duration_sec = float(self.declare_parameter('max_duration_sec', 0.0).value)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not self.output_name:
            self.output_name = f'camera_{stamp}.mp4'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / self.output_name
        self.csv_path = self.video_path.with_suffix('.csv')

        self.bridge = CvBridge()
        self.writer: cv2.VideoWriter | None = None
        self.csv_file = self.csv_path.open('w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_index', 'ros_time_sec', 'wall_time_sec'])
        self.frame_index = 0
        self.start_wall_time = time.monotonic()

        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        if self.max_duration_sec > 0.0:
            self.create_timer(0.5, self.timeout_callback)
        self.get_logger().info(f'recording {self.image_topic} to {self.video_path}')

    def timeout_callback(self) -> None:
        if time.monotonic() - self.start_wall_time >= self.max_duration_sec:
            self.get_logger().info('max_duration_sec reached; stopping recording')
            rclpy.shutdown()

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'failed to convert image: {exc}')
            return

        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f'failed to open video writer: {self.video_path}')

        self.writer.write(frame)
        ros_time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.csv_writer.writerow([self.frame_index, f'{ros_time_sec:.9f}', f'{time.time():.9f}'])
        self.csv_file.flush()
        self.frame_index += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not self.csv_file.closed:
            self.csv_file.close()
        self.get_logger().info(f'wrote {self.frame_index} frames to {self.video_path}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
