#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class PreviewVideoRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('tb4_preview_video_recorder')
        self.bridge = CvBridge()
        self.topic = args.topic or f'/{args.robot}/oakd/rgb/preview/image_raw'
        self.output_path = Path(args.output).expanduser().resolve()
        self.fps = float(args.fps)
        self.duration_sec = float(args.duration) if args.duration is not None else None
        self.show = bool(args.show)
        self.frame_count = 0
        self.writer: cv2.VideoWriter | None = None
        self.started_ns: int | None = None
        self.stop_requested = False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_subscription(Image, self.topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to {self.topic}')
        self.get_logger().info(f'Writing video to {self.output_path}')
        if self.duration_sec is not None:
            self.get_logger().info(f'Recording duration limit: {self.duration_sec:.1f}s')

    def request_stop(self) -> None:
        self.stop_requested = True

    def image_callback(self, msg: Image) -> None:
        if self.stop_requested:
            return

        if self.started_ns is None:
            self.started_ns = self.get_clock().now().nanoseconds

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f'Could not open video writer for {self.output_path}')
            self.get_logger().info(f'Video initialized: {width}x{height} @ {self.fps:.1f} fps')

        self.writer.write(frame)
        self.frame_count += 1

        if self.show:
            cv2.imshow('TB4 OAK-D preview recorder', frame)
            if cv2.waitKey(1) == ord('q'):
                self.request_stop()

        if self.duration_sec is not None and self.started_ns is not None:
            elapsed_sec = (self.get_clock().now().nanoseconds - self.started_ns) * 1e-9
            if elapsed_sec >= self.duration_sec:
                self.request_stop()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.show:
            cv2.destroyAllWindows()
        self.get_logger().info(f'Saved {self.frame_count} frames to {self.output_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Record a TurtleBot4 OAK-D preview image topic to an MP4 file.'
    )
    parser.add_argument('robot', nargs='?', default='tb4_1', help='Robot name, e.g. tb4_1. Default: tb4_1')
    parser.add_argument(
        '--topic',
        default='',
        help='Image topic override. Default: /<robot>/oakd/rgb/preview/image_raw',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Output .mp4 path. Default: videos/<robot>_preview_<timestamp>.mp4',
    )
    parser.add_argument('--fps', type=float, default=15.0, help='Output video FPS. Default: 15')
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Seconds to record. Default: record until Ctrl-C',
    )
    parser.add_argument('--show', action='store_true', help='Show a live OpenCV preview window.')
    args = parser.parse_args()

    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'videos/{args.robot}_preview_{timestamp}.mp4'
    if args.fps <= 0:
        parser.error('--fps must be positive')
    if args.duration is not None and args.duration <= 0:
        parser.error('--duration must be positive')
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = PreviewVideoRecorder(args)

    def handle_signal(_signum, _frame):
        node.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
