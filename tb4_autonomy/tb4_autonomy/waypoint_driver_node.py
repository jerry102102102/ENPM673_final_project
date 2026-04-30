from __future__ import annotations

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from tb4_autonomy.utils.geometry import clamp, normalize_angle, yaw_from_quaternion


DEFAULT_WORLD_WAYPOINTS = [
    (6.66, 0.30),
    (6.65, 0.05),
    (6.65, -0.23),
    (6.69, -0.58),
    (6.87, -0.92),
    (7.18, -1.11),
    (7.54, -1.17),
    (7.92, -1.07),
    (8.19, -0.82),
    (8.29, -0.46),
    (8.28, -0.10),
    (8.28, 0.30),
]


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    if not text.strip():
        return list(DEFAULT_WORLD_WAYPOINTS)
    waypoints: list[tuple[float, float]] = []
    for item in text.split(';'):
        if not item.strip():
            continue
        x_text, y_text = item.split(',', maxsplit=1)
        waypoints.append((float(x_text), float(y_text)))
    return waypoints


class WaypointDriverNode(Node):
    def __init__(self):
        super().__init__('waypoint_driver_node')
        self.cmd_vel_topic = str(self.declare_parameter('cmd_vel_topic', '/cmd_vel').value)
        self.odom_topic = str(self.declare_parameter('odom_topic', '/odom').value)
        self.map_origin_x = float(self.declare_parameter('map_origin_x', 6.66).value)
        self.map_origin_y = float(self.declare_parameter('map_origin_y', 0.327).value)
        self.map_origin_yaw = float(self.declare_parameter('map_origin_yaw', -1.57).value)
        world_waypoints = parse_waypoints(str(self.declare_parameter('waypoints', '').value))
        self.waypoints = [
            self._world_to_odom(point_x, point_y)
            for point_x, point_y in world_waypoints
        ]
        self.xy_tolerance = float(self.declare_parameter('xy_tolerance', 0.08).value)
        self.max_linear_x = float(self.declare_parameter('max_linear_x', 0.08).value)
        self.min_linear_x = float(self.declare_parameter('min_linear_x', 0.020).value)
        self.max_angular_z = float(self.declare_parameter('max_angular_z', 0.45).value)
        self.kp_heading = float(self.declare_parameter('kp_heading', 1.4).value)
        self.cross_track_gain = float(self.declare_parameter('cross_track_gain', 1.8).value)
        self.max_heading_bias_rad = math.radians(
            float(self.declare_parameter('max_heading_bias_deg', 18.0).value)
        )
        self.segment_advance_margin = float(self.declare_parameter('segment_advance_margin', 0.04).value)
        self.control_rate_hz = float(self.declare_parameter('control_rate_hz', 20.0).value)
        self.stop_at_end = bool(self.declare_parameter('stop_at_end', True).value)

        self.pose: tuple[float, float, float] | None = None
        self.index = 0
        self.done = False
        self.last_log_sec = 0.0
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_timer(1.0 / max(1.0, self.control_rate_hz), self.timer_callback)
        self.get_logger().info(
            f'following {len(self.waypoints)} path points with odom closed-loop on {self.cmd_vel_topic}'
        )
        self.get_logger().info(
            f'map->odom origin world=({self.map_origin_x:.3f},{self.map_origin_y:.3f},'
            f'{math.degrees(self.map_origin_yaw):.1f}deg) '
            f'first_odom_wp=({self.waypoints[0][0]:.3f},{self.waypoints[0][1]:.3f}) '
            f'last_odom_wp=({self.waypoints[-1][0]:.3f},{self.waypoints[-1][1]:.3f})'
        )

    def odom_callback(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (pos.x, pos.y, yaw)

    def _world_to_odom(self, world_x: float, world_y: float) -> tuple[float, float]:
        dx = world_x - self.map_origin_x
        dy = world_y - self.map_origin_y
        cos_yaw = math.cos(self.map_origin_yaw)
        sin_yaw = math.sin(self.map_origin_yaw)
        return (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        )

    def timer_callback(self) -> None:
        twist = Twist()
        if self.done:
            self.cmd_pub.publish(twist)
            return
        if self.pose is None:
            self.cmd_pub.publish(twist)
            return

        x, y, yaw = self.pose
        if self.index >= len(self.waypoints) - 1:
            self.done = True
            if self.stop_at_end:
                self.cmd_pub.publish(Twist())
            self.get_logger().info('waypoint path complete')
            return

        segment = self._current_segment(x, y)
        if segment is None:
            self.done = True
            self.cmd_pub.publish(Twist())
            self.get_logger().info('waypoint path complete')
            return

        (
            start_x,
            start_y,
            target_x,
            target_y,
            segment_length,
            along_track,
            cross_track,
            path_yaw,
        ) = segment

        heading_bias = clamp(
            -math.atan2(self.cross_track_gain * cross_track, 1.0),
            -self.max_heading_bias_rad,
            self.max_heading_bias_rad,
        )
        desired_yaw = normalize_angle(path_yaw + heading_bias)
        heading_error = normalize_angle(desired_yaw - yaw)
        twist.angular.z = clamp(self.kp_heading * heading_error, -self.max_angular_z, self.max_angular_z)

        heading_alignment = max(0.0, math.cos(heading_error))
        linear = self.max_linear_x * heading_alignment
        if abs(heading_error) < math.radians(75.0):
            linear = max(self.min_linear_x, linear)
        remaining = max(0.0, segment_length - along_track)
        twist.linear.x = min(self.max_linear_x, linear, max(self.min_linear_x, remaining))
        self.cmd_pub.publish(twist)
        self._log_status(
            x,
            y,
            yaw,
            target_x,
            target_y,
            path_yaw,
            cross_track,
            heading_bias,
            desired_yaw,
            heading_error,
            twist,
        )

    def _current_segment(self, x: float, y: float):
        while self.index < len(self.waypoints) - 1:
            start_x, start_y = self.waypoints[self.index]
            target_x, target_y = self.waypoints[self.index + 1]
            dx = target_x - start_x
            dy = target_y - start_y
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                self.index += 1
                continue

            tx = dx / segment_length
            ty = dy / segment_length
            rel_x = x - start_x
            rel_y = y - start_y
            along_track = rel_x * tx + rel_y * ty
            cross_track = -ty * rel_x + tx * rel_y

            if along_track >= segment_length - self.segment_advance_margin:
                self.get_logger().info(
                    f'segment {self.index + 1}/{len(self.waypoints) - 1} reached: '
                    f'x={x:.3f} y={y:.3f} cross_track={cross_track:.3f}'
                )
                self.index += 1
                continue

            path_yaw = math.atan2(dy, dx)
            return (
                start_x,
                start_y,
                target_x,
                target_y,
                segment_length,
                max(0.0, along_track),
                cross_track,
                path_yaw,
            )
        return None

    def _log_status(
        self,
        x: float,
        y: float,
        yaw: float,
        target_x: float,
        target_y: float,
        path_yaw: float,
        cross_track: float,
        heading_bias: float,
        desired_yaw: float,
        heading_error: float,
        twist: Twist,
    ) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self.last_log_sec < 1.0:
            return
        self.last_log_sec = now_sec
        self.get_logger().info(
            'path_follow '
            f'segment={self.index + 1}/{len(self.waypoints) - 1} '
            f'pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) '
            f'target=({target_x:.3f},{target_y:.3f}) '
            f'path_yaw={math.degrees(path_yaw):.1f}deg '
            f'cross_track={cross_track:.3f} '
            f'heading_bias={math.degrees(heading_bias):.1f}deg '
            f'desired_yaw={math.degrees(desired_yaw):.1f}deg '
            f'heading_error={math.degrees(heading_error):.1f}deg '
            f'cmd=({twist.linear.x:.3f},{twist.angular.z:.3f})'
        )

    def stop(self) -> None:
        if not rclpy.ok():
            return
        try:
            self.cmd_pub.publish(Twist())
        except RCLError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = WaypointDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
