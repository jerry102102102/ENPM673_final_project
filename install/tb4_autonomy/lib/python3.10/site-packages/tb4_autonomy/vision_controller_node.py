from __future__ import annotations

import time

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String

from tb4_autonomy.data_types import AutonomyState, DetectionResults, FrameContext
from tb4_autonomy.detectors.arrow_detector import ArrowDetector, ArrowDetectorConfig
from tb4_autonomy.detectors.horizon_detector import HorizonDetector
from tb4_autonomy.detectors.logo_detector import LogoDetector
from tb4_autonomy.detectors.moving_ball_detector import MovingBallDetector
from tb4_autonomy.motion_controller import MotionController, MotionControllerConfig
from tb4_autonomy.state_machine import StateMachine, StateMachineConfig
from tb4_autonomy.utils.geometry import yaw_from_quaternion
from tb4_autonomy.utils.image_tools import draw_detections, draw_status


class VisionControllerNode(Node):
    def __init__(self):
        super().__init__('vision_controller_node')

        self.image_topic = self._declare('image_topic', '/camera/image_raw/image_color')
        self.camera_info_topic = self._declare('camera_info_topic', '/camera/image_raw/camera_info')
        self.odom_topic = self._declare('odom_topic', '/odom')
        self.scan_topic = self._declare('scan_topic', '/scan')
        self.cmd_vel_topic = self._declare('cmd_vel_topic', '/cmd_vel')
        self.annotated_image_topic = self._declare('annotated_image_topic', '/debug/annotated_image')
        self.state_topic = self._declare('state_topic', '/autonomy/state')
        self.perf_topic = self._declare('perf_topic', '/autonomy/perf')
        self.dry_run = bool(self._declare('dry_run', True))
        self.control_rate_hz = float(self._declare('control_rate_hz', 20.0))
        self.horizon_ratio = float(self._declare('horizon_ratio', 0.5))

        motion_config = MotionControllerConfig(
            cruise_linear_x=float(self._declare('cruise_linear_x', 0.12)),
            track_linear_x=float(self._declare('track_linear_x', 0.08)),
            track_kp=float(self._declare('track_kp', 0.002)),
            max_angular_z=float(self._declare('max_angular_z', 0.65)),
            turn_kp=float(self._declare('turn_kp', 1.4)),
            turn_angle_rad=float(self._declare('turn_angle_rad', 1.57079632679)),
            back_turn_angle_rad=float(self._declare('back_turn_angle_rad', 3.14159265359)),
            turn_tolerance_rad=float(self._declare('turn_tolerance_rad', 0.05)),
            min_turn_angular_z=float(self._declare('min_turn_angular_z', 0.18)),
            search_linear_x=float(self._declare('search_linear_x', 0.06)),
            cooldown_linear_x=float(self._declare('cooldown_linear_x', 0.08)),
        )
        state_config = StateMachineConfig(
            logo_stop_s=float(self._declare('logo_stop_s', 3.0)),
            center_tolerance_px=float(self._declare('center_tolerance_px', 40.0)),
            target_bbox_area_ratio=float(self._declare('target_bbox_area_ratio', 0.08)),
            read_timeout_s=float(self._declare('read_timeout_s', 3.0)),
            cooldown_s=float(self._declare('cooldown_s', 1.5)),
        )
        arrow_config = ArrowDetectorConfig(
            threshold_method=str(self._declare('arrow_threshold_method', 'otsu')),
            hsv_v_max=int(self._declare('arrow_hsv_v_max', 80)),
            blur_kernel=int(self._declare('arrow_blur_kernel', 5)),
            morph_open_kernel=int(self._declare('arrow_morph_open_kernel', 3)),
            morph_close_kernel=int(self._declare('arrow_morph_close_kernel', 5)),
            dilate_kernel=int(self._declare('arrow_dilate_kernel', 7)),
            dilate_iterations=int(self._declare('arrow_dilate_iterations', 1)),
            min_area_ratio=float(self._declare('arrow_min_area_ratio', 0.005)),
            max_area_ratio=float(self._declare('arrow_max_area_ratio', 0.40)),
            min_aspect_ratio=float(self._declare('arrow_min_aspect_ratio', 0.5)),
            max_aspect_ratio=float(self._declare('arrow_max_aspect_ratio', 2.0)),
            min_black_pixel_ratio=float(self._declare('arrow_min_black_pixel_ratio', 0.03)),
            warp_width=int(self._declare('arrow_warp_width', 300)),
            warp_height=int(self._declare('arrow_warp_height', 300)),
            inner_crop_margin_ratio=float(self._declare('arrow_inner_crop_margin_ratio', 0.12)),
            history_size=int(self._declare('arrow_history_size', 5)),
            min_stable_count=int(self._declare('arrow_min_stable_count', 4)),
            process_width=int(self._declare('arrow_process_width', 960)),
            bbox_padding_ratio=float(self._declare('arrow_bbox_padding_ratio', 0.20)),
            min_arrow_area_ratio=float(self._declare('arrow_min_arrow_area_ratio', 0.01)),
            paper_v_min=int(self._declare('arrow_paper_v_min', 110)),
            paper_s_max=int(self._declare('arrow_paper_s_max', 90)),
            paper_min_area_ratio=float(self._declare('arrow_paper_min_area_ratio', 0.001)),
            paper_max_area_ratio=float(self._declare('arrow_paper_max_area_ratio', 0.25)),
            paper_min_aspect_ratio=float(self._declare('arrow_paper_min_aspect_ratio', 0.3)),
            paper_max_aspect_ratio=float(self._declare('arrow_paper_max_aspect_ratio', 12.0)),
        )

        self.bridge = CvBridge()
        self.motion = MotionController(motion_config)
        self.state_machine = StateMachine(state_config)
        self.detectors = {
            'arrow': ArrowDetector(arrow_config),
            'logo': LogoDetector(),
            'moving_ball': MovingBallDetector(),
            'horizon': HorizonDetector(self.horizon_ratio),
        }

        self.latest_results = DetectionResults()
        self.latest_camera_info: CameraInfo | None = None
        self.latest_scan: LaserScan | None = None
        self.latest_yaw = 0.0
        self.latest_odom_linear_x = 0.0
        self.latest_image_width = 0
        self.latest_image_height = 0
        self.latest_frame_ms = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.annotated_image_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.perf_pub = self.create_publisher(String, self.perf_topic, 10)

        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data)

        timer_period = 1.0 / max(1.0, self.control_rate_hz)
        self.create_timer(timer_period, self.control_timer_callback)

        self.get_logger().info(
            f'vision_controller_node started image={self.image_topic} dry_run={self.dry_run}'
        )

    def _declare(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.latest_odom_linear_x = msg.twist.twist.linear.x

    def image_callback(self, msg: Image) -> None:
        start = time.perf_counter()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'failed to convert image: {exc}')
            return

        height, width = frame.shape[:2]
        self.latest_image_width = width
        self.latest_image_height = height
        context = FrameContext(
            stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            frame_id=msg.header.frame_id,
            image_width=width,
            image_height=height,
            odom_yaw=self.latest_yaw,
            odom_linear_x=self.latest_odom_linear_x,
            camera_info=self.latest_camera_info,
            scan=self.latest_scan,
        )

        results = DetectionResults()
        timings_ms: dict[str, float] = {}
        for name, detector in self.detectors.items():
            detector_start = time.perf_counter()
            result = detector.detect(frame, context)
            timings_ms[name] = (time.perf_counter() - detector_start) * 1000.0
            if name == 'arrow':
                results.arrow = result
            elif name == 'logo':
                results.logo = result
            elif name == 'moving_ball':
                results.moving_ball = result
            elif name == 'horizon':
                results.horizon = result

        results.timings_ms = timings_ms
        self.latest_results = results
        self.latest_frame_ms = (time.perf_counter() - start) * 1000.0

        annotated = frame.copy()
        draw_detections(annotated, results)
        draw_status(
            annotated,
            [
                f'state: {self.state_machine.state.value}',
                f'dry_run: {self.dry_run}',
                f'frame: {self.latest_frame_ms:.1f} ms',
                f'yaw: {self.latest_yaw:.2f} rad',
                self._arrow_status_line(results),
            ],
        )
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

        perf = String()
        timing_parts = [f'{key}={value:.1f}ms' for key, value in timings_ms.items()]
        perf.data = f'frame={self.latest_frame_ms:.1f}ms ' + ' '.join(timing_parts)
        self.perf_pub.publish(perf)

    def control_timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        turn_complete = False
        if self.state_machine.state == AutonomyState.EXECUTE_TURN:
            _, turn_complete = self.motion.update_turn(self.latest_yaw)

        output = self.state_machine.update(self.latest_results, now_sec, turn_complete)
        if output.entered_state and output.state == AutonomyState.EXECUTE_TURN:
            self.motion.start_turn(output.turn_direction or 'unknown', self.latest_yaw, output.turn_angle_rad)

        twist = self._twist_for_state(output.state)
        if self.dry_run:
            twist = self.motion.stop()

        self.cmd_pub.publish(twist)

        state = String()
        state.data = output.state.value
        self.state_pub.publish(state)

    def _twist_for_state(self, state: AutonomyState) -> Twist:
        if state in (
            AutonomyState.IDLE,
            AutonomyState.LOGO_STOP,
            AutonomyState.BALL_STOP,
            AutonomyState.FINISHED,
            AutonomyState.READ_ARROW,
        ):
            return self.motion.stop()

        if state == AutonomyState.SEARCH_SIGN:
            return self.motion.search()

        if state == AutonomyState.ARROW_COOLDOWN:
            return self.motion.cooldown_forward()

        if state == AutonomyState.EXECUTE_TURN:
            twist, _ = self.motion.update_turn(self.latest_yaw)
            return twist

        if state == AutonomyState.ALIGN_TO_SIGN and self.latest_results.arrow is not None:
            center_x = self.latest_results.arrow.box.center[0]
            error_px = center_x - self.latest_image_width / 2.0
            return self.motion.align_x_error(error_px, self.latest_image_width)

        if state in (AutonomyState.TRACK_ARROW, AutonomyState.APPROACH_SIGN) and self.latest_results.arrow is not None:
            center_x = self.latest_results.arrow.box.center[0]
            error_px = center_x - self.latest_image_width / 2.0
            return self.motion.track_x_error(error_px, self.latest_image_width)

        return self.motion.cruise()

    def _arrow_status_line(self, results: DetectionResults) -> str:
        if results.arrow is None:
            return 'arrow: none'
        arrow = results.arrow
        return (
            f'arrow: stable={arrow.direction} raw={arrow.raw_direction} '
            f'area={arrow.area_ratio:.3f} err={arrow.center_error_px:.0f}px'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisionControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(node.motion.stop())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()