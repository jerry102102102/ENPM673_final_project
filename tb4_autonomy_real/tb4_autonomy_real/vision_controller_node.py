from __future__ import annotations

import math
import time

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import String

from tb4_autonomy_real.arrow_smooth_arc_controller import ArrowSmoothArcConfig, ArrowSmoothArcController
from tb4_autonomy_real.data_types import AutonomyState, DetectionResults, FrameContext
from tb4_autonomy_real.detectors.arrow_detector import ArrowDetector, ArrowDetectorConfig
from tb4_autonomy_real.detectors.horizon_detector import HorizonDetector
from tb4_autonomy_real.detectors.logo_detector import LogoDetector
from tb4_autonomy_real.detectors.moving_ball_detector import MovingBallDetector
from tb4_autonomy_real.detectors.static_ball_detector import StaticObstacleDetector
from tb4_autonomy_real.utils.geometry import yaw_from_quaternion
from tb4_autonomy_real.utils.image_tools import draw_detections, draw_status


class VisionControllerNode(Node):
    def __init__(self):
        super().__init__('real_vision_controller_node')

        self.image_topic = self._declare('image_topic', '/camera/image_raw/image_color')
        self.image_is_compressed = self._declare_bool('image_is_compressed', False)
        self.camera_info_topic = self._declare('camera_info_topic', '/camera/image_raw/camera_info')
        self.odom_topic = self._declare('odom_topic', '/odom')
        self.scan_topic = self._declare('scan_topic', '/scan')
        self.cmd_vel_topic = self._declare('cmd_vel_topic', '/cmd_vel')
        self.cmd_vel_stamped = self._declare_bool('cmd_vel_stamped', False)
        self.annotated_image_topic = self._declare('annotated_image_topic', '/debug/annotated_image')
        self.state_topic = self._declare('state_topic', '/autonomy/state')
        self.perf_topic = self._declare('perf_topic', '/autonomy/perf')
        self.dry_run = self._declare_bool('dry_run', True)
        self.control_rate_hz = float(self._declare('control_rate_hz', 20.0))
        self.horizon_ratio = float(self._declare('horizon_ratio', 0.5))
        self.horizon_roi_top_ratio = float(self._declare('horizon_roi_top_ratio', 0.42))
        self.horizon_roi_bottom_ratio = float(self._declare('horizon_roi_bottom_ratio', 0.60))

        self.logo_stop_s = float(self._declare('logo_stop_s', 3.0))
        self.logo_confirm_frames = int(self._declare('logo_confirm_frames', 5))
        self.logo_stop_until: float | None = None
        self.logo_armed = True
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
            min_area_px=float(self._declare('arrow_min_area_px', 0.0)),
            max_area_px=float(self._declare('arrow_max_area_px', 0.0)),
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
            fallback_bbox_padding_ratio=float(self._declare('arrow_fallback_bbox_padding_ratio', 0.0)),
            use_paper_candidate=self._declare_bool('arrow_use_paper_candidate', True),
            black_frame_enable=self._declare_bool('arrow_black_frame_enable', True),
            black_frame_min_area_ratio=float(self._declare('arrow_black_frame_min_area_ratio', 0.002)),
            black_frame_max_area_ratio=float(self._declare('arrow_black_frame_max_area_ratio', 0.30)),
            black_frame_approx_epsilon_ratio=float(self._declare('arrow_black_frame_approx_epsilon_ratio', 0.04)),
            black_frame_min_border_density=float(self._declare('arrow_black_frame_min_border_density', 0.08)),
            black_frame_max_inner_density=float(self._declare('arrow_black_frame_max_inner_density', 0.35)),
            black_frame_min_border_inner_contrast=float(
                self._declare('arrow_black_frame_min_border_inner_contrast', 0.03)
            ),
            black_frame_border_band_ratio=float(self._declare('arrow_black_frame_border_band_ratio', 0.12)),
            black_frame_close_kernel=int(self._declare('arrow_black_frame_close_kernel', 5)),
            black_frame_dilate_kernel=int(self._declare('arrow_black_frame_dilate_kernel', 3)),
            black_frame_min_score=float(self._declare('arrow_black_frame_min_score', 0.25)),
            black_frame_refine_with_white_paper=self._declare_bool(
                'arrow_black_frame_refine_with_white_paper', True
            ),
            black_frame_white_roi_padding_ratio=float(
                self._declare('arrow_black_frame_white_roi_padding_ratio', 0.18)
            ),
            black_frame_white_min_area_ratio=float(self._declare('arrow_black_frame_white_min_area_ratio', 0.002)),
            black_frame_white_max_area_ratio=float(self._declare('arrow_black_frame_white_max_area_ratio', 0.35)),
            black_frame_white_max_area_expand_ratio=float(
                self._declare('arrow_black_frame_white_max_area_expand_ratio', 1.8)
            ),
            black_frame_white_min_overlap_ratio=float(
                self._declare('arrow_black_frame_white_min_overlap_ratio', 0.25)
            ),
            black_frame_white_center_tolerance_ratio=float(
                self._declare('arrow_black_frame_white_center_tolerance_ratio', 0.35)
            ),
            black_frame_white_approx_epsilon_ratio=float(
                self._declare('arrow_black_frame_white_approx_epsilon_ratio', 0.035)
            ),
            black_frame_white_open_kernel=int(self._declare('arrow_black_frame_white_open_kernel', 3)),
            black_frame_white_close_kernel=int(self._declare('arrow_black_frame_white_close_kernel', 5)),
            merge_black_fragments_fallback=self._declare_bool('arrow_merge_black_fragments_fallback', False),
            black_fragment_group_radius_px=float(self._declare('arrow_black_fragment_group_radius_px', 55.0)),
            black_fragment_seed_max_bottom_ratio=float(
                self._declare('arrow_black_fragment_seed_max_bottom_ratio', 0.0)
            ),
            black_fragment_expand_x_px=float(self._declare('arrow_black_fragment_expand_x_px', 0.0)),
            black_fragment_expand_y_px=float(self._declare('arrow_black_fragment_expand_y_px', 0.0)),
            black_fragment_min_box_w_px=float(self._declare('arrow_black_fragment_min_box_w_px', 0.0)),
            black_fragment_min_box_h_px=float(self._declare('arrow_black_fragment_min_box_h_px', 0.0)),
            min_arrow_area_ratio=float(self._declare('arrow_min_arrow_area_ratio', 0.01)),
            paper_v_min=int(self._declare('arrow_paper_v_min', 130)),
            paper_s_max=int(self._declare('arrow_paper_s_max', 90)),
            paper_min_area_ratio=float(self._declare('arrow_paper_min_area_ratio', 0.001)),
            paper_max_area_ratio=float(self._declare('arrow_paper_max_area_ratio', 0.25)),
            paper_min_area_px=float(self._declare('arrow_paper_min_area_px', 0.0)),
            paper_max_area_px=float(self._declare('arrow_paper_max_area_px', 0.0)),
            paper_min_aspect_ratio=float(self._declare('arrow_paper_min_aspect_ratio', 0.3)),
            paper_max_aspect_ratio=float(self._declare('arrow_paper_max_aspect_ratio', 12.0)),
            floor_roi_min_y_ratio=float(self._declare('arrow_floor_roi_min_y_ratio', 0.45)),
            candidate_max_center_error_ratio=float(self._declare('arrow_candidate_max_center_error_ratio', 0.40)),
            min_candidate_bottom_ratio=float(self._declare('arrow_min_candidate_bottom_ratio', 0.45)),
            max_candidate_height_ratio=float(self._declare('arrow_max_candidate_height_ratio', 0.58)),
            max_candidate_width_ratio=float(self._declare('arrow_max_candidate_width_ratio', 0.75)),
            max_candidate_area_ratio=float(self._declare('arrow_max_candidate_area_ratio', 0.18)),
            min_candidate_area_px=float(self._declare('arrow_min_candidate_area_px', 0.0)),
            max_candidate_area_px=float(self._declare('arrow_max_candidate_area_px', 0.0)),
            reject_back_direction=self._declare_bool('arrow_reject_back_direction', True),
            max_valid_bbox_width_ratio=float(self._declare('arrow_max_valid_bbox_width_ratio', 0.70)),
            max_valid_bbox_height_ratio=float(self._declare('arrow_max_valid_bbox_height_ratio', 0.70)),
            max_valid_final_area_ratio=float(self._declare('arrow_max_valid_final_area_ratio', 0.16)),
            min_valid_final_area_ratio=float(self._declare('arrow_min_valid_final_area_ratio', 0.001)),
            max_border_touch_ratio=float(self._declare('arrow_max_border_touch_ratio', 0.03)),
            max_valid_paper_aspect_ratio=float(self._declare('arrow_max_valid_paper_aspect_ratio', 4.0)),
            min_valid_paper_aspect_ratio=float(self._declare('arrow_min_valid_paper_aspect_ratio', 0.25)),
            min_history_confidence=float(self._declare('arrow_min_history_confidence', 0.45)),
            use_axis_direction=self._declare_bool('arrow_use_axis_direction', False),
            use_base_tip_direction=self._declare_bool('arrow_use_base_tip_direction', False),
            use_paper_orientation_heading=self._declare_bool('arrow_use_paper_orientation_heading', False),
            paper_heading_forward_angle_rad=float(self._declare('arrow_paper_heading_forward_angle_rad', 1.57079632679)),
            paper_heading_use_previous_when_ambiguous=self._declare_bool(
                'arrow_paper_heading_use_previous_when_ambiguous', True
            ),
            paper_heading_ambiguity_margin_rad=float(
                self._declare('arrow_paper_heading_ambiguity_margin_rad', 0.20)
            ),
            min_arrow_presence_confidence=float(self._declare('arrow_min_arrow_presence_confidence', 0.35)),
            min_arrow_component_density=float(self._declare('arrow_min_arrow_component_density', 0.25)),
            min_arrow_component_solidity=float(self._declare('arrow_min_arrow_component_solidity', 0.42)),
            min_arrow_component_compactness=float(self._declare('arrow_min_arrow_component_compactness', 0.055)),
            max_arrow_component_area_ratio=float(self._declare('arrow_max_arrow_component_area_ratio', 0.095)),
        )
        smooth_arc_config = ArrowSmoothArcConfig(
            min_confidence=float(self._declare('arrow_smooth_arc_controller.min_confidence', 0.45)),
            acquire_area_threshold=float(self._declare('arrow_smooth_arc_controller.acquire_area_threshold', 0.020)),
            close_area_threshold=float(self._declare('arrow_smooth_arc_controller.close_area_threshold', 0.24)),
            close_bottom_ratio=float(self._declare('arrow_smooth_arc_controller.close_bottom_ratio', 0.995)),
            focal_px=float(self._declare('arrow_smooth_arc_controller.focal_px', 600.0)),
            heading_sign=float(self._declare('arrow_smooth_arc_controller.heading_sign', 1.0)),
            heading_scale=float(self._declare('arrow_smooth_arc_controller.heading_scale', 0.15)),
            heading_oversteer_deg=float(self._declare('arrow_smooth_arc_controller.heading_oversteer_deg', 2.0)),
            latched_yaw_alpha=float(self._declare('arrow_smooth_arc_controller.latched_yaw_alpha', 0.20)),
            previous_heading_pull_gain=float(
                self._declare('arrow_smooth_arc_controller.previous_heading_pull_gain', 0.30)
            ),
            previous_heading_pull_decay_sec=float(
                self._declare('arrow_smooth_arc_controller.previous_heading_pull_decay_sec', 2.20)
            ),
            previous_heading_pull_start_delta_deg=float(
                self._declare('arrow_smooth_arc_controller.previous_heading_pull_start_delta_deg', 10.0)
            ),
            previous_heading_pull_max_angular_z=float(
                self._declare('arrow_smooth_arc_controller.previous_heading_pull_max_angular_z', 0.050)
            ),
            latched_heading_confidence_min=float(
                self._declare('arrow_smooth_arc_controller.latched_heading_confidence_min', 0.55)
            ),
            latched_arrow_presence_confidence_min=float(
                self._declare('arrow_smooth_arc_controller.latched_arrow_presence_confidence_min', 0.45)
            ),
            yaw_latch_alpha=float(self._declare('arrow_smooth_arc_controller.yaw_latch_alpha', 0.15)),
            heading_sample_window=int(self._declare('arrow_smooth_arc_controller.heading_sample_window', 8)),
            heading_sample_min_count=int(self._declare('arrow_smooth_arc_controller.heading_sample_min_count', 3)),
            heading_sample_tolerance_deg=float(
                self._declare('arrow_smooth_arc_controller.heading_sample_tolerance_deg', 30.0)
            ),
            min_heading_confidence=float(self._declare('arrow_smooth_arc_controller.min_heading_confidence', 0.65)),
            min_arrow_presence_confidence=float(
                self._declare('arrow_smooth_arc_controller.min_arrow_presence_confidence', 0.45)
            ),
            heading_update_max_area_ratio=float(
                self._declare('arrow_smooth_arc_controller.heading_update_max_area_ratio', 0.09)
            ),
            heading_update_max_bottom_ratio=float(
                self._declare('arrow_smooth_arc_controller.heading_update_max_bottom_ratio', 0.95)
            ),
            center_sign=float(self._declare('arrow_smooth_arc_controller.center_sign', -1.0)),
            center_capture_threshold_px=float(self._declare('arrow_smooth_arc_controller.center_capture_threshold_px', 90.0)),
            min_center_bias_gain=float(self._declare('arrow_smooth_arc_controller.min_center_bias_gain', 0.10)),
            max_center_bias_gain=float(self._declare('arrow_smooth_arc_controller.max_center_bias_gain', 0.60)),
            max_center_bias_deg=float(self._declare('arrow_smooth_arc_controller.max_center_bias_deg', 6.0)),
            kp_yaw=float(self._declare('arrow_smooth_arc_controller.kp_yaw', 1.2)),
            kp_center=float(self._declare('arrow_smooth_arc_controller.kp_center', 0.45)),
            kp_heading=float(self._declare('arrow_smooth_arc_controller.kp_heading', 0.45)),
            max_angular_z=float(self._declare('arrow_smooth_arc_controller.max_angular_z', 0.18)),
            max_angular_accel=float(self._declare('arrow_smooth_arc_controller.max_angular_accel', 0.35)),
            yaw_error_deadband_deg=float(self._declare('arrow_smooth_arc_controller.yaw_error_deadband_deg', 3.0)),
            angular_lowpass_alpha=float(self._declare('arrow_smooth_arc_controller.angular_lowpass_alpha', 0.18)),
            heading_confidence_soft_min=float(
                self._declare('arrow_smooth_arc_controller.heading_confidence_soft_min', 0.40)
            ),
            heading_confidence_full=float(self._declare('arrow_smooth_arc_controller.heading_confidence_full', 0.80)),
            arrow_presence_confidence_soft_min=float(
                self._declare('arrow_smooth_arc_controller.arrow_presence_confidence_soft_min', 0.35)
            ),
            arrow_presence_confidence_full=float(
                self._declare('arrow_smooth_arc_controller.arrow_presence_confidence_full', 0.80)
            ),
            track_speed=float(self._declare('arrow_smooth_arc_controller.track_speed', 0.030)),
            slow_track_speed=float(self._declare('arrow_smooth_arc_controller.slow_track_speed', 0.022)),
            slow_yaw_error_deg=float(self._declare('arrow_smooth_arc_controller.slow_yaw_error_deg', 25.0)),
            slow_center_error_px=float(self._declare('arrow_smooth_arc_controller.slow_center_error_px', 100.0)),
            wait_linear_speed=float(self._declare('arrow_smooth_arc_controller.wait_linear_speed', 0.025)),
            pass_speed=float(self._declare('arrow_smooth_arc_controller.pass_speed', 0.035)),
            pass_time_sec=float(self._declare('arrow_smooth_arc_controller.pass_time_sec', 0.15)),
            missing_detection_hold_sec=float(
                self._declare('arrow_smooth_arc_controller.missing_detection_hold_sec', 0.20)
            ),
            execute_latched_on_lost=self._declare_bool('arrow_smooth_arc_controller.execute_latched_on_lost', True),
            pass_max_heading_error_deg=float(
                self._declare('arrow_smooth_arc_controller.pass_max_heading_error_deg', 6.0)
            ),
            finish_heading_timeout_sec=float(
                self._declare('arrow_smooth_arc_controller.finish_heading_timeout_sec', 1.5)
            ),
            finish_heading_speed=float(self._declare('arrow_smooth_arc_controller.finish_heading_speed', 0.015)),
            finish_heading_kp=float(self._declare('arrow_smooth_arc_controller.finish_heading_kp', 0.40)),
            finish_heading_max_angular_z=float(
                self._declare('arrow_smooth_arc_controller.finish_heading_max_angular_z', 0.08)
            ),
            finish_center_kp=float(self._declare('arrow_smooth_arc_controller.finish_center_kp', 0.35)),
            finish_center_max_bias_deg=float(
                self._declare('arrow_smooth_arc_controller.finish_center_max_bias_deg', 5.0)
            ),
            finish_center_tolerance_px=float(
                self._declare('arrow_smooth_arc_controller.finish_center_tolerance_px', 45.0)
            ),
            min_track_time_sec=float(self._declare('arrow_smooth_arc_controller.min_track_time_sec', 0.8)),
            close_camera_bottom_distance_in=float(
                self._declare('arrow_smooth_arc_controller.close_camera_bottom_distance_in', 18.5)
            ),
            close_arrow_bottom_distance_in=float(
                self._declare('arrow_smooth_arc_controller.close_arrow_bottom_distance_in', 22.375)
            ),
            post_close_min_travel_m=float(self._declare('arrow_smooth_arc_controller.post_close_min_travel_m', 0.0)),
            post_close_speed=float(self._declare('arrow_smooth_arc_controller.post_close_speed', 0.055)),
            post_close_timeout_sec=float(self._declare('arrow_smooth_arc_controller.post_close_timeout_sec', 3.0)),
            active_target_area_drop_ratio=float(
                self._declare('arrow_smooth_arc_controller.active_target_area_drop_ratio', 0.70)
            ),
            active_target_bottom_drop_ratio=float(
                self._declare('arrow_smooth_arc_controller.active_target_bottom_drop_ratio', 0.08)
            ),
            debug_log=self._declare_bool('arrow_smooth_arc_controller.debug_log', True),
        )

        self.bridge = CvBridge()
        self.arrow_controller = ArrowSmoothArcController(smooth_arc_config)
        self.previous_arrow_state = self.arrow_controller.state
        self.latest_controller_debug: dict[str, object] = {}
        self.detectors = {
            'arrow': ArrowDetector(arrow_config),
            'logo': LogoDetector(detect_threshold=self.logo_confirm_frames),
            'moving_ball': MovingBallDetector(),
            'static_ball': StaticObstacleDetector(),
            'horizon': HorizonDetector(
                self.horizon_ratio,
                roi_top_ratio=self.horizon_roi_top_ratio,
                roi_bottom_ratio=self.horizon_roi_bottom_ratio,
            ),
        }

        self.latest_results = DetectionResults()
        self.latest_camera_info: CameraInfo | None = None
        self.latest_scan: LaserScan | None = None
        self.latest_yaw: float | None = None
        self.latest_odom_xy: tuple[float, float] | None = None
        self.latest_odom_linear_x = 0.0
        self.latest_image_width = 0
        self.latest_image_height = 0
        self.latest_frame_ms = 0.0

        cmd_msg_type = TwistStamped if self.cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_msg_type, self.cmd_vel_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.annotated_image_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.perf_pub = self.create_publisher(String, self.perf_topic, 10)

        image_msg_type = CompressedImage if self.image_is_compressed else Image
        self.create_subscription(image_msg_type, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data)

        timer_period = 1.0 / max(1.0, self.control_rate_hz)
        self.create_timer(timer_period, self.control_timer_callback)

        self.get_logger().info(
            f'real_vision_controller_node started image={self.image_topic} '
            f'compressed={self.image_is_compressed} dry_run={self.dry_run}'
        )

    def _declare(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _declare_bool(self, name: str, default: bool) -> bool:
        value = self._declare(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.latest_odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.latest_odom_linear_x = msg.twist.twist.linear.x

    def image_callback(self, msg: Image | CompressedImage) -> None:
        start = time.perf_counter()
        try:
            frame = self._message_to_bgr_frame(msg)
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
            elif name == 'static_ball':
                results.static_ball = result
            elif name == 'horizon':
                results.horizon = result

        results.timings_ms = timings_ms
        self.latest_results = results
        self.latest_frame_ms = (time.perf_counter() - start) * 1000.0
        if self.arrow_controller.config.debug_log:
            self.get_logger().info(self._format_detector_debug(results.arrow))

        annotated = frame.copy()
        draw_detections(annotated, results)
        draw_status(
            annotated,
            [
                f"STATE: {self._debug_value('state', self.arrow_controller.state.value)}",
                f'dry_run: {self.dry_run}',
                f'frame: {self.latest_frame_ms:.1f} ms',
                f'yaw: {0.0 if self.latest_yaw is None else self.latest_yaw:.2f} rad',
                self._logo_status_line(results),
                self._static_ball_status_line(results),
                self._arrow_status_line(results),
                self._controller_status_line('CENTER_ERR', 'center_error_px', '.0f', 'px'),
                self._controller_status_line('CENTER_BIAS', 'center_bias_deg', '.1f', 'deg'),
                self._controller_status_line('HEAD_ERR', 'heading_error_rad', '.2f', 'rad'),
                self._control_mode_status_line(),
                self._controller_status_line('HEAD_W', 'heading_weight', '.2f', ''),
                self._controller_status_line('PREV_W', 'previous_heading_pull_weight', '.2f', ''),
                self._controller_status_line('PREV_T', 'previous_heading_pull_term', '.3f', ''),
                self._controller_status_line('C_TERM', 'center_term', '.3f', ''),
                self._controller_status_line('H_TERM', 'heading_term', '.3f', ''),
                self._controller_status_line('ANG_TGT', 'angular_target', '.3f', ''),
                self._controller_status_line('YAW_ERR', 'yaw_error_deg', '.1f', 'deg'),
                self._controller_status_line('AREA', 'area_ratio', '.3f', ''),
                self._cmd_status_line(),
            ],
        )
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

        perf = String()
        timing_parts = [f'{key}={value:.1f}ms' for key, value in timings_ms.items()]
        perf.data = f'frame={self.latest_frame_ms:.1f}ms ' + ' '.join(timing_parts)
        self.perf_pub.publish(perf)

    def _message_to_bgr_frame(self, msg: Image | CompressedImage):
        if self.image_is_compressed:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('cv2.imdecode returned None for compressed image')
            return frame
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def control_timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        state = self.arrow_controller.state

        if self.latest_results.has_moving_ball or self.latest_results.has_static_ball:
            twist = Twist()
            state = AutonomyState.BALL_STOP
            self.arrow_controller.reset()
            reason = 'moving_ball_stop' if self.latest_results.has_moving_ball else 'static_ball_stop'
            self._set_stop_debug(state, now_sec, twist, reason)
        elif self.logo_stop_until is not None and now_sec < self.logo_stop_until:
            twist = Twist()
            state = AutonomyState.LOGO_STOP
            self.arrow_controller.reset()
            self._set_stop_debug(state, now_sec, twist, 'logo_stop_hold')
        elif self.logo_stop_until is not None:
            self.logo_stop_until = None
            twist, state = self._arrow_twist(now_sec)
        elif self.latest_results.logo is not None and self.logo_armed:
            twist = Twist()
            state = AutonomyState.LOGO_STOP
            self.logo_stop_until = now_sec + self.logo_stop_s
            self.logo_armed = False
            self.arrow_controller.reset()
            self._set_stop_debug(state, now_sec, twist, 'logo_confirmed_stop')
        else:
            twist, state = self._arrow_twist(now_sec)

        if self.dry_run:
            twist = Twist()

        self._publish_cmd_vel(twist)

        state_msg = String()
        state_msg.data = state.value
        self.state_pub.publish(state_msg)

    def _publish_cmd_vel(self, twist: Twist) -> None:
        if not self.cmd_vel_stamped:
            self.cmd_pub.publish(twist)
            return

        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'base_link'
        stamped.twist = twist
        self.cmd_pub.publish(stamped)

    def _arrow_twist(self, now_sec: float) -> tuple[Twist, AutonomyState]:
        output = self.arrow_controller.update(
            self.latest_results.arrow,
            self.latest_image_width,
            self.latest_image_height,
            now_sec,
            self.latest_yaw,
            self.latest_odom_xy,
        )
        arrow_detector = self.detectors.get('arrow')
        if self.previous_arrow_state == AutonomyState.PASS_TO_NEXT and output.current_state == AutonomyState.WAIT_FOR_TARGET:
            arrow_detector.reset_history()
        if str(output.debug_info.get('transition_reason', '')).startswith('acquired_'):
            arrow_detector.reset_history()
        self.previous_arrow_state = output.current_state
        self.latest_controller_debug = output.debug_info
        if self.arrow_controller.config.debug_log:
            self.get_logger().info(self._format_debug_info(output.debug_info))
        return output.twist, output.current_state

    def _set_stop_debug(self, state: AutonomyState, now_sec: float, twist: Twist, reason: str) -> None:
        remaining = 0.0
        if self.logo_stop_until is not None:
            remaining = max(0.0, self.logo_stop_until - now_sec)
        logo_detector = self.detectors.get('logo')
        logo_count = getattr(logo_detector, 'detect_count', 0)
        logo_threshold = getattr(logo_detector, 'detect_threshold', self.logo_confirm_frames)
        self.latest_controller_debug = {
            'state': state.value,
            'control_mode': reason,
            'linear_x': twist.linear.x,
            'angular_z': twist.angular.z,
            'angular_target': twist.angular.z,
            'angular_smoothed': twist.angular.z,
            'logo_stop_remaining_sec': remaining,
            'logo_detect_count': logo_count,
            'logo_confirm_frames': logo_threshold,
            'moving_ball_detected': self.latest_results.moving_ball is not None,
            'static_ball_detected': self.latest_results.static_ball is not None,
            'transition_reason': reason,
        }

    def _arrow_status_line(self, results: DetectionResults) -> str:
        if results.arrow is None:
            return 'arrow: none'
        arrow = results.arrow
        return (
            f'arrow: stable={arrow.direction} raw={arrow.raw_direction} '
            f'area={arrow.area_ratio:.3f} err={arrow.center_error_px:.0f}px '
            f'src={arrow.heading_source} '
            f'valid={arrow.heading_valid} '
            f'head={0.0 if arrow.heading_error_rad is None else arrow.heading_error_rad:.2f} '
            f'angle={0.0 if arrow.heading_angle_deg is None else arrow.heading_angle_deg:.1f}deg '
            f'exist={arrow.arrow_presence_confidence:.2f} '
            f'paper_dbg={0.0 if arrow.paper_heading_angle_rad is None else math.degrees(arrow.paper_heading_angle_rad):.1f}deg '
            f'black={arrow.black_arrow_direction}:{arrow.black_arrow_confidence:.2f} '
            f'tpl={arrow.template_direction}:{arrow.template_dominance:.2f} '
            f'axis_dbg={arrow.axis_direction}:{arrow.axis_confidence:.2f} '
            f'box=({arrow.box.x},{arrow.box.y},{arrow.box.w},{arrow.box.h}) '
            f'black_px={arrow.black_pixel_ratio:.3f}'
        )

    def _logo_status_line(self, results: DetectionResults) -> str:
        logo_detector = self.detectors.get('logo')
        count = getattr(logo_detector, 'detect_count', 0)
        threshold = getattr(logo_detector, 'detect_threshold', self.logo_confirm_frames)
        remaining = 0.0
        if self.logo_stop_until is not None:
            remaining = max(0.0, self.logo_stop_until - self.get_clock().now().nanoseconds * 1e-9)
        if results.logo is None:
            return f'logo: count={count}/{threshold} stop={remaining:.1f}s'
        return (
            f'logo: CONFIRMED count={count}/{threshold} '
            f'conf={results.logo.confidence:.1f} '
            f'box=({results.logo.box.x},{results.logo.box.y},{results.logo.box.w},{results.logo.box.h}) '
            f'stop={remaining:.1f}s'
        )

    def _static_ball_status_line(self, results: DetectionResults) -> str:
        if results.static_ball is None:
            return 'static_ball: none'
        ball = results.static_ball
        return (
            f'static_ball: conf={ball.confidence:.2f} '
            f'box=({ball.box.x},{ball.box.y},{ball.box.w},{ball.box.h})'
        )

    def _debug_value(self, key: str, default=''):
        return self.latest_controller_debug.get(key, default)

    def _controller_status_line(self, label: str, key: str, fmt: str, suffix: str) -> str:
        value = self.latest_controller_debug.get(key, 0.0)
        try:
            text = format(float(value), fmt)
        except (TypeError, ValueError):
            text = str(value)
        return f'{label}: {text}{suffix}'

    def _cmd_status_line(self) -> str:
        linear = float(self.latest_controller_debug.get('linear_x', 0.0))
        angular = float(self.latest_controller_debug.get('angular_z', 0.0))
        return f'CMD: v={linear:.3f}, w={angular:.3f}'

    def _control_mode_status_line(self) -> str:
        return f"CTRL: {self.latest_controller_debug.get('control_mode', '')}"

    def _format_debug_info(self, info: dict[str, object]) -> str:
        keys = [
            'state',
            'direction',
            'confidence',
            'is_stable',
            'area_ratio',
            'bbox_bottom_ratio',
            'center_error_px',
            'center_angle_deg',
            'center_weight',
            'effective_center_gain',
            'center_bias_deg',
            'heading_error_rad',
            'corrected_heading_error_rad',
            'heading_valid',
            'heading_confidence',
            'arrow_presence_confidence',
            'heading_weight',
            'previous_heading_pull_weight',
            'previous_heading_pull_term',
            'center_term',
            'heading_term',
            'angular_target',
            'angular_smoothed',
            'control_mode',
            'current_odom_yaw',
            'arrow_world_yaw',
            'latched_world_yaw',
            'finish_heading_started_at',
            'close_lock_started_at',
            'close_travel_m',
            'close_required_travel_m',
            'active_target_lost_reason',
            'pass_max_heading_error_deg',
            'desired_yaw',
            'yaw_error_deg',
            'linear_x',
            'angular_z',
            'transition_reason',
        ]
        parts = []
        for key in keys:
            value = info.get(key)
            if isinstance(value, float):
                parts.append(f'{key}={value:.3f}')
            else:
                parts.append(f'{key}={value}')
        return 'smooth_arc ' + ' '.join(parts)

    def _format_detector_debug(self, arrow) -> str:
        detector = self.detectors.get('arrow')
        reject_reason = getattr(detector, 'last_reject_reason', '')
        if arrow is None:
            return f'arrow_detector final_raw_direction=None final_confidence=0.000 reject_reason={reject_reason}'
        axis_angle_deg = 0.0 if arrow.axis_angle_rad is None else math.degrees(arrow.axis_angle_rad)
        paper_axis_deg = 0.0 if arrow.paper_axis_angle_rad is None else math.degrees(arrow.paper_axis_angle_rad)
        paper_heading_deg = (
            0.0 if arrow.paper_heading_angle_rad is None else math.degrees(arrow.paper_heading_angle_rad)
        )
        return (
            'arrow_detector '
            f'box=({arrow.box.x},{arrow.box.y},{arrow.box.w},{arrow.box.h}) '
            f'area_ratio={arrow.area_ratio:.3f} black_pixel_ratio={arrow.black_pixel_ratio:.3f} '
            f'center_error_px={arrow.center_error_px:.1f} '
            f'heading_source={arrow.heading_source} '
            f'heading_valid={arrow.heading_valid} '
            f'heading_confidence={arrow.heading_confidence:.3f} '
            f'arrow_presence_confidence={arrow.arrow_presence_confidence:.3f} '
            f'heading_angle_deg={0.0 if arrow.heading_angle_deg is None else arrow.heading_angle_deg:.1f} '
            f'paper_heading_source=debug_only '
            f'paper_axis_deg={paper_axis_deg:.1f} '
            f'paper_heading_deg={paper_heading_deg:.1f} '
            f'black_arrow_raw={arrow.black_arrow_direction} '
            f'black_arrow_confidence={arrow.black_arrow_confidence:.3f} '
            f'template_direction={arrow.template_direction} '
            f'template_dominance={arrow.template_dominance:.3f} '
            f'axis_direction={arrow.axis_direction} '
            f'axis_confidence={arrow.axis_confidence:.3f} '
            f'axis_angle_deg={axis_angle_deg:.1f} '
            f'final_raw_direction={arrow.raw_direction} '
            f'final_confidence={arrow.confidence:.3f} '
            f'heading_error_rad={0.0 if arrow.heading_error_rad is None else arrow.heading_error_rad:.3f} '
            f'reject_reason={reject_reason}'
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
            node._publish_cmd_vel(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
