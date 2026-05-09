from __future__ import annotations

from dataclasses import dataclass, field
import math

from geometry_msgs.msg import Twist

from tb4_autonomy_real.data_types import ArrowDetection, AutonomyState
from tb4_autonomy_real.utils.geometry import clamp, normalize_angle, shortest_angular_distance


def smooth_angle(old: float, new: float, alpha: float) -> float:
    delta = normalize_angle(new - old)
    return normalize_angle(old + clamp(alpha, 0.0, 1.0) * delta)


@dataclass
class ArrowSmoothArcConfig:
    min_confidence: float = 0.45
    acquire_area_threshold: float = 0.020
    close_area_threshold: float = 0.12
    close_bottom_ratio: float = 0.90
    focal_px: float = 600.0

    heading_sign: float = 1.0
    heading_scale: float = 0.15
    heading_oversteer_deg: float = 2.0
    latched_yaw_alpha: float = 0.20
    previous_heading_pull_gain: float = 0.30
    previous_heading_pull_decay_sec: float = 2.20
    previous_heading_pull_start_delta_deg: float = 10.0
    previous_heading_pull_max_angular_z: float = 0.050
    latched_heading_confidence_min: float = 0.55
    latched_arrow_presence_confidence_min: float = 0.45
    yaw_latch_alpha: float = 0.15
    heading_sample_window: int = 8
    heading_sample_min_count: int = 3
    heading_sample_tolerance_deg: float = 30.0
    min_heading_confidence: float = 0.65
    min_arrow_presence_confidence: float = 0.45
    heading_update_max_area_ratio: float = 0.09
    heading_update_max_bottom_ratio: float = 0.95

    center_sign: float = -1.0
    center_capture_threshold_px: float = 90.0
    min_center_bias_gain: float = 0.20
    max_center_bias_gain: float = 0.85
    max_center_bias_deg: float = 6.0

    kp_yaw: float = 1.2
    kp_center: float = 0.45
    kp_heading: float = 0.30
    max_angular_z: float = 0.10
    max_angular_accel: float = 0.15
    yaw_error_deadband_deg: float = 2.5
    angular_lowpass_alpha: float = 0.12

    heading_confidence_soft_min: float = 0.40
    heading_confidence_full: float = 0.80
    arrow_presence_confidence_soft_min: float = 0.35
    arrow_presence_confidence_full: float = 0.80

    track_speed: float = 0.028
    slow_track_speed: float = 0.020
    slow_yaw_error_deg: float = 25.0
    slow_center_error_px: float = 100.0
    wait_linear_speed: float = 0.025
    pass_speed: float = 0.040
    pass_time_sec: float = 0.15
    missing_detection_hold_sec: float = 0.20
    execute_latched_on_lost: bool = True
    pass_max_heading_error_deg: float = 6.0
    finish_heading_timeout_sec: float = 1.5
    finish_heading_speed: float = 0.015
    finish_heading_kp: float = 0.40
    finish_heading_max_angular_z: float = 0.08
    finish_center_kp: float = 0.35
    finish_center_max_bias_deg: float = 5.0
    finish_center_tolerance_px: float = 45.0
    min_track_time_sec: float = 0.8
    close_camera_bottom_distance_in: float = 18.5
    close_arrow_bottom_distance_in: float = 22.375
    post_close_min_travel_m: float = 0.0
    post_close_speed: float = 0.055
    post_close_timeout_sec: float = 3.0
    active_target_area_drop_ratio: float = 0.70
    active_target_bottom_drop_ratio: float = 0.08
    debug_log: bool = True

    @property
    def max_center_bias_rad(self) -> float:
        return math.radians(self.max_center_bias_deg)

    @property
    def slow_yaw_error_rad(self) -> float:
        return math.radians(self.slow_yaw_error_deg)

    @property
    def heading_sample_tolerance_rad(self) -> float:
        return math.radians(self.heading_sample_tolerance_deg)

    @property
    def yaw_error_deadband_rad(self) -> float:
        return math.radians(self.yaw_error_deadband_deg)

    @property
    def previous_heading_pull_start_delta_rad(self) -> float:
        return math.radians(self.previous_heading_pull_start_delta_deg)

    @property
    def pass_max_heading_error_rad(self) -> float:
        return math.radians(self.pass_max_heading_error_deg)

    @property
    def finish_center_max_bias_rad(self) -> float:
        return math.radians(self.finish_center_max_bias_deg)

    @property
    def calibrated_post_close_travel_m(self) -> float:
        if self.post_close_min_travel_m > 0.0:
            return self.post_close_min_travel_m
        extra_inches = max(0.0, self.close_arrow_bottom_distance_in - self.close_camera_bottom_distance_in)
        return extra_inches * 0.0254


@dataclass
class ArrowSmoothArcOutput:
    twist: Twist
    debug_info: dict[str, object] = field(default_factory=dict)
    current_state: AutonomyState = AutonomyState.WAIT_FOR_TARGET


class ArrowSmoothArcController:
    def __init__(self, config: ArrowSmoothArcConfig | None = None):
        self.config = config or ArrowSmoothArcConfig()
        self.state = AutonomyState.WAIT_FOR_TARGET
        self.arrow_world_yaw: float | None = None
        self.latched_world_yaw: float | None = None
        self.pass_until: float | None = None
        self.last_update_time: float | None = None
        self.last_seen_time: float | None = None
        self.track_started_at: float | None = None
        self.finish_heading_started_at: float | None = None
        self.close_lock_started_at: float | None = None
        self.close_lock_started_xy: tuple[float, float] | None = None
        self.previous_heading_pull_yaw: float | None = None
        self.previous_heading_pull_started_at: float | None = None
        self.last_active_area_ratio: float | None = None
        self.last_active_bottom_ratio: float | None = None
        self.active_target_lost_reason = ''
        self.last_angular_z = 0.0
        self.last_debug_info: dict[str, object] = {}

    def reset(self) -> None:
        self.state = AutonomyState.WAIT_FOR_TARGET
        self.arrow_world_yaw = None
        self.latched_world_yaw = None
        self.pass_until = None
        self.last_update_time = None
        self.last_seen_time = None
        self.track_started_at = None
        self.finish_heading_started_at = None
        self.close_lock_started_at = None
        self.close_lock_started_xy = None
        self.previous_heading_pull_yaw = None
        self.previous_heading_pull_started_at = None
        self.last_active_area_ratio = None
        self.last_active_bottom_ratio = None
        self.active_target_lost_reason = ''
        self.last_angular_z = 0.0
        self.last_debug_info = {}

    def update(
        self,
        detection: ArrowDetection | None,
        image_width: int,
        image_height: int,
        now_sec: float,
        current_odom_yaw: float | None,
        current_odom_xy: tuple[float, float] | None = None,
    ) -> ArrowSmoothArcOutput:
        transition_reason = ''
        self.active_target_lost_reason = ''
        paper_valid = self._paper_is_valid(detection)
        if (
            self.state == AutonomyState.SMOOTH_ARC_TRACK
            and paper_valid
            and self._looks_like_new_far_target(detection, image_height)
        ):
            paper_valid = False
        if paper_valid:
            self.last_seen_time = now_sec

        if self.state == AutonomyState.PASS_TO_NEXT:
            if self.pass_until is not None and now_sec < self.pass_until:
                output = self._pass_output(detection, image_height, 'pass_to_next', current_odom_yaw)
                self._remember_timing(now_sec, output.twist.angular.z)
                return output
            self.pass_until = None
            self.last_angular_z = 0.0
            self.latched_world_yaw = None
            self.previous_heading_pull_yaw = None
            self.previous_heading_pull_started_at = None
            self.finish_heading_started_at = None
            self.track_started_at = None
            self.close_lock_started_at = None
            self.close_lock_started_xy = None
            self.state = AutonomyState.WAIT_FOR_TARGET
            transition_reason = 'pass_complete'

        if self.state == AutonomyState.WAIT_FOR_TARGET:
            if paper_valid:
                self.state = AutonomyState.SMOOTH_ARC_TRACK
                self.track_started_at = now_sec
                self._remember_active_target(detection, image_height)
                transition_reason = 'acquired_paper'
            else:
                output = self._wait_output(detection, image_height, transition_reason, current_odom_yaw)
                self._remember_timing(now_sec, output.twist.angular.z)
                return output

        if self.state == AutonomyState.SMOOTH_ARC_TRACK:
            latched_updated = False
            is_close = paper_valid and self._is_close(detection, image_height)
            if paper_valid:
                if not is_close or self.latched_world_yaw is None:
                    latched_updated = self._update_latched_world_yaw(detection, current_odom_yaw, now_sec)
                self._remember_active_target(detection, image_height)

            if not paper_valid:
                if self._can_hold_missing_detection(now_sec):
                    output = self._missing_hold_output(detection, image_height, now_sec, current_odom_yaw)
                    self._remember_timing(now_sec, output.twist.angular.z)
                    return output
                if self.config.execute_latched_on_lost and self.latched_world_yaw is not None:
                    self._start_close_lock(now_sec, current_odom_xy)
                    if self._should_finish_latched_heading(detection, now_sec, current_odom_yaw, current_odom_xy):
                        output = self._finish_latched_heading_output(
                            None,
                            image_height,
                            now_sec,
                            current_odom_yaw,
                            current_odom_xy,
                            'target_lost_execute_latched_heading',
                        )
                        output.debug_info['latched_updated'] = latched_updated
                        output.debug_info['heading_update_accepted'] = latched_updated
                        self._remember_timing(now_sec, output.twist.angular.z)
                        return output
                    self.state = AutonomyState.PASS_TO_NEXT
                    self.pass_until = now_sec + self.config.pass_time_sec
                    self.last_angular_z = 0.0
                    self.finish_heading_started_at = None
                    output = self._pass_output(None, image_height, 'target_lost_latched_heading_done', current_odom_yaw)
                    output.debug_info['latched_updated'] = latched_updated
                    output.debug_info['heading_update_accepted'] = latched_updated
                    self._remember_timing(now_sec, output.twist.angular.z)
                    return output
                self.state = AutonomyState.WAIT_FOR_TARGET
                self.last_angular_z = 0.0
                output = self._wait_output(detection, image_height, 'lost_target', current_odom_yaw)
                self._remember_timing(now_sec, output.twist.angular.z)
                return output

            output = self._track_output(detection, image_width, image_height, now_sec, current_odom_yaw, transition_reason)
            output.debug_info['latched_updated'] = latched_updated
            output.debug_info['heading_update_accepted'] = latched_updated
            self._remember_timing(now_sec, output.twist.angular.z)
            return output

        output = self._wait_output(detection, image_height, transition_reason or 'fallback_wait', current_odom_yaw)
        self._remember_timing(now_sec, output.twist.angular.z)
        return output

    def _paper_is_valid(self, detection: ArrowDetection | None) -> bool:
        return (
            detection is not None
            and detection.confidence >= self.config.min_confidence
            and detection.area_ratio >= self.config.acquire_area_threshold
        )

    def _is_close(self, detection: ArrowDetection | None, image_height: int) -> bool:
        if detection is None:
            return False
        bottom_ratio = self._bottom_ratio(detection, image_height)
        return (
            detection.area_ratio >= self.config.close_area_threshold
            or bottom_ratio >= self.config.close_bottom_ratio
        )

    def _heading_is_latchable(
        self,
        detection: ArrowDetection | None,
        current_odom_yaw: float | None,
    ) -> bool:
        if detection is None or current_odom_yaw is None:
            return False
        if not bool(getattr(detection, 'heading_valid', False)):
            return False
        if detection.heading_error_rad is None:
            return False
        if float(getattr(detection, 'heading_confidence', 0.0) or 0.0) < self.config.latched_heading_confidence_min:
            return False
        if (
            float(getattr(detection, 'arrow_presence_confidence', 0.0) or 0.0)
            < self.config.latched_arrow_presence_confidence_min
        ):
            return False
        return True

    def _looks_like_new_far_target(self, detection: ArrowDetection | None, image_height: int) -> bool:
        if detection is None or self.latched_world_yaw is None:
            return False
        if self.last_active_area_ratio is None or self.last_active_bottom_ratio is None:
            return False
        bottom_ratio = self._bottom_ratio(detection, image_height)
        area_ratio = float(detection.area_ratio)
        area_drop = area_ratio < self.last_active_area_ratio * self.config.active_target_area_drop_ratio
        bottom_drop = bottom_ratio < self.last_active_bottom_ratio - self.config.active_target_bottom_drop_ratio
        if area_drop and bottom_drop:
            self.active_target_lost_reason = 'new_far_candidate_after_active_target'
            return True
        return False

    def _remember_active_target(self, detection: ArrowDetection | None, image_height: int) -> None:
        if detection is None:
            return
        self.last_active_area_ratio = max(float(detection.area_ratio), self.last_active_area_ratio or 0.0)
        self.last_active_bottom_ratio = max(self._bottom_ratio(detection, image_height), self.last_active_bottom_ratio or 0.0)

    def _update_latched_world_yaw(
        self,
        detection: ArrowDetection | None,
        current_odom_yaw: float | None,
        now_sec: float,
    ) -> bool:
        if not self._heading_is_latchable(detection, current_odom_yaw):
            return False

        observed_world_yaw = normalize_angle(
            current_odom_yaw
            + self.config.heading_sign
            * self.config.heading_scale
            * float(detection.heading_error_rad)
        )
        if self.latched_world_yaw is None:
            self.latched_world_yaw = observed_world_yaw
        else:
            yaw_delta = abs(shortest_angular_distance(self.latched_world_yaw, observed_world_yaw))
            if (
                self.previous_heading_pull_yaw is None
                and yaw_delta >= self.config.previous_heading_pull_start_delta_rad
            ):
                self.previous_heading_pull_yaw = self.latched_world_yaw
                self.previous_heading_pull_started_at = now_sec
            self.latched_world_yaw = smooth_angle(
                self.latched_world_yaw,
                observed_world_yaw,
                self.config.latched_yaw_alpha,
            )
        return True

    def _should_finish_latched_heading(
        self,
        detection: ArrowDetection | None,
        now_sec: float,
        current_odom_yaw: float | None,
        current_odom_xy: tuple[float, float] | None = None,
    ) -> bool:
        close_travel = self._close_travel_m(now_sec, current_odom_xy)
        required_travel = self.config.calibrated_post_close_travel_m
        close_elapsed = 0.0 if self.close_lock_started_at is None else now_sec - self.close_lock_started_at
        close_timeout_active = close_elapsed <= self.config.post_close_timeout_sec
        if required_travel > 0.0 and close_travel < required_travel and close_timeout_active:
            if self.finish_heading_started_at is None:
                self.finish_heading_started_at = now_sec
            return True

        if current_odom_yaw is None or self.latched_world_yaw is None:
            return False

        if self._needs_finish_center(detection):
            if self.finish_heading_started_at is None:
                self.finish_heading_started_at = now_sec
                return True
            return now_sec - self.finish_heading_started_at <= self.config.finish_heading_timeout_sec

        if self.last_debug_info.get('control_mode') == 'finish_center_first':
            self.finish_heading_started_at = None

        if self.track_started_at is not None:
            if now_sec - self.track_started_at < self.config.min_track_time_sec:
                return True

        yaw_error = shortest_angular_distance(current_odom_yaw, self.latched_world_yaw)
        if abs(yaw_error) <= self.config.pass_max_heading_error_rad:
            return False

        if self.finish_heading_started_at is None:
            self.finish_heading_started_at = now_sec
            return True

        if now_sec - self.finish_heading_started_at > self.config.finish_heading_timeout_sec:
            return False

        return True

    def _start_close_lock(self, now_sec: float, current_odom_xy: tuple[float, float] | None) -> None:
        if self.close_lock_started_at is not None:
            return
        self.close_lock_started_at = now_sec
        self.close_lock_started_xy = current_odom_xy

    def _close_travel_m(self, now_sec: float, current_odom_xy: tuple[float, float] | None) -> float:
        if self.close_lock_started_at is None:
            return 0.0
        elapsed_travel = max(0.0, now_sec - self.close_lock_started_at) * max(0.0, self.config.post_close_speed)
        if self.close_lock_started_xy is not None and current_odom_xy is not None:
            odom_travel = math.hypot(
                current_odom_xy[0] - self.close_lock_started_xy[0],
                current_odom_xy[1] - self.close_lock_started_xy[1],
            )
            return max(odom_travel, elapsed_travel)
        return elapsed_travel

    def _needs_finish_center(self, detection: ArrowDetection | None) -> bool:
        if detection is None:
            return False
        return abs(float(detection.center_error_px)) > self.config.finish_center_tolerance_px

    def _finish_latched_heading_output(
        self,
        detection: ArrowDetection | None,
        image_height: int,
        now_sec: float,
        current_odom_yaw: float | None,
        current_odom_xy: tuple[float, float] | None,
        transition_reason: str,
    ) -> ArrowSmoothArcOutput:
        twist = Twist()
        if current_odom_yaw is None or self.latched_world_yaw is None:
            twist.linear.x = self.config.pass_speed
            twist.angular.z = 0.0
            return self._output(
                twist,
                detection,
                image_height,
                transition_reason,
                control_mode='finish_latched_heading_no_odom',
                current_odom_yaw=current_odom_yaw,
            )

        yaw_error = shortest_angular_distance(current_odom_yaw, self.latched_world_yaw)
        raw_yaw_term = clamp(
            self.config.finish_heading_kp * yaw_error,
            -self.config.finish_heading_max_angular_z,
            self.config.finish_heading_max_angular_z,
        )
        if detection is not None:
            center_error_px = float(detection.center_error_px)
            center_angle = math.atan2(center_error_px, max(1.0, self.config.focal_px))
            raw_center_term = self.config.center_sign * self.config.finish_center_kp * center_angle
            raw_center_term = clamp(
                raw_center_term,
                -self.config.finish_center_max_bias_rad,
                self.config.finish_center_max_bias_rad,
            )
        else:
            center_error_px = 0.0
            center_angle = 0.0
            raw_center_term = 0.0

        if abs(center_error_px) > self.config.finish_center_tolerance_px:
            yaw_term = 0.0
            center_term = raw_center_term
            control_mode = 'finish_center_first'
        else:
            yaw_term = raw_yaw_term
            center_term = 0.0
            control_mode = 'finish_latched_heading'

        angular_target = yaw_term + center_term
        angular_target = clamp(
            angular_target,
            -self.config.finish_heading_max_angular_z,
            self.config.finish_heading_max_angular_z,
        )
        angular_z = self._smooth_angular(angular_target, now_sec)

        twist.linear.x = self.config.post_close_speed if self.close_lock_started_at is not None else self.config.finish_heading_speed
        twist.angular.z = angular_z
        return self._output(
            twist,
            detection,
            image_height,
            transition_reason,
            heading_error=yaw_error,
            heading_weight=1.0,
            center_angle=center_angle,
            center_term=center_term,
            finish_yaw_term=yaw_term,
            finish_center_term=center_term,
            finish_center_error_px=center_error_px,
            close_travel_m=self._close_travel_m(now_sec, current_odom_xy),
            close_required_travel_m=self.config.calibrated_post_close_travel_m,
            angular_target=angular_target,
            angular_smoothed=angular_z,
            control_mode=control_mode,
            current_odom_yaw=current_odom_yaw,
            arrow_world_yaw=self.latched_world_yaw,
            yaw_error=yaw_error,
        )

    def _track_output(
        self,
        detection: ArrowDetection,
        image_width: int,
        image_height: int,
        now_sec: float,
        current_odom_yaw: float | None,
        transition_reason: str,
    ) -> ArrowSmoothArcOutput:
        center_error_px = float(detection.center_error_px)
        center_angle = math.atan2(center_error_px, max(1.0, self.config.focal_px))
        center_term = self.config.center_sign * self.config.kp_center * center_angle
        center_term = clamp(center_term, -self.config.max_center_bias_rad, self.config.max_center_bias_rad)

        heading_valid = bool(getattr(detection, 'heading_valid', False))
        heading_error = detection.heading_error_rad if detection.heading_error_rad is not None else 0.0
        if heading_valid and detection.heading_error_rad is not None:
            heading_weight = self._confidence_weight(detection)
            corrected_heading_error = self._oversteer_heading_error(float(heading_error))
            heading_term = (
                self.config.heading_sign
                * self.config.heading_scale
                * self.config.kp_heading
                * corrected_heading_error
            )
        else:
            heading_weight = 0.0
            heading_term = 0.0
            corrected_heading_error = 0.0

        previous_heading_pull_term, previous_heading_pull_weight = self._previous_heading_pull_term(
            now_sec,
            current_odom_yaw,
        )
        angular_target = center_term + heading_weight * heading_term + previous_heading_pull_term
        effective_max_angular_z = max(0.0, self.config.max_angular_z)
        angular_target = clamp(angular_target, -effective_max_angular_z, effective_max_angular_z)
        angular_z = self._smooth_angular(angular_target, now_sec)

        linear_x = self.config.track_speed
        if abs(center_error_px) > self.config.slow_center_error_px:
            linear_x = self.config.slow_track_speed
        if abs(angular_z) > effective_max_angular_z * 0.65:
            linear_x = min(linear_x, self.config.slow_track_speed)

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        control_mode = 'confidence_weighted_heading' if heading_weight > 0.01 else 'center_only'
        return self._output(
            twist,
            detection,
            image_height,
            transition_reason,
            center_angle=center_angle,
            center_weight=clamp(abs(center_error_px) / max(1.0, self.config.center_capture_threshold_px), 0.0, 1.0),
            effective_center_gain=self.config.kp_center,
            center_bias=center_term,
            heading_error=heading_error,
            corrected_heading_error=corrected_heading_error,
            heading_valid=heading_valid,
            heading_weight=heading_weight,
            center_term=center_term,
            heading_term=heading_term,
            previous_heading_pull_term=previous_heading_pull_term,
            previous_heading_pull_weight=previous_heading_pull_weight,
            angular_target=angular_target,
            angular_smoothed=angular_z,
            control_mode=control_mode,
            current_odom_yaw=current_odom_yaw,
            yaw_error=0.0,
        )

    def _previous_heading_pull_term(
        self,
        now_sec: float,
        current_odom_yaw: float | None,
    ) -> tuple[float, float]:
        if (
            self.previous_heading_pull_yaw is None
            or self.previous_heading_pull_started_at is None
            or current_odom_yaw is None
        ):
            return 0.0, 0.0

        age = max(0.0, now_sec - self.previous_heading_pull_started_at)
        decay_sec = max(0.05, self.config.previous_heading_pull_decay_sec)
        weight = math.exp(-age / decay_sec)
        if weight < 0.02:
            self.previous_heading_pull_yaw = None
            self.previous_heading_pull_started_at = None
            return 0.0, 0.0

        yaw_error = shortest_angular_distance(current_odom_yaw, self.previous_heading_pull_yaw)
        term = self.config.previous_heading_pull_gain * yaw_error * weight
        term = clamp(
            term,
            -self.config.previous_heading_pull_max_angular_z,
            self.config.previous_heading_pull_max_angular_z,
        )
        return term, weight

    def _oversteer_heading_error(self, heading_error: float) -> float:
        if abs(heading_error) < self.config.yaw_error_deadband_rad:
            return heading_error
        oversteer = math.radians(max(0.0, self.config.heading_oversteer_deg))
        return normalize_angle(heading_error + math.copysign(oversteer, heading_error))

    def _smooth_angular(self, target: float, now_sec: float) -> float:
        alpha = clamp(self.config.angular_lowpass_alpha, 0.0, 1.0)
        smoothed = self.last_angular_z + alpha * (target - self.last_angular_z)
        if self.last_update_time is None:
            return smoothed

        dt = max(0.0, now_sec - self.last_update_time)
        max_delta = self.config.max_angular_accel * dt
        return clamp(smoothed, self.last_angular_z - max_delta, self.last_angular_z + max_delta)

    def _confidence_weight(self, detection: ArrowDetection) -> float:
        heading_conf = float(getattr(detection, 'heading_confidence', 0.0) or 0.0)
        presence_conf = float(getattr(detection, 'arrow_presence_confidence', 0.0) or 0.0)
        heading_w = self._smoothstep(
            self.config.heading_confidence_soft_min,
            self.config.heading_confidence_full,
            heading_conf,
        )
        presence_w = self._smoothstep(
            self.config.arrow_presence_confidence_soft_min,
            self.config.arrow_presence_confidence_full,
            presence_conf,
        )
        return heading_w * presence_w

    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        if edge1 <= edge0:
            return 1.0 if x >= edge1 else 0.0
        t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _can_hold_missing_detection(self, now_sec: float) -> bool:
        return (
            self.last_seen_time is not None
            and now_sec - self.last_seen_time <= self.config.missing_detection_hold_sec
        )

    def _missing_hold_output(
        self,
        detection: ArrowDetection | None,
        image_height: int,
        now_sec: float,
        current_odom_yaw: float | None,
    ) -> ArrowSmoothArcOutput:
        angular_z = self._smooth_angular(0.0, now_sec)
        twist = Twist()
        twist.linear.x = self.config.slow_track_speed
        twist.angular.z = angular_z
        return self._output(
            twist,
            detection,
            image_height,
            'missing_detection_hold',
            heading_weight=0.0,
            angular_target=0.0,
            angular_smoothed=angular_z,
            control_mode='missing_hold',
            current_odom_yaw=current_odom_yaw,
        )

    def _heading_error(self, detection: ArrowDetection | None) -> float:
        if detection is None or detection.heading_error_rad is None:
            return 0.0
        return detection.heading_error_rad

    def _wait_output(
        self,
        detection: ArrowDetection | None,
        image_height: int,
        transition_reason: str,
        current_odom_yaw: float | None,
    ) -> ArrowSmoothArcOutput:
        twist = Twist()
        twist.linear.x = self.config.wait_linear_speed
        return self._output(
            twist,
            detection,
            image_height,
            transition_reason,
            heading_weight=0.0,
            angular_target=0.0,
            angular_smoothed=0.0,
            control_mode='wait',
            current_odom_yaw=current_odom_yaw,
        )

    def _pass_output(
        self,
        detection: ArrowDetection | None,
        image_height: int,
        transition_reason: str,
        current_odom_yaw: float | None,
    ) -> ArrowSmoothArcOutput:
        twist = Twist()
        twist.linear.x = self.config.pass_speed
        twist.angular.z = 0.0
        return self._output(
            twist,
            detection,
            image_height,
            transition_reason,
            heading_weight=0.0,
            angular_target=0.0,
            angular_smoothed=0.0,
            control_mode='pass_to_next',
            current_odom_yaw=current_odom_yaw,
        )

    def _bottom_ratio(self, detection: ArrowDetection | None, image_height: int) -> float:
        if detection is None or image_height <= 0:
            return 0.0
        return (detection.box.y + detection.box.h) / float(image_height)

    def _remember_timing(self, now_sec: float, angular_z: float) -> None:
        self.last_update_time = now_sec
        self.last_angular_z = angular_z

    def _output(
        self,
        twist: Twist,
        detection: ArrowDetection | None,
        image_height: int,
        transition_reason: str,
        **values,
    ) -> ArrowSmoothArcOutput:
        heading_confidence = 0.0 if detection is None else float(getattr(detection, 'heading_confidence', 0.0) or 0.0)
        presence_confidence = (
            0.0 if detection is None else float(getattr(detection, 'arrow_presence_confidence', 0.0) or 0.0)
        )
        info = {
            'state': self.state.value,
            'direction': None if detection is None else detection.direction,
            'confidence': 0.0 if detection is None else detection.confidence,
            'is_stable': False if detection is None else detection.is_stable,
            'area_ratio': 0.0 if detection is None else detection.area_ratio,
            'bbox_bottom_ratio': self._bottom_ratio(detection, image_height),
            'center_error_px': 0.0 if detection is None else detection.center_error_px,
            'center_angle_deg': math.degrees(float(values.get('center_angle', 0.0))),
            'center_weight': float(values.get('center_weight', 0.0)),
            'effective_center_gain': float(values.get('effective_center_gain', 0.0)),
            'center_bias_deg': math.degrees(float(values.get('center_bias', 0.0))),
            'heading_error_rad': float(values.get('heading_error', self._heading_error(detection))),
            'corrected_heading_error_rad': float(values.get('corrected_heading_error', 0.0)),
            'heading_valid': bool(values.get('heading_valid', False if detection is None else detection.heading_valid)),
            'heading_confidence': heading_confidence,
            'arrow_presence_confidence': presence_confidence,
            'heading_weight': float(values.get('heading_weight', 0.0)),
            'center_term': float(values.get('center_term', 0.0)),
            'heading_term': float(values.get('heading_term', 0.0)),
            'previous_heading_pull_term': float(values.get('previous_heading_pull_term', 0.0)),
            'previous_heading_pull_weight': float(values.get('previous_heading_pull_weight', 0.0)),
            'previous_heading_pull_yaw': self.previous_heading_pull_yaw,
            'finish_yaw_term': float(values.get('finish_yaw_term', 0.0)),
            'finish_center_term': float(values.get('finish_center_term', 0.0)),
            'finish_center_error_px': float(values.get('finish_center_error_px', 0.0)),
            'finish_center_tolerance_px': self.config.finish_center_tolerance_px,
            'angular_target': float(values.get('angular_target', 0.0)),
            'angular_smoothed': float(values.get('angular_smoothed', twist.angular.z)),
            'control_mode': values.get('control_mode', ''),
            'current_odom_yaw': values.get('current_odom_yaw'),
            'arrow_world_yaw': values.get('arrow_world_yaw', self.latched_world_yaw),
            'latched_world_yaw': self.latched_world_yaw,
            'track_started_at': self.track_started_at,
            'finish_heading_started_at': self.finish_heading_started_at,
            'close_lock_started_at': self.close_lock_started_at,
            'close_travel_m': float(values.get('close_travel_m', 0.0)),
            'close_required_travel_m': float(values.get('close_required_travel_m', self.config.calibrated_post_close_travel_m)),
            'active_target_lost_reason': self.active_target_lost_reason,
            'pass_max_heading_error_deg': self.config.pass_max_heading_error_deg,
            'desired_yaw': values.get('desired_yaw'),
            'yaw_error_deg': math.degrees(float(values.get('yaw_error', 0.0))),
            'heading_sample_count': 0,
            'heading_consensus_yaw': None,
            'heading_update_accepted': False,
            'heading_update_reason': '',
            'linear_x': twist.linear.x,
            'angular_z': twist.angular.z,
            'transition_reason': transition_reason,
        }
        self.last_debug_info = info
        return ArrowSmoothArcOutput(twist=twist, debug_info=info, current_state=self.state)
