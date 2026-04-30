import math

import pytest

from tb4_autonomy.arrow_smooth_arc_controller import (
    ArrowSmoothArcConfig,
    ArrowSmoothArcController,
    smooth_angle,
)
from tb4_autonomy.data_types import ArrowDetection, AutonomyState, Box2D


def _arrow(**overrides):
    values = {
        'box': Box2D(280, 300, 120, 120),
        'direction': 'left',
        'confidence': 0.8,
        'area_ratio': 0.04,
        'center_error_px': 80.0,
        'is_stable': True,
        'heading_error_rad': 0.40,
    }
    values.update(overrides)
    return ArrowDetection(**values)


def _heading_arrow(**overrides):
    values = {
        'box': Box2D(280, 240, 120, 120),
        'heading_valid': True,
        'heading_confidence': 0.9,
        'arrow_presence_confidence': 0.9,
        'is_stable': False,
        'direction': 'unknown',
        'raw_direction': 'unknown',
    }
    values.update(overrides)
    return _arrow(**values)


def test_wait_for_target_acquires_valid_paper_without_stable_label():
    controller = ArrowSmoothArcController(ArrowSmoothArcConfig(debug_log=False))

    output = controller.update(_arrow(confidence=0.1, is_stable=False), 640, 480, 1.0, 0.0)
    assert output.current_state == AutonomyState.WAIT_FOR_TARGET
    assert output.twist.linear.x == pytest.approx(controller.config.wait_linear_speed)

    output = controller.update(_arrow(is_stable=False, direction='unknown'), 640, 480, 1.1, 0.0)
    assert output.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert output.debug_info['transition_reason'] == 'acquired_paper'


def test_smooth_arc_tracks_with_small_limited_angular_velocity():
    config = ArrowSmoothArcConfig(
        max_angular_z=0.30,
        max_angular_accel=0.60,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)

    first = controller.update(_arrow(center_error_px=120.0), 640, 480, 1.0, 0.0)
    second = controller.update(_arrow(center_error_px=-120.0), 640, 480, 1.05, 0.0)

    assert second.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert second.twist.linear.x == pytest.approx(config.slow_track_speed)
    assert abs(second.twist.angular.z) <= config.max_angular_z
    assert abs(second.twist.angular.z - first.twist.angular.z) <= config.max_angular_accel * 0.05 + 1e-6


def test_center_bias_sign_is_configurable():
    left_bias = ArrowSmoothArcController(
        ArrowSmoothArcConfig(center_sign=-1.0, debug_log=False)
    )
    right_bias = ArrowSmoothArcController(
        ArrowSmoothArcConfig(center_sign=1.0, debug_log=False)
    )

    left_output = left_bias.update(_arrow(center_error_px=100.0), 640, 480, 1.0, 0.0)
    right_output = right_bias.update(_arrow(center_error_px=100.0), 640, 480, 1.0, 0.0)

    assert left_output.debug_info['center_bias_deg'] < 0.0
    assert right_output.debug_info['center_bias_deg'] > 0.0


def test_close_arrow_enters_short_pass_state_then_waits_again():
    controller = ArrowSmoothArcController(
        ArrowSmoothArcConfig(pass_time_sec=0.2, debug_log=False)
    )

    controller.update(_arrow(), 640, 480, 1.0, 0.0)
    close = _arrow(area_ratio=0.13)
    output = controller.update(close, 640, 480, 1.1, 0.0)
    assert output.current_state == AutonomyState.PASS_TO_NEXT
    assert output.twist.linear.x == pytest.approx(controller.config.pass_speed)
    assert output.debug_info['current_odom_yaw'] == pytest.approx(0.0)

    output = controller.update(None, 640, 480, 1.31, 0.0)
    assert output.current_state == AutonomyState.WAIT_FOR_TARGET


def test_smooth_angle_takes_shortest_path_across_wraparound():
    old = math.radians(179.0)
    new = math.radians(-179.0)
    smoothed = smooth_angle(old, new, 0.5)
    assert abs(abs(smoothed) - math.pi) < math.radians(2.0)


def test_heading_confidence_weights_heading_term():
    config = ArrowSmoothArcConfig(
        angular_lowpass_alpha=1.0,
        max_angular_accel=100.0,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)

    low = controller.update(
        _heading_arrow(center_error_px=0.0, heading_error_rad=1.0, heading_confidence=0.2),
        640,
        480,
        1.0,
        0.0,
    )
    assert low.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert low.debug_info['control_mode'] == 'center_only'
    assert low.debug_info['heading_weight'] == pytest.approx(0.0)

    high = controller.update(
        _heading_arrow(center_error_px=0.0, heading_error_rad=1.0, heading_confidence=0.95),
        640,
        480,
        1.1,
        0.0,
    )
    assert high.debug_info['control_mode'] == 'confidence_weighted_heading'
    assert high.debug_info['heading_weight'] > 0.9


def test_missing_detection_holds_briefly_then_waits():
    config = ArrowSmoothArcConfig(
        missing_detection_hold_sec=0.2,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)

    controller.update(_heading_arrow(area_ratio=0.04), 640, 480, 1.0, 0.0)
    held = controller.update(None, 640, 480, 1.1, 0.0)
    assert held.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert held.debug_info['control_mode'] == 'missing_hold'

    waited = controller.update(None, 640, 480, 1.31, 0.0)
    assert waited.current_state == AutonomyState.WAIT_FOR_TARGET


def test_heading_error_oversteers_by_configured_degrees():
    config = ArrowSmoothArcConfig(
        heading_oversteer_deg=5.0,
        angular_lowpass_alpha=1.0,
        max_angular_accel=100.0,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)

    output = controller.update(
        _heading_arrow(center_error_px=0.0, heading_error_rad=0.20, heading_confidence=0.95),
        640,
        480,
        1.0,
        0.0,
    )
    assert output.debug_info['corrected_heading_error_rad'] == pytest.approx(0.20 + math.radians(5.0))


def test_finish_latched_heading_centers_before_yaw():
    config = ArrowSmoothArcConfig(
        angular_lowpass_alpha=1.0,
        max_angular_accel=100.0,
        min_track_time_sec=0.0,
        finish_heading_kp=0.40,
        finish_heading_max_angular_z=0.08,
        finish_center_kp=0.35,
        finish_center_tolerance_px=45.0,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)
    controller.state = AutonomyState.SMOOTH_ARC_TRACK
    controller.latched_world_yaw = 0.20
    controller.track_started_at = 0.0

    output = controller.update(
        _heading_arrow(area_ratio=0.30, center_error_px=100.0),
        640,
        480,
        1.0,
        0.0,
    )

    assert output.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert output.debug_info['control_mode'] == 'finish_center_first'
    assert output.debug_info['finish_yaw_term'] == pytest.approx(0.0)
    assert output.debug_info['finish_center_term'] < 0.0
    assert output.debug_info['angular_target'] == pytest.approx(output.debug_info['finish_center_term'])


def test_finish_latched_heading_uses_yaw_after_center_is_close():
    config = ArrowSmoothArcConfig(
        angular_lowpass_alpha=1.0,
        max_angular_accel=100.0,
        min_track_time_sec=0.0,
        finish_heading_kp=0.40,
        finish_heading_max_angular_z=0.08,
        finish_center_kp=0.35,
        finish_center_tolerance_px=45.0,
        debug_log=False,
    )
    controller = ArrowSmoothArcController(config)
    controller.state = AutonomyState.SMOOTH_ARC_TRACK
    controller.latched_world_yaw = 0.20
    controller.track_started_at = 0.0

    output = controller.update(
        _heading_arrow(area_ratio=0.30, center_error_px=20.0),
        640,
        480,
        1.0,
        0.0,
    )

    assert output.current_state == AutonomyState.SMOOTH_ARC_TRACK
    assert output.debug_info['control_mode'] == 'finish_latched_heading'
    assert output.debug_info['finish_yaw_term'] == pytest.approx(0.08)
    assert output.debug_info['finish_center_term'] == pytest.approx(0.0)
    assert output.debug_info['angular_target'] == pytest.approx(output.debug_info['finish_yaw_term'])
