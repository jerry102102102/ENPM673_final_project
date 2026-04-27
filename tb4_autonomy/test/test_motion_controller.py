import math

import pytest

from tb4_autonomy.motion_controller import MotionController, MotionControllerConfig


def test_stop_returns_zero_twist():
    controller = MotionController()
    msg = controller.stop()
    assert msg.linear.x == 0.0
    assert msg.angular.z == 0.0


def test_cruise_uses_configured_speed():
    controller = MotionController(MotionControllerConfig(cruise_linear_x=0.2))
    msg = controller.cruise()
    assert msg.linear.x == pytest.approx(0.2)
    assert msg.angular.z == 0.0


def test_track_x_error_turns_toward_target():
    controller = MotionController(MotionControllerConfig(track_kp=0.01, max_angular_z=0.5))

    right_of_center = controller.track_x_error(error_px=100.0, image_width=640)
    left_of_center = controller.track_x_error(error_px=-100.0, image_width=640)

    assert right_of_center.angular.z < 0.0
    assert left_of_center.angular.z > 0.0
    assert abs(right_of_center.angular.z) <= 0.5


def test_align_x_error_rotates_without_forward_motion():
    controller = MotionController(MotionControllerConfig(track_kp=0.01, max_angular_z=0.5))
    msg = controller.align_x_error(error_px=-50.0, image_width=640)

    assert msg.linear.x == 0.0
    assert msg.angular.z > 0.0


def test_left_turn_completes_near_target_yaw():
    controller = MotionController(MotionControllerConfig(turn_angle_rad=math.pi / 2.0))

    controller.start_turn('left', current_yaw=0.0)
    msg, done = controller.update_turn(current_yaw=0.0)
    assert not done
    assert msg.angular.z > 0.0

    msg, done = controller.update_turn(current_yaw=math.pi / 2.0)
    assert done
    assert msg.angular.z == 0.0


def test_right_turn_completes_near_target_yaw():
    controller = MotionController(MotionControllerConfig(turn_angle_rad=math.pi / 2.0))

    controller.start_turn('right', current_yaw=0.0)
    msg, done = controller.update_turn(current_yaw=0.0)
    assert not done
    assert msg.angular.z < 0.0

    msg, done = controller.update_turn(current_yaw=-math.pi / 2.0)
    assert done
    assert msg.angular.z == 0.0


def test_back_turn_targets_half_rotation():
    controller = MotionController(MotionControllerConfig(back_turn_angle_rad=math.pi))

    controller.start_turn('back', current_yaw=0.0)
    msg, done = controller.update_turn(current_yaw=0.0)
    assert not done
    assert msg.angular.z > 0.0

    msg, done = controller.update_turn(current_yaw=math.pi)
    assert done
    assert msg.angular.z == 0.0
