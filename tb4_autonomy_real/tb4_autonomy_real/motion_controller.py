from __future__ import annotations

from dataclasses import dataclass
import math

from geometry_msgs.msg import Twist

from tb4_autonomy_real.utils.geometry import clamp, normalize_angle, shortest_angular_distance


@dataclass
class MotionControllerConfig:
    cruise_linear_x: float = 0.12
    track_linear_x: float = 0.08
    track_kp: float = 0.002
    max_angular_z: float = 0.65
    turn_kp: float = 1.4
    turn_angle_rad: float = math.pi / 2.0
    back_turn_angle_rad: float = math.pi
    turn_tolerance_rad: float = 0.05
    min_turn_angular_z: float = 0.18
    search_linear_x: float = 0.06
    cooldown_linear_x: float = 0.08


@dataclass
class TurnGoal:
    direction: str
    target_yaw: float


class MotionController:
    def __init__(self, config: MotionControllerConfig | None = None):
        self.config = config or MotionControllerConfig()
        self.turn_goal: TurnGoal | None = None

    def stop(self) -> Twist:
        return Twist()

    def cruise(self) -> Twist:
        msg = Twist()
        msg.linear.x = self.config.cruise_linear_x
        return msg

    def search(self) -> Twist:
        msg = Twist()
        msg.linear.x = self.config.search_linear_x
        return msg

    def cooldown_forward(self) -> Twist:
        msg = Twist()
        msg.linear.x = self.config.cooldown_linear_x
        return msg

    def track_x_error(self, error_px: float, image_width: int) -> Twist:
        msg = Twist()
        msg.linear.x = self.config.track_linear_x
        msg.angular.z = self._angular_for_image_error(error_px, image_width)
        return msg

    def align_x_error(self, error_px: float, image_width: int) -> Twist:
        msg = Twist()
        msg.angular.z = self._angular_for_image_error(error_px, image_width)
        return msg

    def _angular_for_image_error(self, error_px: float, image_width: int) -> float:
        if image_width <= 0:
            return 0.0

        # Positive image error means target is right of center; turn right.
        angular = -self.config.track_kp * error_px
        return clamp(angular, -self.config.max_angular_z, self.config.max_angular_z)

    def start_turn(self, direction: str, current_yaw: float, angle_rad: float | None = None) -> None:
        if direction == 'left':
            turn = angle_rad if angle_rad is not None else self.config.turn_angle_rad
            target = current_yaw + abs(turn)
        elif direction == 'right':
            turn = angle_rad if angle_rad is not None else self.config.turn_angle_rad
            target = current_yaw - abs(turn)
        elif direction == 'back':
            turn = angle_rad if angle_rad is not None else self.config.back_turn_angle_rad
            target = current_yaw + abs(turn)
        else:
            self.turn_goal = None
            return

        self.turn_goal = TurnGoal(direction=direction, target_yaw=normalize_angle(target))

    def update_turn(self, current_yaw: float) -> tuple[Twist, bool]:
        if self.turn_goal is None:
            return self.stop(), True

        error = shortest_angular_distance(current_yaw, self.turn_goal.target_yaw)
        if abs(error) <= self.config.turn_tolerance_rad:
            self.turn_goal = None
            return self.stop(), True

        angular = clamp(
            self.config.turn_kp * error,
            -self.config.max_angular_z,
            self.config.max_angular_z,
        )
        if abs(angular) < self.config.min_turn_angular_z:
            angular = math.copysign(self.config.min_turn_angular_z, angular)

        msg = Twist()
        msg.angular.z = angular
        return msg, False
