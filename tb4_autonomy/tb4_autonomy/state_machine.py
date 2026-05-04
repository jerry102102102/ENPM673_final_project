#state_machine.py
from __future__ import annotations

from dataclasses import dataclass

from tb4_autonomy.data_types import AutonomyState, DetectionResults, StateMachineOutput


@dataclass
class StateMachineConfig:
    logo_stop_s: float = 3.0
    center_tolerance_px: float = 40.0
    target_bbox_area_ratio: float = 0.08
    read_timeout_s: float = 3.0
    cooldown_s: float = 1.5


class StateMachine:
    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self.state = AutonomyState.SEARCH_SIGN
        self.previous_navigation_state = AutonomyState.SEARCH_SIGN
        self.logo_stop_until: float | None = None
        self.logo_armed = True
        self.active_turn_direction: str | None = None
        self.active_turn_angle_rad: float | None = None
        self.read_until: float | None = None
        self.cooldown_until: float | None = None

    def update(
        self,
        detections: DetectionResults,
        now_sec: float,
        turn_complete: bool = False,
    ) -> StateMachineOutput:
        old_state = self.state

        if detections.logo is None and self.state != AutonomyState.LOGO_STOP:
            self.logo_armed = True

        if detections.has_moving_ball:
            self._remember_navigation_state()
            self.state = AutonomyState.BALL_STOP
            return self._output(old_state)
        if detections.static_ball is not None:
            self._remember_navigation_state()
            self.state = AutonomyState.BALL_STOP
            return self._output(old_state)

        if self.state == AutonomyState.BALL_STOP:
            self.state = self.previous_navigation_state
            return self._output(old_state)

        if self.state == AutonomyState.LOGO_STOP:
            if self.logo_stop_until is not None and now_sec < self.logo_stop_until:
                return self._output(old_state)
            self.logo_stop_until = None
            self.state = self.previous_navigation_state
            if detections.logo is None:
                self.logo_armed = True
            return self._output(old_state)

        if detections.logo is not None and self.logo_armed:
            self._remember_navigation_state()
            self.state = AutonomyState.LOGO_STOP
            self.logo_stop_until = now_sec + self.config.logo_stop_s
            self.logo_armed = False
            return self._output(old_state)

        if self.state == AutonomyState.EXECUTE_TURN:
            if turn_complete:
                self.active_turn_direction = None
                self.active_turn_angle_rad = None
                self.cooldown_until = now_sec + self.config.cooldown_s
                self.state = AutonomyState.ARROW_COOLDOWN
            return self._output(old_state)

        if self.state == AutonomyState.ARROW_COOLDOWN:
            if self.cooldown_until is not None and now_sec < self.cooldown_until:
                return self._output(old_state)
            self.cooldown_until = None
            self.state = AutonomyState.SEARCH_SIGN
            return self._output(old_state)

        arrow = detections.arrow

        if self.state == AutonomyState.READ_ARROW:
            if arrow is not None and arrow.is_stable and arrow.direction != 'unknown':
                self._consume_stable_arrow(arrow.direction, now_sec, arrow.arrow_angle_rad)
                return self._output(old_state)
            if self.read_until is not None and now_sec > self.read_until:
                self.cooldown_until = now_sec + self.config.cooldown_s
                self.state = AutonomyState.ARROW_COOLDOWN
            return self._output(old_state)

        if arrow is None:
            self.state = AutonomyState.SEARCH_SIGN
            return self._output(old_state)

        if self.state in (AutonomyState.SEARCH_SIGN, AutonomyState.CRUISE, AutonomyState.TRACK_ARROW):
            self.state = AutonomyState.ALIGN_TO_SIGN
            return self._output(old_state)

        if self.state == AutonomyState.ALIGN_TO_SIGN:
            if abs(arrow.center_error_px) <= self.config.center_tolerance_px:
                self.state = AutonomyState.APPROACH_SIGN
            return self._output(old_state)

        if self.state == AutonomyState.APPROACH_SIGN:
            if arrow.area_ratio >= self.config.target_bbox_area_ratio:
                self.state = AutonomyState.READ_ARROW
                self.read_until = now_sec + self.config.read_timeout_s
            return self._output(old_state)

        if self.state not in (AutonomyState.IDLE, AutonomyState.FINISHED):
            self.state = AutonomyState.SEARCH_SIGN

        return self._output(old_state)

    def _consume_stable_arrow(self, direction: str, now_sec: float, angle_rad: float = 0.0) -> None:
        self.read_until = None
        if direction == 'end':
            self.state = AutonomyState.FINISHED
        elif direction in ('left', 'right', 'back'):
            self.active_turn_direction = direction
            self.active_turn_angle_rad = angle_rad
            self.state = AutonomyState.EXECUTE_TURN
        elif direction == 'straight':
            self.cooldown_until = now_sec + self.config.cooldown_s
            self.state = AutonomyState.ARROW_COOLDOWN
        else:
            self.state = AutonomyState.READ_ARROW

    def _remember_navigation_state(self) -> None:
        if self.state not in (
            AutonomyState.BALL_STOP,
            AutonomyState.LOGO_STOP,
            AutonomyState.IDLE,
            AutonomyState.FINISHED,
        ):
            self.previous_navigation_state = self.state

    def _output(self, old_state: AutonomyState) -> StateMachineOutput:
        entered = self.state != old_state
        return StateMachineOutput(
            state=self.state,
            entered_state=entered,
            turn_direction=self.active_turn_direction,
            turn_angle_rad=self.active_turn_angle_rad,
        )
