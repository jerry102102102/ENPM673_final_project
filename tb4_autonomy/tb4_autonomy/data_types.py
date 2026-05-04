from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AutonomyState(str, Enum):
    IDLE = 'IDLE'
    CRUISE = 'CRUISE'
    SEARCH_SIGN = 'SEARCH_SIGN'
    ALIGN_TO_SIGN = 'ALIGN_TO_SIGN'
    APPROACH_SIGN = 'APPROACH_SIGN'
    READ_ARROW = 'READ_ARROW'
    TRACK_ARROW = 'TRACK_ARROW'
    EXECUTE_TURN = 'EXECUTE_TURN'
    ARROW_COOLDOWN = 'ARROW_COOLDOWN'
    LOGO_STOP = 'LOGO_STOP'
    BALL_STOP = 'BALL_STOP'
    FINISHED = 'FINISHED'
    WAIT_FOR_TARGET = 'WAIT_FOR_TARGET'
    SMOOTH_ARC_TRACK = 'SMOOTH_ARC_TRACK'
    PASS_TO_NEXT = 'PASS_TO_NEXT'


@dataclass(frozen=True)
class Box2D:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def is_valid(self) -> bool:
        return self.w > 0 and self.h > 0


@dataclass
class ArrowDetection:
    box: Box2D
    direction: str = 'unknown'
    confidence: float = 0.0
    raw_direction: str = 'unknown'
    corners: tuple | None = None
    area_ratio: float = 0.0
    black_pixel_ratio: float = 0.0
    center_error_px: float = 0.0
    is_stable: bool = False
    arrow_angle_rad: float = 0.0
    warped_debug_image: object = None
    mask_debug_image: object = None

    # ===== 新畫圖功能相容欄位 =====
    heading_angle_rad: Optional[float] = None
    heading_angle_deg: Optional[float] = None
    heading_valid: bool = False
    heading_confidence: float = 0.0
    arrow_presence_confidence: float = 0.0
    heading_error_rad: Optional[float] = None
    heading_source: str = ''
    heading_base: Optional[tuple[int, int]] = None
    heading_tip: Optional[tuple[int, int]] = None
    final_confidence: float = 0.0
    black_arrow_direction: str = 'unknown'
    black_arrow_confidence: float = 0.0
    template_direction: str = 'unknown'
    template_dominance: float = 0.0
    axis_angle_rad: Optional[float] = None
    axis_direction: str = 'unknown'
    axis_confidence: float = 0.0
    paper_axis_angle_rad: Optional[float] = None
    paper_heading_angle_rad: Optional[float] = None
    warped_heading_debug_image: object = None

@dataclass
class LogoDetection:
    box: Box2D
    confidence: float = 0.0


@dataclass
class BallDetection:
    box: Box2D
    moving: bool = False
    ttc: float | None = None
    confidence: float = 0.0
    mask_debug_image: object = None


@dataclass
class HorizonDetection:
    p1: tuple[int, int]
    p2: tuple[int, int]
    confidence: float = 0.0


@dataclass
class DetectionResults:
    arrow: ArrowDetection | None = None
    logo: LogoDetection | None = None
    moving_ball: BallDetection | None = None
    horizon: HorizonDetection | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def has_moving_ball(self) -> bool:
        return self.moving_ball is not None and self.moving_ball.moving


@dataclass
class FrameContext:
    stamp_sec: float
    frame_id: str = ''
    image_width: int = 0
    image_height: int = 0
    odom_yaw: float | None = None
    odom_linear_x: float = 0.0
    camera_info: Any | None = None
    scan: Any | None = None


@dataclass(frozen=True)
class StateMachineOutput:
    state: AutonomyState
    entered_state: bool = False
    turn_direction: str | None = None
    turn_angle_rad: float | None = None
