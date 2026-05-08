import numpy as np
import cv2

from tb4_autonomy.data_types import (
    ArrowDetection,
    BallDetection,
    Box2D,
    DetectionResults,
    FrameContext,
    LogoDetection,
    StaticBallDetection,
)
from tb4_autonomy.detectors.arrow_detector import ArrowDetector, ArrowDetectorConfig
from tb4_autonomy.detectors.horizon_detector import HorizonDetector
from tb4_autonomy.detectors.logo_detector import LogoDetector
from tb4_autonomy.detectors.moving_ball_detector import MovingBallDetector
from tb4_autonomy.detectors.static_ball_detector import StaticBallDetectorConfig, StaticObstacleDetector
from tb4_autonomy.utils.image_tools import draw_detections


def test_detector_interfaces_do_not_crash_on_empty_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detectors = [
        ArrowDetector(),
        LogoDetector(),
        MovingBallDetector(),
        StaticObstacleDetector(),
        HorizonDetector(),
    ]

    for detector in detectors:
        detector.detect(frame, context=None)


def test_horizon_placeholder_returns_visible_line():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = HorizonDetector(horizon_ratio=0.5).detect(frame, context=None)

    assert result is not None
    assert result.p1 == (0, 50)
    assert result.p2 == (199, 50)


def test_overlay_draws_all_result_types_without_crashing():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    results = DetectionResults(
        arrow=ArrowDetection(Box2D(10, 10, 20, 20), direction='left', confidence=1.0),
        logo=LogoDetection(Box2D(40, 10, 20, 20), confidence=1.0),
        moving_ball=BallDetection(Box2D(70, 10, 20, 20), moving=True, ttc=1.2, confidence=1.0),
        static_ball=StaticBallDetection(Box2D(100, 10, 20, 20), confidence=0.8),
        horizon=HorizonDetector().detect(frame, context=None),
    )

    draw_detections(frame, results)
    assert int(frame.sum()) > 0


def test_moving_ball_stop_hold_expires_when_ball_becomes_static():
    detector = MovingBallDetector()

    first = detector.detect(
        _synthetic_orange_ball_frame(cx=60),
        FrameContext(stamp_sec=0.0, odom_linear_x=0.05),
    )
    second = detector.detect(
        _synthetic_orange_ball_frame(cx=80),
        FrameContext(stamp_sec=0.1, odom_linear_x=0.05),
    )
    held = detector.detect(
        _synthetic_orange_ball_frame(cx=80),
        FrameContext(stamp_sec=0.2, odom_linear_x=0.05),
    )
    expired = detector.detect(
        _synthetic_orange_ball_frame(cx=80),
        FrameContext(stamp_sec=2.0, odom_linear_x=0.05),
    )

    assert first is not None and first.moving
    assert second is not None and second.moving
    assert held is not None and held.moving
    assert expired is None


def test_static_ball_detector_reports_round_flow_candidate():
    detector = StaticObstacleDetector(
        StaticBallDetectorConfig(
            min_area_px=200.0,
            min_circularity=0.35,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.60,
        )
    )

    first = detector.detect(_synthetic_white_ball_frame(cx=65), context=None)
    second = detector.detect(_synthetic_white_ball_frame(cx=90), context=None)

    assert first is None
    assert second is not None
    assert second.box.is_valid()
    assert second.confidence > 0.0


def test_arrow_detector_classifies_clear_synthetic_arrows():
    for direction in ('left', 'right', 'straight', 'back'):
        detector = ArrowDetector(
            ArrowDetectorConfig(
                history_size=1,
                min_stable_count=1,
                min_area_ratio=0.001,
                max_area_ratio=0.8,
                process_width=0,
            )
        )
        frame = _synthetic_arrow_scene(direction)
        result = detector.detect(frame, context=None)

        assert result is not None
        assert result.direction == direction
        assert result.raw_direction == direction
        assert result.is_stable
        assert result.box.is_valid()
        assert result.corners is not None


def test_arrow_detector_requires_temporal_agreement_for_stable_direction():
    detector = ArrowDetector(
        ArrowDetectorConfig(
            history_size=3,
            min_stable_count=2,
            min_area_ratio=0.001,
            max_area_ratio=0.8,
            process_width=0,
        )
    )
    frame = _synthetic_arrow_scene('left')

    first = detector.detect(frame, context=None)
    second = detector.detect(frame, context=None)

    assert first is not None
    assert first.raw_direction == 'left'
    assert first.direction == 'unknown'
    assert second is not None
    assert second.direction == 'left'


def _synthetic_arrow_scene(direction: str):
    frame = np.full((500, 500, 3), 180, dtype=np.uint8)
    sign = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(sign, (20, 20), (280, 280), (0, 0, 0), 8)
    points = _arrow_polygon(direction)
    cv2.fillPoly(sign, [np.array(points, dtype=np.int32)], (0, 0, 0))
    frame[100:400, 100:400] = sign
    return frame


def _synthetic_orange_ball_frame(cx: int):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (cx, 75), 18, (0, 128, 255), -1)
    return frame


def _synthetic_white_ball_frame(cx: int):
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.circle(frame, (cx, 95), 28, (255, 255, 255), -1)
    return frame


def _arrow_polygon(direction: str):
    right = np.array(
        [
            [65, 130],
            [170, 130],
            [170, 90],
            [245, 150],
            [170, 210],
            [170, 170],
            [65, 170],
        ],
        dtype=np.float32,
    )
    center = np.array([150, 150], dtype=np.float32)

    if direction == 'right':
        return right
    if direction == 'left':
        left = right.copy()
        left[:, 0] = 2 * center[0] - left[:, 0]
        return left
    if direction == 'straight':
        up = right.copy()
        shifted = up - center
        rotated = np.column_stack([shifted[:, 1], -shifted[:, 0]]) + center
        return rotated
    if direction == 'back':
        down = right.copy()
        shifted = down - center
        rotated = np.column_stack([-shifted[:, 1], shifted[:, 0]]) + center
        return rotated
    raise ValueError(direction)
