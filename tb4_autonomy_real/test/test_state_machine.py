from tb4_autonomy_real.data_types import (
    ArrowDetection,
    AutonomyState,
    BallDetection,
    Box2D,
    DetectionResults,
    LogoDetection,
)
from tb4_autonomy_real.state_machine import StateMachine, StateMachineConfig


def test_logo_detection_enters_timed_stop_then_returns_to_cruise():
    sm = StateMachine(StateMachineConfig(logo_stop_s=3.0))
    logo = DetectionResults(logo=LogoDetection(Box2D(10, 10, 20, 20), confidence=1.0))

    output = sm.update(logo, now_sec=10.0)
    assert output.state == AutonomyState.LOGO_STOP

    output = sm.update(logo, now_sec=12.9)
    assert output.state == AutonomyState.LOGO_STOP

    output = sm.update(logo, now_sec=13.1)
    assert output.state == AutonomyState.SEARCH_SIGN

    output = sm.update(logo, now_sec=13.2)
    assert output.state == AutonomyState.SEARCH_SIGN


def test_logo_rearms_after_logo_disappears():
    sm = StateMachine(StateMachineConfig(logo_stop_s=0.1))
    logo = DetectionResults(logo=LogoDetection(Box2D(10, 10, 20, 20), confidence=1.0))

    assert sm.update(logo, now_sec=1.0).state == AutonomyState.LOGO_STOP
    assert sm.update(DetectionResults(), now_sec=1.2).state == AutonomyState.SEARCH_SIGN
    assert sm.update(logo, now_sec=1.3).state == AutonomyState.LOGO_STOP


def test_moving_ball_has_highest_priority():
    sm = StateMachine()
    logo = LogoDetection(Box2D(10, 10, 20, 20), confidence=1.0)
    ball = BallDetection(Box2D(30, 30, 20, 20), moving=True, confidence=1.0)

    output = sm.update(DetectionResults(logo=logo, moving_ball=ball), now_sec=5.0)
    assert output.state == AutonomyState.BALL_STOP

    output = sm.update(DetectionResults(logo=logo), now_sec=5.1)
    assert output.state == AutonomyState.SEARCH_SIGN


def test_arrow_pipeline_aligns_approaches_reads_then_turns():
    sm = StateMachine(
        StateMachineConfig(
            center_tolerance_px=40.0,
            target_bbox_area_ratio=0.08,
            read_timeout_s=3.0,
            cooldown_s=1.5,
        )
    )
    far_left_arrow = ArrowDetection(
        Box2D(10, 10, 20, 20),
        direction='unknown',
        raw_direction='left',
        center_error_px=-100.0,
        area_ratio=0.02,
        confidence=1.0,
    )
    centered_small_arrow = ArrowDetection(
        Box2D(100, 10, 20, 20),
        direction='unknown',
        raw_direction='left',
        center_error_px=10.0,
        area_ratio=0.02,
        confidence=1.0,
    )
    centered_large_stable_arrow = ArrowDetection(
        Box2D(100, 10, 80, 80),
        direction='left',
        raw_direction='left',
        center_error_px=10.0,
        area_ratio=0.10,
        confidence=1.0,
        is_stable=True,
    )

    output = sm.update(DetectionResults(arrow=far_left_arrow), now_sec=1.0)
    assert output.state == AutonomyState.ALIGN_TO_SIGN

    output = sm.update(DetectionResults(arrow=centered_small_arrow), now_sec=1.1)
    assert output.state == AutonomyState.APPROACH_SIGN

    output = sm.update(DetectionResults(arrow=centered_large_stable_arrow), now_sec=1.2)
    assert output.state == AutonomyState.READ_ARROW

    output = sm.update(DetectionResults(arrow=centered_large_stable_arrow), now_sec=1.3)
    assert output.state == AutonomyState.EXECUTE_TURN
    assert output.turn_direction == 'left'

    output = sm.update(DetectionResults(), now_sec=1.4, turn_complete=True)
    assert output.state == AutonomyState.ARROW_COOLDOWN

    output = sm.update(DetectionResults(), now_sec=3.0)
    assert output.state == AutonomyState.SEARCH_SIGN
