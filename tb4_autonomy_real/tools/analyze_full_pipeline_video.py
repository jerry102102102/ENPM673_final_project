#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / 'tb4_autonomy_real'
TOOLS_ROOT = PKG_ROOT / 'tools'
for path in (PKG_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_arrow_video import _arrow_config, _controller_config, _params_from_yaml
from tb4_autonomy_real.arrow_smooth_arc_controller import ArrowSmoothArcController
from tb4_autonomy_real.data_types import AutonomyState, DetectionResults, FrameContext
from tb4_autonomy_real.detectors.arrow_detector import ArrowDetector
from tb4_autonomy_real.detectors.horizon_detector import HorizonDetector
from tb4_autonomy_real.detectors.logo_detector import LogoDetector
from tb4_autonomy_real.detectors.moving_ball_detector import MovingBallDetector
from tb4_autonomy_real.detectors.static_ball_detector import StaticObstacleDetector
from tb4_autonomy_real.utils.image_tools import draw_detections, draw_status


def _bool_param(params: dict, key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _line_arrow(results: DetectionResults) -> str:
    if results.arrow is None:
        return 'arrow: none'
    arrow = results.arrow
    return (
        f'arrow: stable={arrow.direction} raw={arrow.raw_direction} '
        f'area={arrow.area_ratio:.3f} err={arrow.center_error_px:.0f}px '
        f'valid={arrow.heading_valid} '
        f'head={0.0 if arrow.heading_error_rad is None else arrow.heading_error_rad:.2f} '
        f'angle={0.0 if arrow.heading_angle_deg is None else arrow.heading_angle_deg:.1f}deg '
        f'exist={arrow.arrow_presence_confidence:.2f} '
        f'black={arrow.black_arrow_direction}:{arrow.black_arrow_confidence:.2f} '
        f'box=({arrow.box.x},{arrow.box.y},{arrow.box.w},{arrow.box.h})'
    )


def _line_logo(results: DetectionResults, logo_detector: LogoDetector, stop_remaining: float) -> str:
    threshold = getattr(logo_detector, 'detect_threshold', 1)
    if results.logo is None:
        return f'logo: count={logo_detector.detect_count}/{threshold} stop={stop_remaining:.1f}s'
    box = results.logo.box
    return (
        f'logo: CONFIRMED count={logo_detector.detect_count}/{threshold} '
        f'conf={results.logo.confidence:.1f} '
        f'box=({box.x},{box.y},{box.w},{box.h}) stop={stop_remaining:.1f}s'
    )


def _line_ball(results: DetectionResults) -> str:
    if results.moving_ball is not None:
        ball = results.moving_ball
        return (
            f'moving_ball: conf={ball.confidence:.2f} ttc={ball.ttc} '
            f'box=({ball.box.x},{ball.box.y},{ball.box.w},{ball.box.h})'
        )
    if results.static_ball is not None:
        ball = results.static_ball
        return (
            f'static_ball: conf={ball.confidence:.2f} '
            f'box=({ball.box.x},{ball.box.y},{ball.box.w},{ball.box.h})'
        )
    return 'ball: none'


def analyze(video_path: Path, output_path: Path, config_path: Path, dry_run: bool = True) -> None:
    params = _params_from_yaml(config_path)

    arrow_detector = ArrowDetector(_arrow_config(params))
    logo_detector = LogoDetector(detect_threshold=int(params.get('logo_confirm_frames', 5)))
    moving_ball_detector = MovingBallDetector()
    static_ball_detector = StaticObstacleDetector()
    horizon_detector = HorizonDetector(
        float(params.get('horizon_ratio', 0.5)),
        roi_top_ratio=float(params.get('horizon_roi_top_ratio', 0.42)),
        roi_bottom_ratio=float(params.get('horizon_roi_bottom_ratio', 0.60)),
    )
    arrow_controller = ArrowSmoothArcController(_controller_config(params))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'failed to open input video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'failed to open output video: {output_path}')

    logo_stop_s = float(params.get('logo_stop_s', 3.0))
    logo_stop_until: float | None = None
    logo_armed = True

    counts = {
        'frames': 0,
        'arrow': 0,
        'logo': 0,
        'moving_ball': 0,
        'static_ball': 0,
        'horizon': 0,
        'logo_stop': 0,
        'ball_stop': 0,
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_start = time.perf_counter()
        now_sec = counts['frames'] / fps
        context = FrameContext(
            stamp_sec=now_sec,
            frame_id='offline_full_pipeline',
            image_width=width,
            image_height=height,
            odom_yaw=0.0,
            odom_linear_x=0.03,
        )

        timings_ms: dict[str, float] = {}
        results = DetectionResults()
        for name, detector in (
            ('arrow', arrow_detector),
            ('logo', logo_detector),
            ('moving_ball', moving_ball_detector),
            ('static_ball', static_ball_detector),
            ('horizon', horizon_detector),
        ):
            start = time.perf_counter()
            result = detector.detect(frame, context)
            timings_ms[name] = (time.perf_counter() - start) * 1000.0
            setattr(results, name if name != 'static_ball' else 'static_ball', result)

        if results.arrow is not None:
            counts['arrow'] += 1
        if results.logo is not None:
            counts['logo'] += 1
        if results.moving_ball is not None:
            counts['moving_ball'] += 1
        if results.static_ball is not None:
            counts['static_ball'] += 1
        if results.horizon is not None:
            counts['horizon'] += 1

        state = arrow_controller.state
        if results.has_moving_ball or results.has_static_ball:
            state = AutonomyState.BALL_STOP
            arrow_controller.reset()
            controller_debug = {
                'state': state.value,
                'control_mode': 'moving_ball_stop' if results.has_moving_ball else 'static_ball_stop',
                'linear_x': 0.0,
                'angular_z': 0.0,
            }
            counts['ball_stop'] += 1
        elif logo_stop_until is not None and now_sec < logo_stop_until:
            state = AutonomyState.LOGO_STOP
            arrow_controller.reset()
            controller_debug = {
                'state': state.value,
                'control_mode': 'logo_stop_hold',
                'linear_x': 0.0,
                'angular_z': 0.0,
            }
            counts['logo_stop'] += 1
        elif logo_stop_until is not None:
            logo_stop_until = None
            output = arrow_controller.update(results.arrow, width, height, now_sec, 0.0, None)
            state = output.current_state
            controller_debug = output.debug_info
        elif results.logo is not None and logo_armed:
            state = AutonomyState.LOGO_STOP
            logo_stop_until = now_sec + logo_stop_s
            logo_armed = False
            arrow_controller.reset()
            controller_debug = {
                'state': state.value,
                'control_mode': 'logo_confirmed_stop',
                'linear_x': 0.0,
                'angular_z': 0.0,
            }
            counts['logo_stop'] += 1
        else:
            output = arrow_controller.update(results.arrow, width, height, now_sec, 0.0, None)
            state = output.current_state
            controller_debug = output.debug_info

        if dry_run:
            controller_debug['linear_x'] = 0.0
            controller_debug['angular_z'] = 0.0

        results.timings_ms = timings_ms
        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        remaining = 0.0 if logo_stop_until is None else max(0.0, logo_stop_until - now_sec)

        overlay = frame.copy()
        draw_detections(overlay, results)
        draw_status(
            overlay,
            [
                f'STATE: {state.value}',
                f'dry_run: {dry_run}',
                f'frame: {frame_ms:.1f} ms',
                _line_logo(results, logo_detector, remaining),
                _line_ball(results),
                _line_arrow(results),
                f'CTRL: {controller_debug.get("control_mode", "")}',
                f'AREA: {float(controller_debug.get("area_ratio", 0.0)):.3f}',
                f'CMD: v={float(controller_debug.get("linear_x", 0.0)):.3f}, w={float(controller_debug.get("angular_z", 0.0)):.3f}',
                'timing: ' + ' '.join(f'{key}={value:.1f}ms' for key, value in timings_ms.items()),
            ],
        )
        writer.write(overlay)
        counts['frames'] += 1

    cap.release()
    writer.release()

    summary_path = output_path.with_suffix('.summary.txt')
    with summary_path.open('w') as file:
        for key, value in counts.items():
            file.write(f'{key}: {value}\n')
    print(f'wrote {output_path}')
    print(f'wrote {summary_path}')
    print(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path('tb4_autonomy_real/config/arrow_detection_real_scene.yaml'))
    parser.add_argument('--dry-run', action='store_true', default=True)
    args = parser.parse_args()
    output = args.output or args.video.with_name(args.video.stem + '_full_pipeline_overlay.mp4')
    analyze(args.video, output, args.config, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
