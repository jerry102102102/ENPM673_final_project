#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2

from tb4_autonomy.arrow_smooth_arc_controller import ArrowSmoothArcConfig, ArrowSmoothArcController
from tb4_autonomy.data_types import FrameContext
from tb4_autonomy.detectors.arrow_detector import ArrowDetector, ArrowDetectorConfig
from tb4_autonomy.utils.image_tools import draw_detections, draw_status


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _params_from_yaml(path: Path) -> dict:
    data = _load_yaml(path)
    return data.get('vision_controller_node', {}).get('ros__parameters', {})


def _arrow_config(params: dict) -> ArrowDetectorConfig:
    return ArrowDetectorConfig(
        threshold_method=str(params.get('arrow_threshold_method', 'otsu')),
        hsv_v_max=int(params.get('arrow_hsv_v_max', 80)),
        blur_kernel=int(params.get('arrow_blur_kernel', 5)),
        morph_open_kernel=int(params.get('arrow_morph_open_kernel', 3)),
        morph_close_kernel=int(params.get('arrow_morph_close_kernel', 5)),
        dilate_kernel=int(params.get('arrow_dilate_kernel', 7)),
        dilate_iterations=int(params.get('arrow_dilate_iterations', 1)),
        min_area_ratio=float(params.get('arrow_min_area_ratio', 0.005)),
        max_area_ratio=float(params.get('arrow_max_area_ratio', 0.40)),
        min_area_px=float(params.get('arrow_min_area_px', 0.0)),
        max_area_px=float(params.get('arrow_max_area_px', 0.0)),
        min_aspect_ratio=float(params.get('arrow_min_aspect_ratio', 0.5)),
        max_aspect_ratio=float(params.get('arrow_max_aspect_ratio', 2.0)),
        min_black_pixel_ratio=float(params.get('arrow_min_black_pixel_ratio', 0.03)),
        warp_width=int(params.get('arrow_warp_width', 300)),
        warp_height=int(params.get('arrow_warp_height', 300)),
        inner_crop_margin_ratio=float(params.get('arrow_inner_crop_margin_ratio', 0.12)),
        history_size=int(params.get('arrow_history_size', 5)),
        min_stable_count=int(params.get('arrow_min_stable_count', 4)),
        process_width=int(params.get('arrow_process_width', 960)),
        bbox_padding_ratio=float(params.get('arrow_bbox_padding_ratio', 0.20)),
        fallback_bbox_padding_ratio=float(params.get('arrow_fallback_bbox_padding_ratio', 0.0)),
        merge_black_fragments_fallback=_as_bool(params.get('arrow_merge_black_fragments_fallback'), False),
        black_fragment_group_radius_px=float(params.get('arrow_black_fragment_group_radius_px', 55.0)),
        min_arrow_area_ratio=float(params.get('arrow_min_arrow_area_ratio', 0.01)),
        paper_v_min=int(params.get('arrow_paper_v_min', 130)),
        paper_s_max=int(params.get('arrow_paper_s_max', 90)),
        paper_min_area_ratio=float(params.get('arrow_paper_min_area_ratio', 0.001)),
        paper_max_area_ratio=float(params.get('arrow_paper_max_area_ratio', 0.25)),
        paper_min_area_px=float(params.get('arrow_paper_min_area_px', 0.0)),
        paper_max_area_px=float(params.get('arrow_paper_max_area_px', 0.0)),
        paper_min_aspect_ratio=float(params.get('arrow_paper_min_aspect_ratio', 0.3)),
        paper_max_aspect_ratio=float(params.get('arrow_paper_max_aspect_ratio', 12.0)),
        floor_roi_min_y_ratio=float(params.get('arrow_floor_roi_min_y_ratio', 0.45)),
        candidate_max_center_error_ratio=float(params.get('arrow_candidate_max_center_error_ratio', 0.45)),
        min_candidate_bottom_ratio=float(params.get('arrow_min_candidate_bottom_ratio', 0.45)),
        max_candidate_height_ratio=float(params.get('arrow_max_candidate_height_ratio', 0.58)),
        max_candidate_width_ratio=float(params.get('arrow_max_candidate_width_ratio', 0.75)),
        max_candidate_area_ratio=float(params.get('arrow_max_candidate_area_ratio', 0.18)),
        min_candidate_area_px=float(params.get('arrow_min_candidate_area_px', 0.0)),
        max_candidate_area_px=float(params.get('arrow_max_candidate_area_px', 0.0)),
        reject_back_direction=_as_bool(params.get('arrow_reject_back_direction'), True),
        max_valid_bbox_width_ratio=float(params.get('arrow_max_valid_bbox_width_ratio', 0.70)),
        max_valid_bbox_height_ratio=float(params.get('arrow_max_valid_bbox_height_ratio', 0.70)),
        max_valid_final_area_ratio=float(params.get('arrow_max_valid_final_area_ratio', 0.16)),
        min_valid_final_area_ratio=float(params.get('arrow_min_valid_final_area_ratio', 0.001)),
        max_border_touch_ratio=float(params.get('arrow_max_border_touch_ratio', 0.03)),
        max_valid_paper_aspect_ratio=float(params.get('arrow_max_valid_paper_aspect_ratio', 4.0)),
        min_valid_paper_aspect_ratio=float(params.get('arrow_min_valid_paper_aspect_ratio', 0.25)),
        min_history_confidence=float(params.get('arrow_min_history_confidence', 0.45)),
        use_axis_direction=bool(params.get('arrow_use_axis_direction', False)),
        use_paper_orientation_heading=bool(params.get('arrow_use_paper_orientation_heading', False)),
        paper_heading_forward_angle_rad=float(params.get('arrow_paper_heading_forward_angle_rad', 1.57079632679)),
        paper_heading_use_previous_when_ambiguous=bool(
            params.get('arrow_paper_heading_use_previous_when_ambiguous', True)
        ),
        paper_heading_ambiguity_margin_rad=float(params.get('arrow_paper_heading_ambiguity_margin_rad', 0.20)),
        min_arrow_presence_confidence=float(params.get('arrow_min_arrow_presence_confidence', 0.30)),
        min_arrow_component_density=float(params.get('arrow_min_arrow_component_density', 0.25)),
        min_arrow_component_solidity=float(params.get('arrow_min_arrow_component_solidity', 0.42)),
        min_arrow_component_compactness=float(params.get('arrow_min_arrow_component_compactness', 0.055)),
        max_arrow_component_area_ratio=float(params.get('arrow_max_arrow_component_area_ratio', 0.12)),
    )


def _controller_config(params: dict) -> ArrowSmoothArcConfig:
    cfg = params.get('arrow_smooth_arc_controller', {})
    return ArrowSmoothArcConfig(
        min_confidence=float(cfg.get('min_confidence', 0.45)),
        acquire_area_threshold=float(cfg.get('acquire_area_threshold', 0.020)),
        close_area_threshold=float(cfg.get('close_area_threshold', 0.24)),
        close_bottom_ratio=float(cfg.get('close_bottom_ratio', 0.995)),
        focal_px=float(cfg.get('focal_px', 600.0)),
        heading_sign=float(cfg.get('heading_sign', 1.0)),
        heading_scale=float(cfg.get('heading_scale', 0.15)),
        heading_oversteer_deg=float(cfg.get('heading_oversteer_deg', 2.0)),
        latched_yaw_alpha=float(cfg.get('latched_yaw_alpha', 0.20)),
        previous_heading_pull_gain=float(cfg.get('previous_heading_pull_gain', 0.22)),
        previous_heading_pull_decay_sec=float(cfg.get('previous_heading_pull_decay_sec', 1.30)),
        previous_heading_pull_start_delta_deg=float(cfg.get('previous_heading_pull_start_delta_deg', 12.0)),
        previous_heading_pull_max_angular_z=float(cfg.get('previous_heading_pull_max_angular_z', 0.035)),
        latched_heading_confidence_min=float(cfg.get('latched_heading_confidence_min', 0.55)),
        latched_arrow_presence_confidence_min=float(cfg.get('latched_arrow_presence_confidence_min', 0.45)),
        yaw_latch_alpha=float(cfg.get('yaw_latch_alpha', 0.15)),
        heading_sample_window=int(cfg.get('heading_sample_window', 8)),
        heading_sample_min_count=int(cfg.get('heading_sample_min_count', 3)),
        heading_sample_tolerance_deg=float(cfg.get('heading_sample_tolerance_deg', 30.0)),
        min_heading_confidence=float(cfg.get('min_heading_confidence', 0.65)),
        min_arrow_presence_confidence=float(cfg.get('min_arrow_presence_confidence', 0.45)),
        heading_update_max_area_ratio=float(cfg.get('heading_update_max_area_ratio', 0.09)),
        heading_update_max_bottom_ratio=float(cfg.get('heading_update_max_bottom_ratio', 0.95)),
        center_sign=float(cfg.get('center_sign', -1.0)),
        center_capture_threshold_px=float(cfg.get('center_capture_threshold_px', 90.0)),
        min_center_bias_gain=float(cfg.get('min_center_bias_gain', 0.10)),
        max_center_bias_gain=float(cfg.get('max_center_bias_gain', 0.60)),
        max_center_bias_deg=float(cfg.get('max_center_bias_deg', 10.0)),
        kp_yaw=float(cfg.get('kp_yaw', 1.2)),
        kp_center=float(cfg.get('kp_center', 0.70)),
        kp_heading=float(cfg.get('kp_heading', 0.45)),
        max_angular_z=float(cfg.get('max_angular_z', 0.18)),
        max_angular_accel=float(cfg.get('max_angular_accel', 0.35)),
        yaw_error_deadband_deg=float(cfg.get('yaw_error_deadband_deg', 3.0)),
        angular_lowpass_alpha=float(cfg.get('angular_lowpass_alpha', 0.18)),
        heading_confidence_soft_min=float(cfg.get('heading_confidence_soft_min', 0.40)),
        heading_confidence_full=float(cfg.get('heading_confidence_full', 0.80)),
        arrow_presence_confidence_soft_min=float(cfg.get('arrow_presence_confidence_soft_min', 0.35)),
        arrow_presence_confidence_full=float(cfg.get('arrow_presence_confidence_full', 0.80)),
        track_speed=float(cfg.get('track_speed', 0.030)),
        slow_track_speed=float(cfg.get('slow_track_speed', 0.022)),
        slow_yaw_error_deg=float(cfg.get('slow_yaw_error_deg', 25.0)),
        slow_center_error_px=float(cfg.get('slow_center_error_px', 100.0)),
        wait_linear_speed=float(cfg.get('wait_linear_speed', 0.025)),
        pass_speed=float(cfg.get('pass_speed', 0.035)),
        pass_time_sec=float(cfg.get('pass_time_sec', 0.15)),
        missing_detection_hold_sec=float(cfg.get('missing_detection_hold_sec', 0.20)),
        execute_latched_on_lost=_as_bool(cfg.get('execute_latched_on_lost'), True),
        pass_max_heading_error_deg=float(cfg.get('pass_max_heading_error_deg', 6.0)),
        finish_heading_timeout_sec=float(cfg.get('finish_heading_timeout_sec', 1.5)),
        finish_heading_speed=float(cfg.get('finish_heading_speed', 0.015)),
        finish_heading_kp=float(cfg.get('finish_heading_kp', 0.40)),
        finish_heading_max_angular_z=float(cfg.get('finish_heading_max_angular_z', 0.08)),
        finish_center_kp=float(cfg.get('finish_center_kp', 0.35)),
        finish_center_max_bias_deg=float(cfg.get('finish_center_max_bias_deg', 5.0)),
        finish_center_tolerance_px=float(cfg.get('finish_center_tolerance_px', 45.0)),
        min_track_time_sec=float(cfg.get('min_track_time_sec', 0.8)),
        close_camera_bottom_distance_in=float(cfg.get('close_camera_bottom_distance_in', 18.5)),
        close_arrow_bottom_distance_in=float(cfg.get('close_arrow_bottom_distance_in', 22.375)),
        post_close_min_travel_m=float(cfg.get('post_close_min_travel_m', 0.0)),
        post_close_speed=float(cfg.get('post_close_speed', 0.055)),
        post_close_timeout_sec=float(cfg.get('post_close_timeout_sec', 3.0)),
        active_target_area_drop_ratio=float(cfg.get('active_target_area_drop_ratio', 0.70)),
        active_target_bottom_drop_ratio=float(cfg.get('active_target_bottom_drop_ratio', 0.08)),
        debug_log=False,
    )


def _as_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    return None


def _paste_debug_thumbnail(canvas, image, title: str, x: int, y: int, width: int, height: int):
    image = _as_bgr(image)
    if image is None:
        return

    thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    title_h = max(12, int(height * 0.18))
    text_scale = max(0.28, min(0.45, width / 260.0))
    cv2.rectangle(thumb, (0, 0), (width, title_h), (0, 0, 0), -1)
    cv2.putText(thumb, title, (3, max(9, title_h - 4)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)

    h, w = canvas.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return

    x2 = min(w, x + width)
    y2 = min(h, y + height)
    canvas[y:y2, x:x2] = thumb[:y2 - y, :x2 - x]


def _draw_homography_debug_panel(canvas, detection):
    if detection is None:
        return

    h, w = canvas.shape[:2]
    low_res = min(h, w) <= 320
    panel_w = min(180, max(52, int(w * (0.18 if low_res else 0.28))))
    panel_h = panel_w
    gap = max(3, int(panel_w * 0.05))

    x = max(0, w - panel_w - max(4, int(w * 0.02)))

    warped = getattr(detection, 'warped_heading_debug_image', None)
    if warped is None:
        warped = getattr(detection, 'warped_debug_image', None)
    mask = getattr(detection, 'mask_debug_image', None)

    if low_res:
        # Keep the main scene visible for 250x250 lab recordings.
        y = max(0, h - panel_h - max(4, int(h * 0.02)))
        _paste_debug_thumbnail(canvas, mask if mask is not None else warped, 'MASK', x, y, panel_w, panel_h)
        return

    y = 12
    _paste_debug_thumbnail(canvas, warped, 'WARPED PAPER', x, y, panel_w, panel_h)
    _paste_debug_thumbnail(canvas, mask, 'ARROW MASK', x, y + panel_h + gap, panel_w, panel_h)


def _draw_analysis_status(image, lines: list[str]) -> None:
    height, width = image.shape[:2]
    scale = max(0.30, min(0.55, width / 700.0))
    thickness = 1
    line_height = max(12, int(24 * scale))
    max_lines = max(5, min(len(lines), (height - 8) // line_height))
    x = max(4, int(width * 0.02))
    y = max(12, int(line_height * 0.90))
    for line in lines[:max_lines]:
        cv2.putText(image, line, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height


def _safe_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _action_text(detection, debug: dict) -> str:
    state = str(debug.get('state', ''))
    direction = None if detection is None else detection.direction
    raw = None if detection is None else detection.raw_direction
    linear = float(debug.get('linear_x', 0.0))
    angular = float(debug.get('angular_z', 0.0))
    if detection is None:
        return f'NO TARGET: creep/search v={linear:.3f} w={angular:.3f}'
    if state == 'WAIT_FOR_TARGET':
        return f'WAIT: raw={raw} not acquired v={linear:.3f} w={angular:.3f}'
    if state == 'PASS_TO_NEXT':
        return f'PASS: leave current paper v={linear:.3f} w={angular:.3f}'
    if angular > 0.04:
        turn = 'smooth arc LEFT'
    elif angular < -0.04:
        turn = 'smooth arc RIGHT'
    else:
        turn = 'forward / tiny correction'
    return f'{turn}: stable={direction} raw={raw} v={linear:.3f} w={angular:.3f}'


def _write_summary(rows: list[dict], summary_path: Path) -> None:
    total = len(rows)
    detected = sum(row['raw_direction'] != 'none' for row in rows)
    stable = sum(row['stable'] == 'true' for row in rows)
    by_raw: dict[str, int] = {}
    by_stable: dict[str, int] = {}
    for row in rows:
        by_raw[row['raw_direction']] = by_raw.get(row['raw_direction'], 0) + 1
        by_stable[row['direction']] = by_stable.get(row['direction'], 0) + 1
    with summary_path.open('w') as file:
        file.write(f'total_frames: {total}\n')
        file.write(f'detected_frames: {detected}\n')
        file.write(f'stable_frames: {stable}\n')
        file.write(f'detected_ratio: {detected / total if total else 0.0:.3f}\n')
        file.write(f'stable_ratio: {stable / total if total else 0.0:.3f}\n')
        file.write(f'raw_direction_counts: {by_raw}\n')
        file.write(f'stable_direction_counts: {by_stable}\n')


def analyze(video_path: Path, output_path: Path, csv_path: Path, summary_path: Path, config_path: Path) -> None:
    params = _params_from_yaml(config_path)
    detector = ArrowDetector(_arrow_config(params))
    controller = ArrowSmoothArcController(_controller_config(params))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'failed to open input video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'failed to open output video: {output_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with csv_path.open('w', newline='') as csv_file:
        fieldnames = [
            'frame',
            'time_sec',
            'direction',
            'raw_direction',
            'stable',
            'confidence',
            'area_ratio',
            'box_x',
            'box_y',
            'box_w',
            'box_h',
            'box_area_px',
            'box_bottom_ratio',
            'center_error_px',
            'heading_error_rad',
            'heading_angle_deg',
            'heading_valid',
            'heading_confidence',
            'arrow_presence_confidence',
            'heading_source',
            'paper_axis_angle_rad',
            'paper_heading_angle_rad',
            'black_arrow_direction',
            'black_arrow_confidence',
            'axis_confidence',
            'axis_direction',
            'template_direction',
            'template_dominance',
            'state',
            'control_mode',
            'heading_weight',
            'heading_term',
            'previous_heading_pull_weight',
            'previous_heading_pull_term',
            'center_bias_deg',
            'yaw_error_deg',
            'linear_x',
            'angular_z',
            'action',
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now_sec = frame_index / fps
            context = FrameContext(
                stamp_sec=now_sec,
                frame_id='offline_video',
                image_width=width,
                image_height=height,
                odom_yaw=0.0,
            )
            detection = detector.detect(frame, context)
            output = controller.update(detection, width, height, now_sec, 0.0)
            debug = output.debug_info
            action = _action_text(detection, debug)

            annotated = frame.copy()
            if detection is not None:
                from tb4_autonomy.data_types import DetectionResults
                draw_detections(annotated, DetectionResults(arrow=detection))
            box_area_px = 0 if detection is None else int(detection.box.area)
            box_text = 'box=none' if detection is None else (
                f'box={detection.box.w}x{detection.box.h}px area={box_area_px}px bottom={(detection.box.y + detection.box.h) / height:.2f}'
            )
            _draw_analysis_status(
                annotated,
                [
                    f'FRAME: {frame_index}  T: {now_sec:.2f}s',
                    f'raw={None if detection is None else detection.raw_direction} dir={debug.get("direction")} stable={debug.get("is_stable")} conf={float(debug.get("confidence", 0.0)):.2f}',
                    box_text,
                    f'center={float(debug.get("center_error_px", 0.0)):.0f}px head_valid={False if detection is None else detection.heading_valid} head_conf={0.0 if detection is None else detection.heading_confidence:.2f}',
                    f'state={debug.get("state")} mode={debug.get("control_mode")}',
                    f'cmd v={float(debug.get("linear_x", 0.0)):.3f} w={float(debug.get("angular_z", 0.0)):.3f}',
                ],
            )
            _draw_homography_debug_panel(annotated, detection)
            writer.write(annotated)

            row = {
                'frame': frame_index,
                'time_sec': f'{now_sec:.3f}',
                'direction': 'none' if detection is None else detection.direction,
                'raw_direction': 'none' if detection is None else detection.raw_direction,
                'stable': 'false' if detection is None else str(detection.is_stable).lower(),
                'confidence': f'{float(debug.get("confidence", 0.0)):.6f}',
                'area_ratio': f'{float(debug.get("area_ratio", 0.0)):.6f}',
                'box_x': '0' if detection is None else str(detection.box.x),
                'box_y': '0' if detection is None else str(detection.box.y),
                'box_w': '0' if detection is None else str(detection.box.w),
                'box_h': '0' if detection is None else str(detection.box.h),
                'box_area_px': '0' if detection is None else str(int(detection.box.area)),
                'box_bottom_ratio': '0.000000' if detection is None else f'{(detection.box.y + detection.box.h) / height:.6f}',
                'center_error_px': f'{float(debug.get("center_error_px", 0.0)):.3f}',
                'heading_error_rad': '0.000000' if detection is None or detection.heading_error_rad is None else f'{detection.heading_error_rad:.6f}',
                'heading_angle_deg': '0.000000' if detection is None or detection.heading_angle_deg is None else f'{detection.heading_angle_deg:.6f}',
                'heading_valid': 'false' if detection is None else str(detection.heading_valid).lower(),
                'heading_confidence': '0.000000' if detection is None else f'{detection.heading_confidence:.6f}',
                'arrow_presence_confidence': '0.000000' if detection is None else f'{detection.arrow_presence_confidence:.6f}',
                'heading_source': 'none' if detection is None else detection.heading_source,
                'paper_axis_angle_rad': '0.000000' if detection is None or detection.paper_axis_angle_rad is None else f'{detection.paper_axis_angle_rad:.6f}',
                'paper_heading_angle_rad': '0.000000' if detection is None or detection.paper_heading_angle_rad is None else f'{detection.paper_heading_angle_rad:.6f}',
                'black_arrow_direction': 'none' if detection is None else detection.black_arrow_direction,
                'black_arrow_confidence': '0.000000' if detection is None else f'{detection.black_arrow_confidence:.6f}',
                'axis_confidence': '0.000000' if detection is None else f'{detection.axis_confidence:.6f}',
                'axis_direction': 'none' if detection is None else detection.axis_direction,
                'template_direction': 'none' if detection is None else detection.template_direction,
                'template_dominance': '0.000000' if detection is None else f'{detection.template_dominance:.6f}',
                'state': str(debug.get('state')),
                'control_mode': str(debug.get('control_mode')),
                'heading_weight': f'{float(debug.get("heading_weight", 0.0)):.6f}',
                'heading_term': f'{float(debug.get("heading_term", 0.0)):.6f}',
                'previous_heading_pull_weight': f'{float(debug.get("previous_heading_pull_weight", 0.0)):.6f}',
                'previous_heading_pull_term': f'{float(debug.get("previous_heading_pull_term", 0.0)):.6f}',
                'center_bias_deg': f'{float(debug.get("center_bias_deg", 0.0)):.3f}',
                'yaw_error_deg': f'{float(debug.get("yaw_error_deg", 0.0)):.3f}',
                'linear_x': f'{float(debug.get("linear_x", 0.0)):.6f}',
                'angular_z': f'{float(debug.get("angular_z", 0.0)):.6f}',
                'action': action,
            }
            rows.append(row)
            csv_writer.writerow(row)
            frame_index += 1

    cap.release()
    writer.release()
    _write_summary(rows, summary_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--config', type=Path, default=Path('tb4_autonomy/config/offline_video_analysis.yaml'))
    args = parser.parse_args()

    video_path = args.video
    if args.output is None:
        args.output = video_path.with_name(video_path.stem + '_arrow_overlay.mp4')
    if args.csv is None:
        args.csv = args.output.with_suffix('.csv')
    if args.summary is None:
        args.summary = args.output.with_suffix('.summary.txt')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    analyze(video_path, args.output, args.csv, args.summary, args.config)
    print(f'wrote {args.output}')
    print(f'wrote {args.csv}')
    print(f'wrote {args.summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
