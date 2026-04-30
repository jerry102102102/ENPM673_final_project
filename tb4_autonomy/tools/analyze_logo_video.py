#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / 'tb4_autonomy'
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from tb4_autonomy.data_types import DetectionResults, FrameContext
from tb4_autonomy.detectors.logo_detector import LogoDetector
from tb4_autonomy.utils.image_tools import draw_detections, draw_status


def _output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f'{input_path.stem}_{suffix}{input_path.suffix}')


def analyze_video(input_path: Path, output_path: Path | None, max_frames: int | None) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_path or _output_path(input_path, 'logo_overlay')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Unable to write video: {output_path}')

    detector = LogoDetector()
    frame_index = 0
    detected_frames = 0
    confirmed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        context = FrameContext(
            stamp_sec=frame_index / fps,
            image_width=width,
            image_height=height,
        )
        logo = detector.detect(frame, context)
        if detector.detect_count > 0:
            detected_frames += 1
        if logo is not None:
            confirmed_frames += 1

        results = DetectionResults(logo=logo)
        overlay = frame.copy()
        draw_detections(overlay, results)
        draw_status(
            overlay,
            [
                f'FRAME: {frame_index}  T: {frame_index / fps:.2f}s',
                f'LOGO_CONFIRMED: {logo is not None}',
                f'CONSECUTIVE_COUNT: {detector.detect_count}/{detector.detect_threshold}',
                f'MIN_MATCHES: {detector.min_matches}  RATIO: {detector.ratio_thresh:.2f}',
                f'CONF: {0.0 if logo is None else logo.confidence:.1f}',
            ],
        )
        writer.write(overlay)
        frame_index += 1

    cap.release()
    writer.release()
    print(f'wrote {frame_index} frames to {output_path}')
    print(f'detected_frames={detected_frames} confirmed_frames={confirmed_frames}')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run UMD logo detector on a recorded camera video.')
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    analyze_video(args.video, args.output, args.max_frames)


if __name__ == '__main__':
    main()
