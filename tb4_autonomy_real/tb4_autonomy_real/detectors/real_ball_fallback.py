from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tb4_autonomy_real.data_types import Box2D


@dataclass(frozen=True)
class RealBallCandidate:
    box: Box2D
    confidence: float
    mask: np.ndarray
    kind: str


def detect_real_colored_ball(frame, prefer: str | None = None) -> RealBallCandidate | None:
    """Detect either lab ball: red/blue patterned ball or small white ball.

    `prefer` lets the two task detectors display different balls when both are
    visible in the same frame, while still falling back to the other ball type.
    """
    red_blue = _detect_red_blue_ball(frame)
    white = _detect_white_ball(frame)
    if prefer == 'red_blue' and red_blue is not None:
        return red_blue
    if prefer == 'white' and white is not None:
        return white
    if red_blue is None:
        return white
    if white is None:
        return red_blue
    return red_blue if red_blue.confidence >= white.confidence else white


def _detect_red_blue_ball(frame) -> RealBallCandidate | None:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 70, 45]), np.array([12, 255, 255]))
        | cv2.inRange(hsv, np.array([165, 70, 45]), np.array([180, 255, 255]))
    )
    blue_mask = cv2.inRange(hsv, np.array([95, 55, 35]), np.array([135, 255, 255]))
    mask = red_mask | blue_mask
    mask[:int(height * 0.38), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    best = None
    best_score = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 800.0 or area > width * height * 0.18:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 2 or x + w >= width - 2:
            continue
        if y + h < height * 0.55:
            continue
        aspect = w / float(max(1, h))
        if aspect < 0.45 or aspect > 1.55:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0.0 else float(4.0 * np.pi * area / (perimeter**2))
        if circularity < 0.18:
            continue
        fill_ratio = area / float(max(1, w * h))
        score = float(np.clip(0.35 * min(1.0, area / 3500.0) + 0.35 * min(1.0, circularity / 0.55) + 0.30 * min(1.0, fill_ratio / 0.45), 0.0, 1.0))
        if score > best_score:
            best_score = score
            best = RealBallCandidate(Box2D(int(x), int(y), int(w), int(h)), score, mask, 'red_blue')
    return best


def _detect_white_ball(frame) -> RealBallCandidate | None:
    height, _width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 65, 255]))
    mask[:int(height * 0.55), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    best = None
    best_score = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 70.0 or area > 1200.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if y + h < height * 0.75:
            continue
        if w < 18 or h < 15:
            continue
        aspect = w / float(max(1, h))
        if aspect < 0.55 or aspect > 1.90:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter**2))
        if circularity < 0.55:
            continue
        fill_ratio = area / float(max(1, w * h))
        if fill_ratio < 0.35:
            continue
        score = float(np.clip(0.70 * circularity + 0.30 * min(1.0, fill_ratio), 0.0, 1.0))
        if score > best_score:
            best_score = score
            best = RealBallCandidate(Box2D(int(x), int(y), int(w), int(h)), score, mask, 'white')
    return best
