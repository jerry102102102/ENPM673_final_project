# =============================================================================
# ENPM673 - Perception for Autonomous Robots | Spring 2026
# TurtleBot4 Final Project - Task 3: Dynamic Object Detection
# Author:  Swathi
# 
# Detects a moving ball and computes Time To Contact (TTC).
# Two detection paths:
#   Simulation: orange HSV filter + static object rejection
#   Real robot: real_ball_fallback (handles both lab balls)
#               real_candidate bypasses static check
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
from tb4_autonomy_real.data_types import BallDetection, Box2D, FrameContext
from tb4_autonomy_real.detectors.real_ball_fallback import detect_real_colored_ball


class MovingBallDetector:
    name = "moving_ball"

    # Simulation - Webots orange ball
    # house.wbt baseColor 1 0.5 0, radius=0.032m
    ORANGE_HSV_LOWER = np.array([12, 150, 150], dtype=np.uint8)
    ORANGE_HSV_UPPER = np.array([22, 255, 255], dtype=np.uint8)
    MIN_BALL_AREA = 400
    MAX_BALL_AREA = 55000

    def __init__(self):
        self.prev_bbox_area = None
        self.prev_stamp = None
        self.prev_center = None
        self.PERSISTENCE_FRAMES = 3
        self.persistence_counter = 0
        self.last_box = None
        self.last_ttc = None
        self.last_mask = None
        self.min_stop_duration_s = 1.5
        self.stop_until = None
        self.static_count = 0
        self.MAX_STATIC_FRAMES = 5

    def _find_ball_simulation(self, frame):
        """Orange HSV detector for Webots simulation ball."""
        height = frame.shape[0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(
            hsv, self.ORANGE_HSV_LOWER, self.ORANGE_HSV_UPPER)
        # Ball always on floor - exclude top 30%
        color_mask[:int(height * 0.30), :] = 0
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        color_mask = cv2.dilate(
            color_mask, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_BALL_AREA or area > self.MAX_BALL_AREA:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best_contour = contour
        return best_contour, best_score, color_mask

    def detect(self, frame, context: FrameContext) -> BallDetection | None:
        if frame is None:
            return None
        now = 0.0 if context is None else context.stamp_sec

        # Step 1: Try simulation orange detector
        ball_contour, best_score, mask_debug_image = self._find_ball_simulation(frame)

        # Step 2: Try real ball fallback if simulation found nothing
        # prefer=red_blue tries PSG soccer ball first, then white baseball
        real_candidate = None
        if ball_contour is None:
            real_candidate = detect_real_colored_ball(frame, prefer="red_blue")

        # Nothing found at all
        if ball_contour is None and real_candidate is None:
            self.prev_center = None
            self.static_count = 0
            if self.persistence_counter > 0:
                self.persistence_counter -= 1
                if self.last_box is not None:
                    return BallDetection(
                        box=self.last_box, moving=True,
                        ttc=self.last_ttc, confidence=0.3,
                        mask_debug_image=self.last_mask)
            if self.stop_until is not None and now < self.stop_until:
                if self.last_box is not None:
                    return BallDetection(
                        box=self.last_box, moving=True,
                        ttc=self.last_ttc, confidence=0.2,
                        mask_debug_image=self.last_mask)
            self.prev_bbox_area = None
            self.stop_until = None
            return None

        # Get bounding box from whichever detector fired
        if real_candidate is not None:
            box = real_candidate.box
            confidence = real_candidate.confidence
            mask_debug_image = real_candidate.mask
            using_real = True
        else:
            x, y, w, h = cv2.boundingRect(ball_contour)
            box = Box2D(x=x, y=y, w=w, h=h)
            confidence = min(1.0, best_score / 10000.0)
            using_real = False

        cx, cy = box.center
        center = (cx, cy)
        is_moving = self.prev_center is None

        if self.prev_center is not None:
            movement = (
                (cx - self.prev_center[0])**2 +
                (cy - self.prev_center[1])**2
            ) ** 0.5

            if using_real:
                # Real ball fallback already filters false positives
                # Skip static check - baseball moves slowly (<5px at 15Hz)
                # and would be wrongly rejected
                is_moving = True
                self.static_count = 0
                if self.stop_until is None:
                    self.stop_until = now + self.min_stop_duration_s
            else:
                # Simulation orange ball - apply static check
                # Flower pots and other orange objects are static
                if movement < 5.0:
                    self.static_count += 1
                else:
                    is_moving = True
                    self.static_count = 0
                    if self.stop_until is None:
                        self.stop_until = now + self.min_stop_duration_s

        self.prev_center = center

        if not using_real:
            if self.static_count >= self.MAX_STATIC_FRAMES:
                return None
            if not is_moving and self.stop_until is not None and now >= self.stop_until:
                self.stop_until = None
            if not is_moving and self.stop_until is None:
                return None

        self.persistence_counter = self.PERSISTENCE_FRAMES
        if self.stop_until is None:
            self.stop_until = now + self.min_stop_duration_s

        # TTC computation
        # Method 1: ball approaching - use area growth rate
        # Method 2: ball sideways - use robot speed estimate
        bbox_area = float(box.area)
        ttc = None
        if (self.prev_bbox_area is not None
                and self.prev_stamp is not None):
            dt = now - self.prev_stamp
            if dt > 1e-6:
                area_growth = (bbox_area - self.prev_bbox_area) / dt
                if area_growth > 1.0:
                    # Ball approaching camera directly
                    # TTC = current_area / area_growth_rate
                    ttc = bbox_area / area_growth
                    ttc = float(max(0.0, min(ttc, 30.0)))
                    ttc = round(ttc, 1)
                elif (context is not None
                      and context.odom_linear_x > 0.01
                      and bbox_area > 0):
                    # Ball rolling sideways - estimate from robot speed
                    # Larger bbox = closer = less time to contact
                    robot_speed = context.odom_linear_x
                    estimated_distance = 1.0 / max(0.001, (bbox_area / 10000.0) ** 0.5)
                    ttc = estimated_distance / max(0.01, robot_speed)
                    ttc = float(max(0.0, min(ttc, 30.0)))
                    ttc = round(ttc, 1)

        self.prev_bbox_area = bbox_area
        self.prev_stamp = now
        self.last_box = box
        self.last_ttc = ttc
        self.last_mask = mask_debug_image

        return BallDetection(
            box=box, moving=True, ttc=ttc,
            confidence=confidence,
            mask_debug_image=mask_debug_image)
'''

with open('/home/swathi/ENPM673_final_project/tb4_autonomy_real/tb4_autonomy_real/detectors/moving_ball_detector.py', 'w') as f:
    f.write(mbd)
print('moving_ball_detector.py written OK')

# Fix 2: real_ball_fallback.py with relaxed thresholds
rbf = '''from __future__ import annotations
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
    prefer=red_blue: try PSG soccer ball first then white baseball
    prefer=white:    try white baseball first then PSG soccer ball
    """
    red_blue = _detect_red_blue_ball(frame)
    white = _detect_white_ball(frame)
    if prefer == "red_blue" and red_blue is not None:
        return red_blue
    if prefer == "white" and white is not None:
        return white
    if red_blue is None:
        return white
    if white is None:
        return red_blue
    return red_blue if red_blue.confidence >= white.confidence else white


def _detect_red_blue_ball(frame) -> RealBallCandidate | None:
    """PSG blue/red soccer ball detector.
    Red panels:  H=0-12 and H=165-180 (HSV wrap-around)
    Blue panels: H=95-135
    Changes from original:
      y+h threshold 0.55->0.45 (catches ball entering frame earlier)
      edge rejection x<=2->x<=1 (catches ball from sides)
    """
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
        if x <= 1 or x + w >= width - 1:
            continue
        if y + h < height * 0.45:
            continue
        aspect = w / float(max(1, h))
        if aspect < 0.45 or aspect > 1.55:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0.0 else float(4.0 * np.pi * area / (perimeter**2))
        if circularity < 0.18:
            continue
        fill_ratio = area / float(max(1, w * h))
        score = float(np.clip(
            0.35 * min(1.0, area / 3500.0)
            + 0.35 * min(1.0, circularity / 0.55)
            + 0.30 * min(1.0, fill_ratio / 0.45),
            0.0, 1.0))
        if score > best_score:
            best_score = score
            best = RealBallCandidate(Box2D(int(x), int(y), int(w), int(h)), score, mask, "red_blue")
    return best


def _detect_white_ball(frame) -> RealBallCandidate | None:
    """White baseball detector.
    White: H=any, S<65, V>160 (low saturation, high brightness)
    Changes from original:
      area 70-1200 -> 50-3000 (catches closer/farther ball)
      y+h threshold 0.75->0.60 (detects ball higher in frame)
    """
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
        if area < 50.0 or area > 3000.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if y + h < height * 0.60:
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
        score = float(np.clip(
            0.70 * circularity + 0.30 * min(1.0, fill_ratio),
            0.0, 1.0))
        if score > best_score:
            best_score = score
            best = RealBallCandidate(Box2D(int(x), int(y), int(w), int(h)), score, mask, "white")
    return best
