# =============================================================================
# ENPM673 - Perception for Autonomous Robots | Spring 2026
# TurtleBot4 Final Project — Task 3: Dynamic Object Detection
# Author:  Swathi
# Detects a moving ball and computes Time To Contact (TTC).
# Uses HSV color filtering + circularity + static object rejection.
# DEMO DAY SWITCH:
#   Change MODE = 'simulation' to MODE = 'real' at the top of the class
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
from tb4_autonomy.data_types import BallDetection, Box2D, FrameContext


class MovingBallDetector:
    """
    Task 3: Detects a moving ball using HSV color filtering and
    contour analysis. Computes monocular TTC from bbox growth rate.

    Two modes:
      'simulation' — Webots orange ball (radius=0.032m)
      'real'       — RAL Lab white baseball (diameter=2.9in)

    Change MODE below to switch between modes.
    """

    name = 'moving_ball'

    # =========================================================================
    # DEMO DAY SWITCH — change this one line on May 10
    # 'simulation' = Webots orange ball
    # 'real'       = RAL Lab white baseball
    # =========================================================================
    MODE = 'simulation'

    # ── Simulation Mode — Webots orange ball ──────────────────────────────────
    # Ball from house.wbt: baseColor 1 0.5 0 (orange), radius=0.032m
    # H=12-22 (orange), S>150 (saturated), V>150 (bright)
    # Area range: ~400px² at 3m, ~55000px² at 0.5m
    SIM_HSV_LOWER  = np.array([12,  150, 150], dtype=np.uint8)
    SIM_HSV_UPPER  = np.array([22,  255, 255], dtype=np.uint8)
    SIM_MIN_AREA   = 400
    SIM_MAX_AREA   = 55000
    SIM_CIRCULARITY = 0.4

    # ── Real Robot Mode — white baseball with red threading ───────────────────
    # Ball confirmed by Prof. Charifa: white baseball, diameter=2.9in (0.074m)
    # White body: low saturation, high brightness
    # Red threading: wraps around HSV 0/180 boundary — needs two ranges
    # Area range: ~800px² at 3m, ~65000px² at 0.7m
    # Stricter circularity (0.5) because white walls are common in RAL Lab
    REAL_WHITE_HSV_LOWER = np.array([0,   0,   170], dtype=np.uint8)
    REAL_WHITE_HSV_UPPER = np.array([180, 50,  255], dtype=np.uint8)
    REAL_RED_HSV_LOWER_1 = np.array([0,   100, 100], dtype=np.uint8)
    REAL_RED_HSV_UPPER_1 = np.array([10,  255, 255], dtype=np.uint8)
    REAL_RED_HSV_LOWER_2 = np.array([170, 100, 100], dtype=np.uint8)
    REAL_RED_HSV_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)
    REAL_MIN_AREA        = 800
    REAL_MAX_AREA        = 65000
    REAL_CIRCULARITY     = 0.5

    def __init__(self):
        self.prev_bbox_area = None
        self.prev_stamp = None
        self.prev_center = None

        # Hold detection for 3 frames after ball disappears
        self.PERSISTENCE_FRAMES = 3
        self.persistence_counter = 0
        self.last_box = None
        self.last_ttc = None

        # Keep robot stopped minimum 1.5s for demo visibility
        self.min_stop_duration_s = 1.5
        self.stop_until = None

        # Reject static objects — ball must move, walls/pots must not
        self.static_count = 0
        self.MAX_STATIC_FRAMES = 5

    def _get_color_mask(self, hsv):
        """
        Build color mask based on current MODE.
        Simulation: orange HSV range only.
        Real:       white OR red HSV ranges combined.
        """
        if self.MODE == 'real':
            white_mask = cv2.inRange(
                hsv,
                self.REAL_WHITE_HSV_LOWER,
                self.REAL_WHITE_HSV_UPPER,
            )
            red_mask_1 = cv2.inRange(
                hsv,
                self.REAL_RED_HSV_LOWER_1,
                self.REAL_RED_HSV_UPPER_1,
            )
            red_mask_2 = cv2.inRange(
                hsv,
                self.REAL_RED_HSV_LOWER_2,
                self.REAL_RED_HSV_UPPER_2,
            )
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
            return cv2.bitwise_or(white_mask, red_mask)
        else:
            # simulation — orange only
            return cv2.inRange(
                hsv,
                self.SIM_HSV_LOWER,
                self.SIM_HSV_UPPER,
            )

    def _get_thresholds(self):
        """Return (min_area, max_area, min_circularity) for current MODE."""
        if self.MODE == 'real':
            return self.REAL_MIN_AREA, self.REAL_MAX_AREA, self.REAL_CIRCULARITY
        else:
            return self.SIM_MIN_AREA, self.SIM_MAX_AREA, self.SIM_CIRCULARITY

    def _find_ball(self, frame):
        """
        Find best ball-shaped contour in frame using color + circularity.
        Returns (contour, score) or (None, 0.0) if not found.
        """
        height = frame.shape[0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = self._get_color_mask(hsv)
        min_area, max_area, min_circularity = self._get_thresholds()

        # Exclude top 30% — ball always in lower portion of frame
        color_mask[:int(height * 0.30), :] = 0

        # Clean noise and connect nearby fragments
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
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1.0:
                continue
            # Circularity: 1.0=perfect circle
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour, best_score

    def detect(self, frame, context: FrameContext) -> BallDetection | None:
        """
        Run detection on single camera frame.
        Returns BallDetection(moving=True) to trigger BALL_STOP state.
        Returns None when no moving ball detected.
        """
        if frame is None:
            return None
        now = context.stamp_sec

        ball_contour, best_score = self._find_ball(frame)

        # No ball found — handle persistence and stop timer
        if ball_contour is None:
            self.prev_center = None
            self.static_count = 0

            if self.persistence_counter > 0:
                self.persistence_counter -= 1
                if self.last_box is not None:
                    return BallDetection(
                        box=self.last_box,
                        moving=True,
                        ttc=self.last_ttc,
                        confidence=0.3,
                    )

            if self.stop_until is not None and now < self.stop_until:
                if self.last_box is not None:
                    return BallDetection(
                        box=self.last_box,
                        moving=True,
                        ttc=self.last_ttc,
                        confidence=0.2,
                    )

            self.prev_bbox_area = None
            self.stop_until = None
            return None

        # Ball found — get position
        x, y, w, h = cv2.boundingRect(ball_contour)
        box = Box2D(x=x, y=y, w=w, h=h)
        cx, cy = box.center

        # Static object rejection
        # Ball oscillates — position changes each frame
        # Static objects (walls, furniture) stay put — rejected here
        if self.prev_center is not None:
            movement = (
                (cx - self.prev_center[0])**2 +
                (cy - self.prev_center[1])**2
            ) ** 0.5
            if movement < 5.0:
                self.static_count += 1
            else:
                self.static_count = 0
                if self.stop_until is None:
                    self.stop_until = now + self.min_stop_duration_s

        self.prev_center = (cx, cy)

        if self.static_count >= self.MAX_STATIC_FRAMES:
            return None

        if self.prev_center == (cx, cy) and self.stop_until is None:
            return None

        self.persistence_counter = self.PERSISTENCE_FRAMES

        # TTC = current_area / area_growth_rate
        # Assumes translational motion only (per project spec)
        bbox_area = float(box.area)
        ttc = None
        if (
            self.prev_bbox_area is not None
            and self.prev_stamp is not None
            and context.odom_linear_x > 0.01
        ):
            dt = now - self.prev_stamp
            if dt > 1e-6:
                area_growth = (bbox_area - self.prev_bbox_area) / dt
                if area_growth > 1.0:
                    ttc = bbox_area / area_growth
                    ttc = float(max(0.0, min(ttc, 30.0)))
                    ttc = round(ttc, 1)

        self.prev_bbox_area = bbox_area
        self.prev_stamp = now
        self.last_box = box
        self.last_ttc = ttc

        # moving=True required for BALL_STOP state to trigger
        return BallDetection(
            box=box,
            moving=True,
            ttc=ttc,
            confidence=min(1.0, best_score / 10000.0),
        )
