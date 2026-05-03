# =============================================================================
# ENPM673 - Perception for Autonomous Robots | Spring 2026
# TurtleBot4 Final Project — Task 3: Dynamic Object Detection
# Author:  Swathi
# Detects a moving ball and computes Time To Contact (TTC).
# Uses HSV color filtering + circularity + static object rejection.
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
from tb4_autonomy.data_types import BallDetection, Box2D, FrameContext


class MovingBallDetector:
    """
    Task 3: Detects a moving ball using HSV color filtering and
    contour analysis. Computes monocular TTC from bbox growth rate.

    Tuned for Webots simulation orange ball (radius=0.032m).
    For real robot white baseball, update HSV range and area thresholds.
    """

    name = 'moving_ball'

    # Orange ball HSV range — from house.wbt baseColor 1 0.5 0
    # H=12-22 (orange hue), S>150 (saturated), V>150 (bright)
    ORANGE_HSV_LOWER = np.array([12, 150, 150], dtype=np.uint8)
    ORANGE_HSV_UPPER = np.array([22, 255, 255], dtype=np.uint8)

    # Area thresholds based on ball radius=0.032m, FOV=60deg, width=640px
    # ~400px² at 3m distance, ~55000px² at 0.5m distance
    MIN_BALL_AREA = 400
    MAX_BALL_AREA = 55000

    def __init__(self):
        self.prev_bbox_area = None
        self.prev_stamp = None
        self.prev_center = None

        # Hold detection for 3 frames after ball disappears
        self.PERSISTENCE_FRAMES = 3
        self.persistence_counter = 0
        self.last_box = None
        self.last_ttc = None

        # Keep robot stopped for minimum 1.5s for demo visibility
        self.min_stop_duration_s = 1.5
        self.stop_until = None

        # Reject static objects — ball must move, pots/walls must not
        self.static_count = 0
        self.MAX_STATIC_FRAMES = 5

    def _find_ball(self, frame):
        """
        Find best orange circular contour in frame.
        Returns (contour, score) or (None, 0.0) if not found.
        """
        height = frame.shape[0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(
            hsv,
            self.ORANGE_HSV_LOWER,
            self.ORANGE_HSV_UPPER,
        )

        # Ball confirmed at y=326+ in 576px frame — ignore ceiling/upper wall
        color_mask[:int(height * 0.30), :] = 0

        # Clean up noise and connect nearby fragments
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
            # Circularity: 1.0=perfect circle, reject blobs < 0.4
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
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

        # Get bounding box
        x, y, w, h = cv2.boundingRect(ball_contour)
        box = Box2D(x=x, y=y, w=w, h=h)
        cx, cy = box.center

        # Static object rejection — ball moves each frame, pots don't
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
            and context.odom_linear_x > 0.01  # robot must be moving
        ):
            dt = now - self.prev_stamp
            if dt > 1e-6:
                area_growth = (bbox_area - self.prev_bbox_area) / dt
                if area_growth > 1.0:  # ball must be approaching
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
