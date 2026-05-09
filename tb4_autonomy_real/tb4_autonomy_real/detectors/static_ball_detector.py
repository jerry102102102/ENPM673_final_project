from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tb4_autonomy_real.data_types import Box2D, StaticBallDetection


@dataclass(frozen=True)
class StaticBallDetectorConfig:
    min_area_px: float = 1500.0
    min_circularity: float = 0.60
    min_aspect_ratio: float = 0.70
    max_aspect_ratio: float = 1.20
    flow_threshold_std_scale: float = 1.50
    morph_kernel_size: int = 7


class StaticObstacleDetector:
    """Detects a round static ball candidate from frame-to-frame optical flow."""

    name = 'static_ball'

    def __init__(self, config: StaticBallDetectorConfig | None = None):
        self.config = config or StaticBallDetectorConfig()
        self.prev_gray = None

    def reset(self) -> None:
        self.prev_gray = None

    def detect(self, frame, context=None):
        if frame is None or frame.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        self.prev_gray = gray

        fx = flow[:, :, 0]
        fy = flow[:, :, 1]
        mag = np.sqrt(fx**2 + fy**2)
        mag_blur = cv2.GaussianBlur(mag, (15, 15), 0)

        threshold = float(np.mean(mag_blur) + self.config.flow_threshold_std_scale * np.std(mag_blur))
        mask = (mag_blur > threshold).astype(np.uint8) * 255

        kernel_size = max(1, int(self.config.morph_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        debug_image = self._flow_debug_image(frame, mag_blur, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_detection = None
        best_score = 0.0
        for contour in contours:
            detection, score = self._detection_from_contour(contour, debug_image)
            if detection is not None and score > best_score:
                best_detection = detection
                best_score = score
        return best_detection

    def _detection_from_contour(self, contour, debug_image):
        area = float(cv2.contourArea(contour))
        if area < self.config.min_area_px:
            return None, 0.0

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return None, 0.0

        circularity = float(4.0 * np.pi * area / (perimeter**2))
        if circularity < self.config.min_circularity:
            return None, 0.0

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            return None, 0.0
        aspect = float(w / h)
        if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
            return None, 0.0

        aspect_score = 1.0 - min(1.0, abs(aspect - 1.0))
        confidence = float(np.clip(0.65 * circularity + 0.35 * aspect_score, 0.0, 1.0))
        return StaticBallDetection(Box2D(int(x), int(y), int(w), int(h)), confidence, debug_image), confidence

    @staticmethod
    def _flow_debug_image(frame, mag_blur, mask):
        mag_norm = cv2.normalize(mag_blur, None, 0, 255, cv2.NORM_MINMAX)
        flow_color = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_JET)
        flow_view = np.zeros_like(frame)
        flow_view[mask > 0] = flow_color[mask > 0]
        return flow_view


StaticBallDetector = StaticObstacleDetector
