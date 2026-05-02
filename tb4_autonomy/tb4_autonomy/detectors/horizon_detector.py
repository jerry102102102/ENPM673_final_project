from __future__ import annotations

import cv2
import numpy as np

from tb4_autonomy.data_types import HorizonDetection


class HorizonDetector:
    """Horizon line estimator using RANSAC on Canny edge points."""

    name = 'horizon'

    MAX_SLOPE = 0.3  # reject lines steeper than this (|rise/run| threshold)

    def __init__(self, horizon_ratio: float = 0.5, history_size: int = 10):
        self.horizon_ratio = horizon_ratio
        self.history = []
        self.history_size = history_size

    def _fit_line_ransac(self, points, n_trials=200, threshold=15):
        if len(points) < 2:
            return None, 0.0

        xs = points[:, 0].astype(float)
        ys = points[:, 1].astype(float)
        best_inliers = None
        best_count = 0

        for _ in range(n_trials):
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            if abs(p2[0] - p1[0]) < 1e-6:
                continue

            m = (p2[1] - p1[1]) / (p2[0] - p1[0])

            # Reject candidate lines that are too steep (floor tiles, diagonals)
            if abs(m) > self.MAX_SLOPE:
                continue

            c = p1[1] - m * p1[0]
            dists = np.abs(m * xs - ys + c) / np.sqrt(m ** 2 + 1)
            inlier_mask = dists < threshold
            count = inlier_mask.sum()

            if count > best_count:
                best_count = count
                best_inliers = inlier_mask

        if best_count < 2 or best_inliers is None:
            return None, 0.0

        inlier_pts = points[best_inliers]
        ix = inlier_pts[:, 0].astype(float)
        iy = inlier_pts[:, 1].astype(float)

        A = np.vstack([ix, np.ones(len(ix))]).T
        m, c = np.linalg.lstsq(A, iy, rcond=None)[0]

        # Final slope check after lstsq refinement
        if abs(m) > self.MAX_SLOPE:
            return None, 0.0

        inlier_ratio = best_count / len(points)
        return (m, c), inlier_ratio

    def detect(self, frame, context) -> HorizonDetection | None:
        _ = context
        if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return None

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Narrow ROI: focus on the wall/floor boundary band
        roi_top = int(height * 0.42)
        roi_bottom = int(height * 0.60)
        roi = gray[roi_top:roi_bottom, :]

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Only keep pixels where vertical gradient dominates (horizontal edges)
        angle = np.abs(np.arctan2(np.abs(sobelx), np.abs(sobely) + 1e-6))
        horizontal_mask = angle < np.radians(40)
        edges_filtered = edges.copy()
        edges_filtered[~horizontal_mask] = 0

        ys, xs = np.where(edges_filtered > 0)

        if len(xs) < 50:
            y = int(height * self.horizon_ratio)
            return HorizonDetection(p1=(0, y), p2=(width - 1, y), confidence=0.1)

        points = np.column_stack([xs, ys])
        if len(points) > 500:
            idx = np.random.choice(len(points), 500, replace=False)
            points = points[idx]

        result, inlier_ratio = self._fit_line_ransac(points)

        if result is None:
            y = int(height * self.horizon_ratio)
            return HorizonDetection(p1=(0, y), p2=(width - 1, y), confidence=0.1)

        m, c = result
        x1, x2 = 0, width - 1
        y1 = int(np.clip(int(m * x1 + c) + roi_top, 0, height - 1))
        y2 = int(np.clip(int(m * x2 + c) + roi_top, 0, height - 1))

        self.history.append((x1, y1, x2, y2))
        if len(self.history) > self.history_size:
            self.history.pop(0)

        x1, y1, x2, y2 = np.mean(self.history, axis=0).astype(int)

        return HorizonDetection(
            p1=(int(x1), int(y1)),
            p2=(int(x2), int(y2)),
            confidence=float(inlier_ratio),
        )
