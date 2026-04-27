from __future__ import annotations

from tb4_autonomy.data_types import HorizonDetection


class HorizonDetector:
    """Placeholder for Task 4 horizon overlay.

    Until the real estimator is implemented, draw a stable line near the image
    midpoint so the debug pipeline and RViz display can be tested end-to-end.
    """

    name = 'horizon'

    def __init__(self, horizon_ratio: float = 0.5):
        self.horizon_ratio = horizon_ratio

    def detect(self, frame, context):
        _ = context
        if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return None

        height, width = frame.shape[:2]
        y = int(height * self.horizon_ratio)
        return HorizonDetection(p1=(0, y), p2=(width - 1, y), confidence=0.1)
