from __future__ import annotations

import cv2
import numpy as np

from tb4_autonomy.data_types import DetectionResults


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)


def draw_label(image, text: str, origin: tuple[int, int], color=WHITE) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_box(image, box, color, label: str | None = None) -> None:
    if box is None or not box.is_valid():
        return
    p1 = (int(box.x), int(box.y))
    p2 = (int(box.x + box.w), int(box.y + box.h))
    cv2.rectangle(image, p1, p2, color, 2)
    if label:
        draw_label(image, label, (p1[0], max(20, p1[1] - 8)), color)


def draw_status(image, lines: list[str]) -> None:
    y = 26
    for line in lines:
        draw_label(image, line, (12, y), WHITE)
        y += 24


def draw_detections(image, results: DetectionResults) -> None:
   
    if results.horizon is not None:
        h = results.horizon
        color = BLUE if h.confidence >= 0.5 else CYAN 
        cv2.line(image, h.p1, h.p2, color, 4)
        label = f'HORIZON y={h.p1[1]}px conf={h.confidence:.2f}'
        draw_label(image, label, (10, h.p1[1] - 10 if h.p1[1] > 20 else h.p1[1] + 20), color)

    if results.arrow is not None:
        label = f'ARROW {results.arrow.direction.upper()}'
        if results.arrow.direction == 'unknown' and results.arrow.raw_direction != 'unknown':
            label = f'ARROW raw={results.arrow.raw_direction.upper()}'
        if results.arrow.corners is not None and len(results.arrow.corners) == 4:
            points = np.array(results.arrow.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=GREEN, thickness=2)
            x = int(min(point[0] for point in results.arrow.corners))
            y = int(min(point[1] for point in results.arrow.corners))
            draw_label(image, label, (x, max(20, y - 8)), GREEN)
        else:
            draw_box(image, results.arrow.box, GREEN, label)

    if results.logo is not None:
        draw_box(image, results.logo.box, RED, 'UMD LOGO')

    if results.moving_ball is not None:
        label = 'MOVING'
        if results.moving_ball.ttc is not None:
            label = f'MOVING TTC={results.moving_ball.ttc:.1f}s'
        draw_box(image, results.moving_ball.box, YELLOW, label)
