from __future__ import annotations

import cv2
import numpy as np

from tb4_autonomy_real.data_types import DetectionResults


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_label(image, text: str, origin: tuple[int, int], color=WHITE) -> None:
    x, y = origin
    scale = max(0.32, min(0.6, image.shape[1] / 700.0))
    thickness = 1 if min(image.shape[:2]) <= 320 else 2
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 1,
    )
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_box(image, box, color, label: str | None = None) -> None:
    if box is None or not box.is_valid():
        return
    p1 = (int(box.x), int(box.y))
    p2 = (int(box.x + box.w), int(box.y + box.h))
    cv2.rectangle(image, p1, p2, color, 2)
    if label:
        draw_label(image, label, (p1[0], max(20, p1[1] - 8)), color)


def draw_debug_thumbnail(
    image,
    debug_image,
    title: str,
    origin: tuple[int, int],
    size: tuple[int, int] = (180, 100),
) -> None:
    if debug_image is None:
        return

    x, y = origin
    width, height = size
    if y + height > image.shape[0] or x + width > image.shape[1]:
        return

    thumb = debug_image
    if len(thumb.shape) == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    thumb = cv2.resize(thumb, (width, height), interpolation=cv2.INTER_AREA)
    image[y:y + height, x:x + width] = thumb
    cv2.rectangle(image, (x, y), (x + width, y + height), YELLOW, 2)
    draw_label(image, title, (x, max(20, y - 8)), YELLOW)


def draw_status(image, lines: list[str]) -> None:
    y = 26
    for line in lines:
        draw_label(image, line, (12, y), WHITE)
        y += 24


def draw_detections(image, results: DetectionResults) -> None:
    low_res = min(image.shape[:2]) <= 320
    if results.horizon is not None:
        cv2.line(image, results.horizon.p1, results.horizon.p2, CYAN, 2)
        mid_x = (results.horizon.p1[0] + results.horizon.p2[0]) // 2
        mid_y = (results.horizon.p1[1] + results.horizon.p2[1]) // 2
        draw_label(
            image,
            f'HORIZON conf={results.horizon.confidence:.2f}',
            (mid_x - 80, max(20, mid_y - 10)),
            CYAN,
        )

    if results.arrow is not None:
        if results.arrow.heading_valid and results.arrow.heading_angle_deg is not None:
            if low_res:
                label = f'AR {results.arrow.heading_angle_deg:.0f}deg'
            else:
                label = f'ARROW heading={results.arrow.heading_angle_deg:.0f}deg'
        else:
            label = f'AR {results.arrow.direction.upper()}' if low_res else f'ARROW {results.arrow.direction.upper()}'
            if results.arrow.direction == 'unknown' and results.arrow.raw_direction != 'unknown':
                label = (
                    f'AR raw={results.arrow.raw_direction.upper()}'
                    if low_res else f'ARROW raw={results.arrow.raw_direction.upper()}'
                )
        if results.arrow.corners is not None and len(results.arrow.corners) == 4:
            points = np.array(results.arrow.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=GREEN, thickness=2)
            x = int(min(point[0] for point in results.arrow.corners))
            y = int(min(point[1] for point in results.arrow.corners))
            draw_label(image, label, (x, max(20, y - 8)), GREEN)
        else:
            draw_box(image, results.arrow.box, GREEN, label)
        if results.arrow.heading_base is not None and results.arrow.heading_tip is not None:
            cv2.arrowedLine(
                image,
                tuple(int(value) for value in results.arrow.heading_base),
                tuple(int(value) for value in results.arrow.heading_tip),
                CYAN,
                2 if low_res else 3,
                tipLength=0.35,
            )
            if not low_res:
                draw_label(
                    image,
                    f'heading: {results.arrow.heading_source}',
                    tuple(int(value) for value in results.arrow.heading_tip),
                    CYAN,
                )
        if results.arrow.heading_valid:
            center = results.arrow.box.center
            cv2.circle(image, (int(round(center[0])), int(round(center[1]))), 3 if low_res else 5, YELLOW, -1)
            if results.arrow.heading_angle_deg is not None and not low_res:
                draw_label(
                    image,
                    f'heading={results.arrow.heading_angle_deg:.1f}deg exist={results.arrow.arrow_presence_confidence:.2f}',
                    (int(round(center[0])), int(round(center[1] + 24))),
                    CYAN,
                )
        if results.arrow.paper_axis_angle_rad is not None:
            center = results.arrow.box.center
            angle = results.arrow.paper_heading_angle_rad or results.arrow.paper_axis_angle_rad
            length = max(24.0, min(results.arrow.box.w, results.arrow.box.h) * 0.45)
            dx = np.cos(angle)
            dy = -np.sin(angle)
            base = (int(round(center[0] - dx * length * 0.5)), int(round(center[1] - dy * length * 0.5)))
            tip = (int(round(center[0] + dx * length * 0.5)), int(round(center[1] + dy * length * 0.5)))
            cv2.arrowedLine(image, base, tip, YELLOW, 2, tipLength=0.35)
            draw_label(image, 'paper debug-only', tip, YELLOW)

    if results.logo is not None:
        draw_box(image, results.logo.box, RED, 'UMD LOGO')

    if results.moving_ball is not None:
        label = 'MOVING'
        if results.moving_ball.ttc is not None:
            label = f'MOVING TTC={results.moving_ball.ttc:.1f}s'
        draw_box(image, results.moving_ball.box, YELLOW, label)
        draw_debug_thumbnail(
            image,
            results.moving_ball.mask_debug_image,
            'BALL HSV MASK',
            (image.shape[1] - 192, image.shape[0] - 112),
        )

    if results.static_ball is not None:
        draw_box(
            image,
            results.static_ball.box,
            RED,
            f'STATIC FLOW BALL conf={results.static_ball.confidence:.2f}',
        )
        draw_debug_thumbnail(
            image,
            results.static_ball.mask_debug_image,
            'STATIC FLOW MASK',
            (image.shape[1] - 192, image.shape[0] - 224),
        )
