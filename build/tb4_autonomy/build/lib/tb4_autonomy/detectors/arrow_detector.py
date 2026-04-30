from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from tb4_autonomy.data_types import ArrowDetection, Box2D


@dataclass
class ArrowDetectorConfig:
    threshold_method: str = 'otsu'
    hsv_v_max: int = 80
    blur_kernel: int = 5
    morph_open_kernel: int = 3
    morph_close_kernel: int = 5
    dilate_kernel: int = 7
    dilate_iterations: int = 1
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.40
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    min_black_pixel_ratio: float = 0.03
    warp_width: int = 300
    warp_height: int = 300
    inner_crop_margin_ratio: float = 0.12
    history_size: int = 5
    min_stable_count: int = 4
    process_width: int = 960
    bbox_padding_ratio: float = 0.20
    min_arrow_area_ratio: float = 0.01
    paper_v_min: int = 110
    paper_s_max: int = 90
    paper_min_area_ratio: float = 0.001
    paper_max_area_ratio: float = 0.25
    paper_min_aspect_ratio: float = 0.3
    paper_max_aspect_ratio: float = 12.0


@dataclass
class _Candidate:
    box: Box2D
    contour: np.ndarray
    area_ratio: float
    black_pixel_ratio: float
    score: float


@dataclass
class _PaperFragment:
    box: Box2D
    black_pixel_ratio: float


@dataclass
class _InnerFrame:
    box: Box2D
    corners: np.ndarray


class ArrowDetector:
    """Traditional OpenCV detector for printed arrow signs.

    The detector searches for a high-contrast black sign/arrow region, rectifies
    the selected planar region, segments the arrow in the warped image, and
    classifies direction from arrow-contour geometry.
    """

    name = 'arrow'

    def __init__(self, config: ArrowDetectorConfig | None = None):
        self.config = config or ArrowDetectorConfig()
        self.direction_history: deque[str] = deque(maxlen=max(1, self.config.history_size))

    def detect(self, frame, context):
        if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return None

        original_height, original_width = frame.shape[:2]
        if original_width <= 0 or original_height <= 0:
            return None

        scaled_frame, scale = self._resize_for_processing(frame)
        mask = self._preprocess(scaled_frame)
        candidate = self._select_candidate(mask)
        if candidate is None:
            candidate = self._select_paper_candidate(scaled_frame, mask)
        if candidate is None:
            self.direction_history.append('unknown')
            return None

        corners_small = self._estimate_corners(candidate, mask.shape[:2])
        corners = self._scale_corners(corners_small, scale)
        warped = self._warp(frame, corners)
        raw_direction, arrow_mask, arrow_confidence, arrow_angle_rad = self._classify_warped_arrow(warped)

        self.direction_history.append(raw_direction)
        stable_direction, stable_confidence, is_stable = self._stable_direction()
        direction = stable_direction if is_stable else 'unknown'

        box = self._scale_box(candidate.box, scale, original_width, original_height)
        center_error = box.center[0] - original_width / 2.0
        confidence = min(1.0, 0.55 * candidate.score + 0.45 * max(arrow_confidence, stable_confidence))

        return ArrowDetection(
            box=box,
            direction=direction,
            confidence=confidence,
            raw_direction=raw_direction,
            corners=corners,
            area_ratio=box.area / float(original_width * original_height),
            black_pixel_ratio=candidate.black_pixel_ratio,
            center_error_px=center_error,
            is_stable=is_stable,
            arrow_angle_rad=arrow_angle_rad,
            warped_debug_image=warped,
            mask_debug_image=arrow_mask,
        )

    def _resize_for_processing(self, frame):
        height, width = frame.shape[:2]
        if self.config.process_width <= 0 or width <= self.config.process_width:
            return frame, 1.0

        scale = self.config.process_width / float(width)
        new_size = (self.config.process_width, int(height * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale

    def _preprocess(self, frame):
        blur_kernel = self._odd_kernel(self.config.blur_kernel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if self.config.threshold_method.lower() == 'hsv':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv[:, :, 2], 0, int(self.config.hsv_v_max))
        else:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        open_kernel = self._kernel(self.config.morph_open_kernel)
        close_kernel = self._kernel(self.config.morph_close_kernel)
        dilate_kernel = self._kernel(self.config.dilate_kernel)
        if open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        if close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        if dilate_kernel is not None and self.config.dilate_iterations > 0:
            mask = cv2.dilate(mask, dilate_kernel, iterations=self.config.dilate_iterations)
        return mask

    def _select_candidate(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        image_area = float(width * height)
        if image_area <= 0:
            return None

        candidates: list[_Candidate] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            box_area = w * h
            area_ratio = box_area / image_area
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                continue

            aspect = w / float(h)
            if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
                continue

            roi_mask = mask[y:y + h, x:x + w]
            black_ratio = cv2.countNonZero(roi_mask) / float(box_area)
            if black_ratio < self.config.min_black_pixel_ratio:
                continue

            center_x = x + w / 2.0
            center_y = y + h / 2.0
            max_dist = math.hypot(width / 2.0, height / 2.0)
            center_dist = math.hypot(center_x - width / 2.0, center_y - height / 2.0)
            center_score = 1.0 - min(1.0, center_dist / max_dist)
            aspect_score = 1.0 - min(1.0, abs(math.log(max(aspect, 1e-6))))
            density_score = min(1.0, black_ratio / 0.25)
            area_score = min(1.0, area_ratio / 0.08)
            score = 0.35 * area_score + 0.25 * center_score + 0.20 * aspect_score + 0.20 * density_score

            candidates.append(
                _Candidate(
                    box=Box2D(x=x, y=y, w=w, h=h),
                    contour=contour,
                    area_ratio=area_ratio,
                    black_pixel_ratio=black_ratio,
                    score=score,
                )
            )

        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.score)

    def _select_paper_candidate(self, frame, black_mask):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        paper_mask = cv2.inRange(
            hsv,
            (0, 0, int(self.config.paper_v_min)),
            (180, int(self.config.paper_s_max), 255),
        )
        # Ignore the upper wall region; arrow papers are floor-level signs.
        paper_mask[:int(paper_mask.shape[0] * 0.35), :] = 0
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, self._kernel(5))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, self._kernel(15))

        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = paper_mask.shape[:2]
        image_area = float(width * height)
        fragments: list[_PaperFragment] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            box_area = w * h
            area_ratio = box_area / image_area
            if area_ratio < self.config.paper_min_area_ratio or area_ratio > self.config.paper_max_area_ratio:
                continue

            aspect = w / float(h)
            if aspect < self.config.paper_min_aspect_ratio or aspect > self.config.paper_max_aspect_ratio:
                continue

            black_roi = black_mask[y:y + h, x:x + w]
            black_ratio = cv2.countNonZero(black_roi) / float(box_area)
            if black_ratio < self.config.min_black_pixel_ratio:
                continue

            fragments.append(_PaperFragment(box=Box2D(x=x, y=y, w=w, h=h), black_pixel_ratio=black_ratio))

        if not fragments:
            return None

        candidates = self._paper_candidates_from_fragments(fragments, black_mask, width, height, image_area)
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.score)

    def _paper_candidates_from_fragments(
        self,
        fragments: list[_PaperFragment],
        black_mask,
        width: int,
        height: int,
        image_area: float,
    ) -> list[_Candidate]:
        # The distant Webots paper sign often appears as two disconnected white
        # strips split by the printed black arrow. Merge strips in the same
        # horizontal band before rectifying the sign.
        boxes = [fragment.box for fragment in fragments]
        merged_boxes = boxes + self._merge_paper_boxes(boxes, width)
        unique: dict[tuple[int, int, int, int], Box2D] = {}
        for box in merged_boxes:
            key = (box.x, box.y, box.w, box.h)
            unique[key] = box

        candidates: list[_Candidate] = []
        for box in unique.values():
            x, y, w, h = box.x, box.y, box.w, box.h
            box_area = w * h
            if box_area <= 0:
                continue

            area_ratio = box_area / image_area
            if area_ratio < self.config.paper_min_area_ratio or area_ratio > self.config.paper_max_area_ratio:
                continue

            aspect = w / float(h)
            if aspect < self.config.paper_min_aspect_ratio or aspect > self.config.paper_max_aspect_ratio:
                continue

            black_roi = black_mask[y:y + h, x:x + w]
            black_ratio = cv2.countNonZero(black_roi) / float(box_area)
            if black_ratio < self.config.min_black_pixel_ratio:
                continue

            center_x = x + w / 2.0
            center_y = y + h / 2.0
            center_dist = math.hypot(center_x - width / 2.0, center_y - height / 2.0)
            max_dist = math.hypot(width / 2.0, height / 2.0)
            center_score = 1.0 - min(1.0, center_dist / max_dist)
            area_score = min(1.0, area_ratio / 0.04)
            density_score = min(1.0, black_ratio / 0.25)
            covered_fragments = sum(1 for fragment in fragments if self._contains_box(box, fragment.box))
            merge_bonus = 0.10 if covered_fragments > 1 else 0.0
            score = 0.40 * area_score + 0.25 * density_score + 0.20 * center_score + 0.15 + merge_bonus

            candidates.append(
                _Candidate(
                    box=box,
                    contour=self._rect_contour(box),
                    area_ratio=area_ratio,
                    black_pixel_ratio=black_ratio,
                    score=score,
                )
            )
        return candidates

    def _merge_paper_boxes(self, boxes: list[Box2D], image_width: int) -> list[Box2D]:
        merged: list[Box2D] = []
        for index, seed in enumerate(boxes):
            group = [seed]
            current = seed
            changed = True
            while changed:
                changed = False
                for other_index, other in enumerate(boxes):
                    if other_index == index or other in group:
                        continue
                    if self._paper_boxes_related(current, other, image_width):
                        group.append(other)
                        current = self._union_boxes(group)
                        changed = True
            if len(group) > 1:
                merged.append(self._union_boxes(group))
        return merged

    def _paper_boxes_related(self, first: Box2D, second: Box2D, image_width: int) -> bool:
        first_y2 = first.y + first.h
        second_y2 = second.y + second.h
        overlap_y = max(0, min(first_y2, second_y2) - max(first.y, second.y))
        min_height = max(1, min(first.h, second.h))
        vertical_overlap = overlap_y / float(min_height)
        center_y_diff = abs(first.center[1] - second.center[1])
        same_band = vertical_overlap >= 0.25 or center_y_diff <= max(first.h, second.h) * 1.25

        first_x2 = first.x + first.w
        second_x2 = second.x + second.w
        gap_x = max(0, max(first.x, second.x) - min(first_x2, second_x2))
        max_gap = max(20, int(image_width * 0.12), max(first.h, second.h) * 4)
        return same_band and gap_x <= max_gap

    def _union_boxes(self, boxes: list[Box2D]) -> Box2D:
        x1 = min(box.x for box in boxes)
        y1 = min(box.y for box in boxes)
        x2 = max(box.x + box.w for box in boxes)
        y2 = max(box.y + box.h for box in boxes)
        return Box2D(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def _contains_box(self, outer: Box2D, inner: Box2D) -> bool:
        return (
            outer.x <= inner.x
            and outer.y <= inner.y
            and outer.x + outer.w >= inner.x + inner.w
            and outer.y + outer.h >= inner.y + inner.h
        )

    def _rect_contour(self, box: Box2D):
        return np.array(
            [
                [[box.x, box.y]],
                [[box.x + box.w, box.y]],
                [[box.x + box.w, box.y + box.h]],
                [[box.x, box.y + box.h]],
            ],
            dtype=np.int32,
        )

    def _estimate_corners(self, candidate: _Candidate, mask_shape: tuple[int, int]):
        contour = candidate.contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return self._order_points(approx.reshape(4, 2).astype(np.float32))

        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            if cv2.contourArea(box.astype(np.float32)) > 1.0:
                return self._order_points(box.astype(np.float32))

        # Fallback: expand the candidate bbox. This is often better than a
        # tight rotated arrow box when the printed border is not connected.
        height, width = mask_shape
        pad = int(max(candidate.box.w, candidate.box.h) * self.config.bbox_padding_ratio)
        x1 = max(0, candidate.box.x - pad)
        y1 = max(0, candidate.box.y - pad)
        x2 = min(width - 1, candidate.box.x + candidate.box.w + pad)
        y2 = min(height - 1, candidate.box.y + candidate.box.h + pad)
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )

    def _warp(self, frame, corners):
        dst = np.array(
            [
                [0, 0],
                [self.config.warp_width - 1, 0],
                [self.config.warp_width - 1, self.config.warp_height - 1],
                [0, self.config.warp_height - 1],
            ],
            dtype=np.float32,
        )
        src = np.array(corners, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, transform, (self.config.warp_width, self.config.warp_height))

    def _classify_warped_arrow(self, warped):
        if warped is None or warped.size == 0:
            return 'unknown', None, 0.0, 0.0

        margin = int(min(warped.shape[:2]) * self.config.inner_crop_margin_ratio)
        inner = warped[margin:warped.shape[0] - margin, margin:warped.shape[1] - margin]
        if inner.size == 0:
            inner = warped

        mask = self._preprocess_inner(inner)
        frame = self._find_inner_frame(mask)
        if frame is not None:
            arrow_image = self._warp_inner_frame(inner, frame.corners)
            arrow_mask = self._preprocess_inner(arrow_image)
            arrow_mask = self._crop_inner_arrow_mask(arrow_mask)
        else:
            arrow_mask = self._crop_inner_arrow_mask(mask)
        contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 'unknown', mask, 0.0, 0.0

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        min_area = self.config.min_arrow_area_ratio * float(arrow_mask.shape[0] * arrow_mask.shape[1])
        if contour_area < min_area:
            return 'unknown', mask, 0.0, 0.0

        direction, dominance = self._direction_from_mask(contour, arrow_mask)
        area_confidence = min(1.0, contour_area / (min_area * 4.0))
        confidence = min(1.0, 0.65 * dominance + 0.35 * area_confidence)
        angle_rad = self._angle_from_tip(contour, arrow_mask, direction)
        return direction, arrow_mask, confidence, angle_rad

    def _crop_inner_arrow_mask(self, mask):
        pad_x = max(2, int(mask.shape[1] * 0.08))
        pad_y = max(2, int(mask.shape[0] * 0.08))
        x1 = min(mask.shape[1] - 1, pad_x)
        y1 = min(mask.shape[0] - 1, pad_y)
        x2 = max(x1 + 1, mask.shape[1] - pad_x)
        y2 = max(y1 + 1, mask.shape[0] - pad_y)
        cropped = mask[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else mask

    def _find_inner_frame(self, mask) -> _InnerFrame | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        image_area = float(height * width)
        if image_area <= 0:
            return None

        candidates: list[tuple[float, _InnerFrame]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            area_ratio = (w * h) / image_area
            if area_ratio < 0.015 or area_ratio > 0.60:
                continue

            aspect = w / float(h)
            if aspect < 0.55 or aspect > 2.20:
                continue

            contour_area = max(1.0, cv2.contourArea(contour))
            rectangularity = contour_area / float(w * h)
            # The printed border may connect to the arrow after thresholding,
            # so accept both sparse border contours and denser sign contours.
            if rectangularity < 0.03 or rectangularity > 0.95:
                continue

            bottom_score = (y + h) / float(height)
            center_x = x + w / 2.0
            center_score = 1.0 - min(1.0, abs(center_x - width / 2.0) / (width / 2.0))
            area_score = min(1.0, area_ratio / 0.16)
            score = 0.50 * bottom_score + 0.30 * area_score + 0.20 * center_score
            candidates.append((score, _InnerFrame(
                box=Box2D(x=x, y=y, w=w, h=h),
                corners=self._inner_frame_corners(contour),
            )))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _find_inner_frame_box(self, mask) -> Box2D | None:
        frame = self._find_inner_frame(mask)
        return None if frame is None else frame.box

    def _inner_frame_corners(self, contour) -> np.ndarray:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return self._order_points(approx.reshape(4, 2).astype(np.float32))

        rect = cv2.minAreaRect(contour)
        return self._order_points(cv2.boxPoints(rect).astype(np.float32))

    def _warp_inner_frame(self, image, corners):
        ordered = self._order_points(np.array(corners, dtype=np.float32))
        top_width = np.linalg.norm(ordered[1] - ordered[0])
        bottom_width = np.linalg.norm(ordered[2] - ordered[3])
        left_height = np.linalg.norm(ordered[3] - ordered[0])
        right_height = np.linalg.norm(ordered[2] - ordered[1])
        width = max(24, int(round(max(top_width, bottom_width))))
        height = max(24, int(round(max(left_height, right_height))))

        # Keep the sign's observed aspect ratio while normalizing scale.
        scale = 180.0 / float(max(width, height))
        dst_width = max(40, int(round(width * scale)))
        dst_height = max(40, int(round(height * scale)))
        dst = np.array(
            [
                [0, 0],
                [dst_width - 1, 0],
                [dst_width - 1, dst_height - 1],
                [0, dst_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, transform, (dst_width, dst_height))

    def _preprocess_inner(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._odd_kernel(self.config.blur_kernel),) * 2, 0)
        _, mask = cv2.threshold(gray, int(self.config.hsv_v_max), 255, cv2.THRESH_BINARY_INV)
        close_kernel = self._kernel(self.config.morph_close_kernel)
        if close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def _direction_from_mask(self, contour, mask) -> tuple[str, float]:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return 'unknown', 0.0

        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            return 'unknown', 0.0

        template_direction, template_dominance = self._direction_from_templates(roi)
        if template_direction != 'unknown' and template_dominance >= 0.34:
            return template_direction, max(0.55, min(1.0, template_dominance))

        if w >= h * 1.55:
            band = max(3, int(w * 0.33))
            left_score = self._horizontal_end_score(roi[:, :band])
            right_score = self._horizontal_end_score(roi[:, w - band:])
            direction = 'left' if left_score >= right_score else 'right'
            dominance = min(1.0, abs(left_score - right_score) / max(left_score, right_score, 1e-6))
            return direction, max(0.55, dominance)

        if h >= w * 1.55:
            band = max(3, int(h * 0.33))
            top_score = self._vertical_end_score(roi[:band, :])
            bottom_score = self._vertical_end_score(roi[h - band:, :])
            direction = 'straight' if top_score >= bottom_score else 'back'
            dominance = min(1.0, abs(top_score - bottom_score) / max(top_score, bottom_score, 1e-6))
            return direction, max(0.55, dominance)

        band_x = max(3, int(w * 0.33))
        band_y = max(3, int(h * 0.33))
        directional_scores = {
            'left': self._horizontal_end_score(roi[:, :band_x]),
            'right': self._horizontal_end_score(roi[:, w - band_x:]),
            'straight': self._vertical_end_score(roi[:band_y, :]),
            'back': self._vertical_end_score(roi[h - band_y:, :]),
        }
        ranked_scores = sorted(directional_scores.items(), key=lambda item: item[1], reverse=True)
        if ranked_scores[0][1] > 0.0:
            top_score = ranked_scores[0][1]
            runner_up = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
            dominance = min(1.0, (top_score - runner_up) / max(top_score, 1e-6))
            if dominance >= 0.10:
                return ranked_scores[0][0], max(0.55, dominance)

        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-6:
            return 'unknown', 0.0

        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        points = contour.reshape(-1, 2).astype(np.float32)
        deltas = points - np.array([[cx, cy]], dtype=np.float32)
        distances = np.linalg.norm(deltas, axis=1)
        tip = points[int(np.argmax(distances))]
        dx = float(tip[0] - cx)
        dy = float(tip[1] - cy)
        magnitude = math.hypot(dx, dy)
        if magnitude < 1e-6:
            return 'unknown', 0.0
        direction = self._direction_from_vector(dx, dy)
        dominance = max(abs(dx), abs(dy)) / magnitude
        return direction, dominance

    def _direction_from_templates(self, roi) -> tuple[str, float]:
        binary = (roi > 0).astype(np.uint8)
        if binary.size == 0 or cv2.countNonZero(binary) == 0:
            return 'unknown', 0.0

        scores = {}
        for direction in ('left', 'right', 'straight', 'back'):
            template = self._arrow_template(direction, binary.shape[1], binary.shape[0])
            intersection = np.logical_and(binary, template).sum()
            union = np.logical_or(binary, template).sum()
            scores[direction] = 0.0 if union == 0 else intersection / float(union)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_direction, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < 0.34 or (best_score - runner_up) < 0.001:
            return 'unknown', 0.0
        return best_direction, best_score

    def _arrow_template(self, direction: str, width: int, height: int):
        width = max(1, int(width))
        height = max(1, int(height))
        right = np.array(
            [
                [0.10, 0.40],
                [0.55, 0.40],
                [0.55, 0.20],
                [0.90, 0.50],
                [0.55, 0.80],
                [0.55, 0.60],
                [0.10, 0.60],
            ],
            dtype=np.float32,
        )
        center = np.array([0.50, 0.50], dtype=np.float32)
        points = right.copy()
        if direction == 'left':
            points[:, 0] = 1.0 - points[:, 0]
        elif direction == 'straight':
            shifted = points - center
            points = np.column_stack([shifted[:, 1], -shifted[:, 0]]) + center
        elif direction == 'back':
            shifted = points - center
            points = np.column_stack([-shifted[:, 1], shifted[:, 0]]) + center

        points[:, 0] *= width - 1
        points[:, 1] *= height - 1
        template = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(template, [points.astype(np.int32)], 1)
        return template

    def _horizontal_end_score(self, band) -> float:
        if band.size == 0:
            return 0.0
        height, width = band.shape[:2]
        density = cv2.countNonZero(band) / float(max(1, height * width))
        column_span = float(np.max(np.count_nonzero(band, axis=0))) / float(max(1, height))
        return 0.65 * density + 0.35 * column_span

    def _vertical_end_score(self, band) -> float:
        if band.size == 0:
            return 0.0
        height, width = band.shape[:2]
        density = cv2.countNonZero(band) / float(max(1, height * width))
        row_span = float(np.max(np.count_nonzero(band, axis=1))) / float(max(1, width))
        return 0.65 * density + 0.35 * row_span

    def _direction_from_vector(self, dx: float, dy: float) -> str:
        angle = math.degrees(math.atan2(-dy, dx))
        if -45.0 <= angle <= 45.0:
            return 'right'
        if 45.0 < angle <= 135.0:
            return 'straight'
        if angle > 135.0 or angle < -135.0:
            return 'left'
        if -135.0 <= angle < -45.0:
            return 'back'
        return 'unknown'

    def _angle_from_tip(self, contour, mask, direction: str) -> float:
        """Compute a continuous arrow angle using direction-guided tip finding.

        Strategy:
          1. Use the already-classified *direction* to place a reference
             point at the arrow's **base** (the blunt tail-end of the shaft).
          2. Find the contour point farthest from that base → this is the
             arrow **tip** (the pointy end).
          3. Return atan2(base → tip) as the continuous pointing angle.

        This is robust because no matter how noisy the contour is, the
        farthest point from the correct base side is always the tip.

        Convention (image coords, y-down, angles via atan2(-dy, dx)):
          right    ≈  0 rad
          straight ≈ +π/2  (up in image = forward for robot)
          left     ≈ ±π
          back     ≈ -π/2
        """
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return math.pi / 2.0

        # Place reference at the base side (opposite of arrow tip)
        _BASE_OFFSETS = {
            'straight': (x + w / 2.0, y + h),       # bottom-center
            'back':     (x + w / 2.0, float(y)),     # top-center
            'left':     (x + float(w), y + h / 2.0), # right-center
            'right':    (float(x),     y + h / 2.0), # left-center
        }
        base = _BASE_OFFSETS.get(direction)
        if base is None:
            # unknown direction: fall back to centroid
            moments = cv2.moments(contour)
            if abs(moments['m00']) < 1e-6:
                return math.pi / 2.0
            base = (moments['m10'] / moments['m00'],
                    moments['m01'] / moments['m00'])

        base_pt = np.array([[base[0], base[1]]], dtype=np.float32)
        points = contour.reshape(-1, 2).astype(np.float32)
        distances = np.linalg.norm(points - base_pt, axis=1)
        tip = points[int(np.argmax(distances))]

        dx = float(tip[0] - base[0])
        dy = float(tip[1] - base[1])
        if math.hypot(dx, dy) < 1e-6:
            return math.pi / 2.0

        return math.atan2(-dy, dx)

    def _stable_direction(self):
        valid = [direction for direction in self.direction_history if direction != 'unknown']
        if not valid:
            return 'unknown', 0.0, False

        direction, count = Counter(valid).most_common(1)[0]
        confidence = count / float(max(1, len(self.direction_history)))
        return direction, confidence, count >= self.config.min_stable_count

    def _scale_box(self, box: Box2D, scale: float, image_width: int, image_height: int) -> Box2D:
        if scale == 1.0:
            return box
        inv = 1.0 / scale
        x = int(round(box.x * inv))
        y = int(round(box.y * inv))
        w = int(round(box.w * inv))
        h = int(round(box.h * inv))
        x = max(0, min(image_width - 1, x))
        y = max(0, min(image_height - 1, y))
        w = max(1, min(image_width - x, w))
        h = max(1, min(image_height - y, h))
        return Box2D(x=x, y=y, w=w, h=h)

    def _scale_corners(self, corners, scale: float):
        if scale != 1.0:
            corners = np.array(corners, dtype=np.float32) / scale
        return tuple((int(round(x)), int(round(y))) for x, y in corners)

    def _order_points(self, points):
        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def _kernel(self, size: int):
        if size <= 0:
            return None
        size = self._odd_kernel(size)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    def _odd_kernel(self, size: int) -> int:
        size = max(1, int(size))
        if size % 2 == 0:
            size += 1
        return size
