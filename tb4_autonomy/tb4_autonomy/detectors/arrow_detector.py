from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from tb4_autonomy.data_types import ArrowDetection, Box2D


@dataclass
class ArrowDetectorConfig:
    # Legacy detector config
    threshold_method: str = 'otsu'
    hsv_v_max: int = 80
    blur_kernel: int = 5
    morph_open_kernel: int = 3
    morph_close_kernel: int = 5
    dilate_kernel: int = 7
    dilate_iterations: int = 1

    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.40
    min_area_px: float = 0.0
    max_area_px: float = 0.0
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

    # Paper candidate config
    paper_v_min: int = 110
    paper_s_max: int = 90
    paper_min_area_ratio: float = 0.001
    paper_max_area_ratio: float = 0.25
    paper_min_area_px: float = 0.0
    paper_max_area_px: float = 0.0
    paper_min_aspect_ratio: float = 0.3
    paper_max_aspect_ratio: float = 12.0

    # Floor / scene filters
    floor_roi_min_y_ratio: float = 0.45
    candidate_max_center_error_ratio: float = 0.45

    # Extra bbox sanity filters; keep permissive enough for close floor signs.
    min_candidate_bottom_ratio: float = 0.45
    max_candidate_height_ratio: float = 0.58
    max_candidate_width_ratio: float = 0.75
    max_candidate_area_ratio: float = 0.18
    min_candidate_area_px: float = 0.0
    max_candidate_area_px: float = 0.0

    # Compatibility with newer YAML/analyzer parameters.
    reject_back_direction: bool = True
    max_valid_bbox_width_ratio: float = 0.70
    max_valid_bbox_height_ratio: float = 0.70
    max_valid_final_area_ratio: float = 0.16
    min_valid_final_area_ratio: float = 0.001
    max_border_touch_ratio: float = 0.03
    max_valid_paper_aspect_ratio: float = 4.0
    min_valid_paper_aspect_ratio: float = 0.25
    min_history_confidence: float = 0.45

    use_axis_direction: bool = False
    use_paper_orientation_heading: bool = False
    paper_heading_forward_angle_rad: float = math.pi / 2.0
    paper_heading_use_previous_when_ambiguous: bool = True
    paper_heading_ambiguity_margin_rad: float = 0.20

    # Keep false by default. Merging fragments made boxes too large in this scene.
    merge_paper_fragments: bool = False
    min_arrow_presence_confidence: float = 0.35
    min_arrow_component_density: float = 0.25
    min_arrow_component_solidity: float = 0.42
    min_arrow_component_compactness: float = 0.055
    max_arrow_component_area_ratio: float = 0.20


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


@dataclass
class _ArrowGeometry:
    direction: str = 'unknown'
    mask: np.ndarray | None = None
    confidence: float = 0.0
    heading_angle_rad: float | None = None
    heading_valid: bool = False
    heading_confidence: float = 0.0
    arrow_presence_confidence: float = 0.0
    heading_base_warped: tuple[float, float] | None = None
    heading_tip_warped: tuple[float, float] | None = None
    warped_heading_debug_image: np.ndarray | None = None


class ArrowDetector:
    """Traditional OpenCV detector for printed arrow signs.

    This version intentionally keeps the legacy detection logic that worked best,
    but adds:
      - floor/height/large-box filters,
      - no paper-fragment merging by default,
      - compatibility fields for the newer overlay/analyzer.
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

        # In live/offline navigation context, prefer paper candidate first.
        # This avoids grabbing upper black/white wall objects before the floor paper.
        if context is None:
            candidate = self._select_candidate(mask, require_floor=False)
            if candidate is None:
                candidate = self._select_paper_candidate(scaled_frame, mask)
        else:
            candidate = self._select_paper_candidate(scaled_frame, mask)
            if candidate is None:
                candidate = self._select_candidate(mask)

        if candidate is None:
            self.direction_history.append('unknown')
            return None

        corners_small = self._estimate_corners(candidate, mask.shape[:2])
        corners = self._scale_corners(corners_small, scale)
        paper_transform = self._paper_warp_transform(corners)
        warped = cv2.warpPerspective(frame, paper_transform, (self.config.warp_width, self.config.warp_height))
        arrow_geometry = self._classify_warped_arrow(warped)
        raw_direction = arrow_geometry.direction
        arrow_mask = arrow_geometry.mask
        arrow_confidence = arrow_geometry.confidence

        if context is not None and self.config.reject_back_direction and raw_direction == 'back':
            raw_direction = 'unknown'
            arrow_confidence *= 0.25

        self.direction_history.append(raw_direction)
        stable_direction, stable_confidence, is_stable = self._stable_direction()
        direction = stable_direction if is_stable else 'unknown'

        box = self._scale_box(candidate.box, scale, original_width, original_height)
        center_error = box.center[0] - original_width / 2.0
        final_area_ratio = box.area / float(original_width * original_height)
        confidence = min(1.0, 0.55 * candidate.score + 0.45 * max(arrow_confidence, stable_confidence))

        heading_angle_rad = arrow_geometry.heading_angle_rad
        heading_valid = arrow_geometry.heading_valid
        heading_confidence = arrow_geometry.heading_confidence
        arrow_presence_confidence = arrow_geometry.arrow_presence_confidence
        heading_angle_deg = None if heading_angle_rad is None else math.degrees(heading_angle_rad)
        if not heading_valid or heading_angle_rad is None:
            heading_error_rad = None
            heading_source = 'none'
            heading_base = None
            heading_tip = None
        else:
            heading_error_rad = self._normalize_angle(heading_angle_rad - math.pi / 2.0)
            heading_source = 'warped_arrow_continuous'
            heading_base, heading_tip = self._project_warped_heading_line(
                paper_transform,
                arrow_geometry.heading_base_warped,
                arrow_geometry.heading_tip_warped,
            )

        detection = ArrowDetection(
            box=box,
            direction=direction,
            confidence=confidence,
            raw_direction=raw_direction,
            corners=corners,
            area_ratio=final_area_ratio,
            black_pixel_ratio=candidate.black_pixel_ratio,
            center_error_px=center_error,
            is_stable=is_stable,
            arrow_angle_rad=heading_angle_rad or 0.0,
            warped_debug_image=warped,
            mask_debug_image=arrow_mask,

            # New overlay/controller compatibility fields.
            heading_angle_rad=heading_angle_rad,
            heading_angle_deg=heading_angle_deg,
            heading_valid=heading_valid,
            heading_confidence=heading_confidence,
            arrow_presence_confidence=arrow_presence_confidence,
            heading_error_rad=heading_error_rad,
            heading_source=heading_source,
            heading_base=heading_base,
            heading_tip=heading_tip,
            final_confidence=confidence,
            template_direction=raw_direction,
            template_dominance=float(arrow_confidence),
            axis_direction='unknown',
            axis_confidence=0.0,
            paper_axis_angle_rad=None,
            paper_heading_angle_rad=None,
            warped_heading_debug_image=arrow_geometry.warped_heading_debug_image,
        )

        # Dynamic attributes for analyzer versions that expect them.
        detection.black_arrow_direction = raw_direction
        detection.black_arrow_confidence = float(arrow_confidence)
        detection.reject_reason = ''

        return detection

    def reset_history(self) -> None:
        self.direction_history.clear()

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

    def _select_candidate(self, mask, require_floor: bool = True):
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
            if require_floor and not self._is_floor_candidate(x, y, w, h, width, height):
                continue

            box_area = w * h
            area_ratio = box_area / image_area
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                continue
            if self.config.min_area_px > 0.0 and box_area < self.config.min_area_px:
                continue
            if self.config.max_area_px > 0.0 and box_area > self.config.max_area_px:
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

            candidates.append(_Candidate(
                box=Box2D(x=x, y=y, w=w, h=h),
                contour=contour,
                area_ratio=area_ratio,
                black_pixel_ratio=black_ratio,
                score=score,
            ))

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
        paper_mask[:int(paper_mask.shape[0] * self.config.floor_roi_min_y_ratio), :] = 0

        # Conservative kernels. Large kernels merged nearby signs into huge boxes.
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, self._kernel(3))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, self._kernel(5))

        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = paper_mask.shape[:2]
        image_area = float(width * height)
        fragments: list[_PaperFragment] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            if not self._is_floor_candidate(x, y, w, h, width, height):
                continue

            box_area = w * h
            area_ratio = box_area / image_area
            if area_ratio < self.config.paper_min_area_ratio or area_ratio > self.config.paper_max_area_ratio:
                continue
            if self.config.paper_min_area_px > 0.0 and box_area < self.config.paper_min_area_px:
                continue
            if self.config.paper_max_area_px > 0.0 and box_area > self.config.paper_max_area_px:
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
        # Keep each visible floor paper as its own candidate by default.
        boxes = [fragment.box for fragment in fragments]
        if self.config.merge_paper_fragments:
            candidate_boxes = boxes + self._merge_paper_boxes(boxes, width)
        else:
            candidate_boxes = boxes

        unique: dict[tuple[int, int, int, int], Box2D] = {}
        for box in candidate_boxes:
            key = (box.x, box.y, box.w, box.h)
            unique[key] = box

        candidates: list[_Candidate] = []
        for box in unique.values():
            x, y, w, h = box.x, box.y, box.w, box.h
            box_area = w * h
            if box_area <= 0:
                continue
            if not self._is_floor_candidate(x, y, w, h, width, height):
                continue

            area_ratio = box_area / image_area
            if area_ratio < self.config.paper_min_area_ratio or area_ratio > self.config.paper_max_area_ratio:
                continue
            if self.config.paper_min_area_px > 0.0 and box_area < self.config.paper_min_area_px:
                continue
            if self.config.paper_max_area_px > 0.0 and box_area > self.config.paper_max_area_px:
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
            bottom_score = (y + h) / float(height)
            area_score = min(1.0, area_ratio / 0.04)
            density_score = min(1.0, black_ratio / 0.25)
            score = 0.38 * bottom_score + 0.25 * area_score + 0.22 * density_score + 0.15 * center_score

            candidates.append(_Candidate(
                box=box,
                contour=self._rect_contour(box),
                area_ratio=area_ratio,
                black_pixel_ratio=black_ratio,
                score=score,
            ))
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

    def _is_floor_candidate(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
    ) -> bool:
        if image_width <= 0 or image_height <= 0:
            return False

        bottom_ratio = (y + h) / float(image_height)
        height_ratio = h / float(image_height)
        width_ratio = w / float(image_width)
        area_ratio = (w * h) / float(image_width * image_height)
        area_px = float(w * h)

        if bottom_ratio < self.config.min_candidate_bottom_ratio:
            return False
        if height_ratio > self.config.max_candidate_height_ratio:
            return False
        if width_ratio > self.config.max_candidate_width_ratio:
            return False
        if area_ratio > self.config.max_candidate_area_ratio:
            return False
        if self.config.min_candidate_area_px > 0.0 and area_px < self.config.min_candidate_area_px:
            return False
        if self.config.max_candidate_area_px > 0.0 and area_px > self.config.max_candidate_area_px:
            return False

        center_x = x + w / 2.0
        center_error = abs(center_x - image_width / 2.0)
        if center_error > image_width * self.config.candidate_max_center_error_ratio:
            return False
        return True

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
        transform = self._paper_warp_transform(corners)
        return cv2.warpPerspective(frame, transform, (self.config.warp_width, self.config.warp_height))

    def _paper_warp_transform(self, corners):
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
        return cv2.getPerspectiveTransform(src, dst)

    def _project_warped_heading_line(
        self,
        paper_transform,
        base_warped: tuple[float, float] | None,
        tip_warped: tuple[float, float] | None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if base_warped is None or tip_warped is None:
            return None, None
        inverse = np.linalg.inv(paper_transform)
        points = np.array([[base_warped, tip_warped]], dtype=np.float32)
        projected = cv2.perspectiveTransform(points, inverse).reshape(2, 2)
        base = (int(round(float(projected[0, 0]))), int(round(float(projected[0, 1]))))
        tip = (int(round(float(projected[1, 0]))), int(round(float(projected[1, 1]))))
        return base, tip

    def _classify_warped_arrow(self, warped):
        if warped is None or warped.size == 0:
            return _ArrowGeometry()

        mask = self._black_arrow_mask_in_warped_paper(warped)
        component_mask, contour, presence_confidence = self._select_arrow_component(mask)
        if contour is None:
            return _ArrowGeometry(
                mask=mask,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )
        if presence_confidence < self.config.min_arrow_presence_confidence:
            return _ArrowGeometry(
                direction='unknown',
                mask=component_mask,
                confidence=presence_confidence,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )

        contour_area = cv2.contourArea(contour)
        min_area = self.config.min_arrow_area_ratio * float(component_mask.shape[0] * component_mask.shape[1])
        if contour_area < min_area:
            return _ArrowGeometry(
                mask=component_mask,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )

        direction, dominance = self._direction_from_mask(contour, component_mask)
        area_confidence = min(1.0, contour_area / (min_area * 4.0))
        label_confidence = min(1.0, 0.65 * dominance + 0.35 * area_confidence)
        heading = self._continuous_heading_from_contour(contour, component_mask, warped, presence_confidence)
        heading.confidence = label_confidence
        heading.direction = direction
        return heading

    def _black_arrow_mask_in_warped_paper(self, warped):
        mask = self._preprocess_inner(warped)
        margin = int(min(mask.shape[:2]) * self.config.inner_crop_margin_ratio)
        if margin > 0:
            mask[:margin, :] = 0
            mask[mask.shape[0] - margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, mask.shape[1] - margin:] = 0

        frame = self._find_inner_frame(mask)
        if frame is not None:
            x, y, w, h = frame.box.x, frame.box.y, frame.box.w, frame.box.h
            inset = max(2, int(min(w, h) * 0.08))
            x1 = max(0, x + inset)
            y1 = max(0, y + inset)
            x2 = min(mask.shape[1], x + w - inset)
            y2 = min(mask.shape[0], y + h - inset)
            inner_only = np.zeros_like(mask)
            if x2 > x1 and y2 > y1:
                inner_only[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = inner_only

        open_kernel = self._kernel(3)
        close_kernel = self._kernel(5)
        if open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        if close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def _select_arrow_component(self, mask):
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask, None, 0.0

        image_area = float(mask.shape[0] * mask.shape[1])
        min_pixels = max(20, int(image_area * self.config.min_arrow_area_ratio * 0.20))
        best_label = None
        best_contour = None
        best_score = 0.0
        best_confidence = 0.0
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_pixels or w <= 0 or h <= 0:
                continue
            component_area_ratio = float(area) / image_area
            if component_area_ratio > self.config.max_arrow_component_area_ratio:
                continue
            if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 2 or y + h >= mask.shape[0] - 2:
                continue

            density = area / float(w * h)
            if density < self.config.min_arrow_component_density:
                continue
            aspect = w / float(h)
            if aspect < 0.20 or aspect > 5.0:
                continue

            component_mask = np.zeros_like(mask)
            component_mask[labels == label] = 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0.0:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            perimeter = float(cv2.arcLength(contour, True))
            solidity = 0.0 if hull_area <= 1e-6 else contour_area / hull_area
            compactness = 0.0 if perimeter <= 1e-6 else (4.0 * math.pi * contour_area) / (perimeter * perimeter)
            if solidity < self.config.min_arrow_component_solidity:
                continue
            if compactness < self.config.min_arrow_component_compactness:
                continue

            center_y_score = 1.0 - min(1.0, abs((y + h / 2.0) - mask.shape[0] / 2.0) / (mask.shape[0] / 2.0))
            center_x_score = 1.0 - min(1.0, abs((x + w / 2.0) - mask.shape[1] / 2.0) / (mask.shape[1] / 2.0))
            density_score = min(1.0, density / 0.45)
            area_score = min(1.0, area / max(1.0, image_area * self.config.min_arrow_area_ratio * 1.8))
            size_score = min(1.0, max(w, h) / max(1.0, min(mask.shape[:2]) * 0.20))
            solidity_score = min(1.0, solidity / 0.75)
            compactness_score = min(1.0, compactness / 0.22)
            presence_confidence = (
                0.24 * area_score
                + 0.22 * density_score
                + 0.20 * solidity_score
                + 0.16 * compactness_score
                + 0.08 * center_y_score
                + 0.05 * center_x_score
                + 0.05 * size_score
            )
            score = float(area) * presence_confidence
            if score > best_score:
                best_score = score
                best_label = label
                best_contour = contour
                best_confidence = presence_confidence

        if best_label is None or best_contour is None:
            return mask, None, 0.0

        component_mask = np.zeros_like(mask)
        component_mask[labels == best_label] = 255
        return component_mask, best_contour, best_confidence

    def _continuous_heading_from_contour(self, contour, mask, warped, presence_confidence: float) -> _ArrowGeometry:
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-6:
            return _ArrowGeometry(
                mask=mask,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )

        cx = float(moments['m10'] / moments['m00'])
        cy = float(moments['m01'] / moments['m00'])
        base_tip = self._base_and_tip_from_mask_axis(mask, cx, cy)
        if base_tip is None:
            tip = self._tip_from_contour(contour, cx, cy)
            base = np.array([cx, cy], dtype=np.float32)
        else:
            base, tip = base_tip
        if tip is None or base is None:
            return _ArrowGeometry(
                mask=mask,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )
        bx = float(base[0])
        by = float(base[1])
        tx = float(tip[0])
        ty = float(tip[1])
        dx = tx - bx
        dy = ty - by
        distance = math.hypot(dx, dy)
        diagonal = math.hypot(mask.shape[1], mask.shape[0])
        if distance < max(8.0, diagonal * 0.04):
            return _ArrowGeometry(
                mask=mask,
                arrow_presence_confidence=presence_confidence,
                warped_heading_debug_image=self._draw_invalid_warped_debug(warped, presence_confidence),
            )

        heading_angle_rad = self._flip_robot_facing_heading(math.atan2(-dy, dx))
        heading_angle_deg = math.degrees(heading_angle_rad)
        length = max(35.0, min(mask.shape[:2]) * 0.22)
        end_x = bx + math.cos(heading_angle_rad) * length
        end_y = by - math.sin(heading_angle_rad) * length
        geometric_confidence = min(1.0, distance / max(1.0, diagonal * 0.18))
        heading_confidence = min(1.0, 0.55 * geometric_confidence + 0.45 * presence_confidence)

        debug = warped.copy()
        cv2.circle(debug, (int(round(bx)), int(round(by))), 5, (0, 255, 255), -1)
        cv2.circle(debug, (int(round(tx)), int(round(ty))), 5, (0, 0, 255), -1)
        cv2.arrowedLine(
            debug,
            (int(round(bx)), int(round(by))),
            (int(round(end_x)), int(round(end_y))),
            (255, 0, 0),
            3,
            tipLength=0.28,
        )
        cv2.putText(
            debug,
            f'heading={heading_angle_deg:.1f} deg',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return _ArrowGeometry(
            mask=mask,
            confidence=heading_confidence,
            heading_angle_rad=heading_angle_rad,
            heading_valid=True,
            heading_confidence=heading_confidence,
            arrow_presence_confidence=presence_confidence,
            heading_base_warped=(bx, by),
            heading_tip_warped=(end_x, end_y),
            warped_heading_debug_image=debug,
        )

    def _flip_robot_facing_heading(self, heading_angle_rad: float) -> float:
        # In the warped paper convention, robot-forward is +pi/2. A clean arrow
        # that points down toward the camera usually means base/tip were swapped.
        # Flip only down-facing headings; left/right/forward arrows are left alone.
        down_error = abs(self._normalize_angle(heading_angle_rad + math.pi / 2.0))
        if down_error <= math.radians(65.0):
            return self._normalize_angle(heading_angle_rad + math.pi)
        return self._normalize_angle(heading_angle_rad)

    def _draw_invalid_warped_debug(self, warped, presence_confidence: float):
        debug = warped.copy()
        cv2.putText(
            debug,
            f'arrow_conf={presence_confidence:.2f} invalid',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    def _base_and_tip_from_mask_axis(self, mask, cx: float, cy: float):
        ys, xs = np.nonzero(mask)
        if len(xs) < 12:
            return None
        points = np.column_stack([xs, ys]).astype(np.float32)
        centered = points - np.mean(points, axis=0, keepdims=True)
        try:
            _eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        except np.linalg.LinAlgError:
            return None
        axis = eigenvectors[:, -1].astype(np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        axis /= norm
        perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)

        projections = points @ axis
        perp_values = points @ perpendicular
        min_projection = float(np.min(projections))
        max_projection = float(np.max(projections))
        span = max_projection - min_projection
        if span < 8.0:
            return None

        band = max(3.0, span * 0.16)
        low_mask = projections <= min_projection + band
        high_mask = projections >= max_projection - band
        if np.count_nonzero(low_mask) < 2 or np.count_nonzero(high_mask) < 2:
            return None

        low_spread = float(np.std(perp_values[low_mask]))
        high_spread = float(np.std(perp_values[high_mask]))
        centroid_projection = float(np.array([cx, cy], dtype=np.float32) @ axis)
        low_distance = abs(centroid_projection - min_projection)
        high_distance = abs(max_projection - centroid_projection)

        if abs(low_spread - high_spread) > max(1.5, span * 0.02):
            use_high = high_spread < low_spread
        else:
            use_high = high_distance >= low_distance

        if use_high:
            tip_indices = np.where(high_mask)[0]
            base_indices = np.where(low_mask)[0]
            tip = points[tip_indices[int(np.argmax(projections[tip_indices]))]]
        else:
            tip_indices = np.where(low_mask)[0]
            base_indices = np.where(high_mask)[0]
            tip = points[tip_indices[int(np.argmin(projections[tip_indices]))]]

        base_points = points[base_indices]
        base = np.mean(base_points, axis=0)
        return base.astype(np.float32), tip.astype(np.float32)

    def _tip_from_mask_axis(self, mask, cx: float, cy: float):
        base_tip = self._base_and_tip_from_mask_axis(mask, cx, cy)
        if base_tip is None:
            return None
        _base, tip = base_tip
        return tip

    def _tip_from_contour(self, contour, cx: float, cy: float):
        hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
        if len(hull) < 3:
            return None

        center = np.array([cx, cy], dtype=np.float32)
        best_point = None
        best_score = 0.0
        for index, point in enumerate(hull):
            prev_point = hull[(index - 1) % len(hull)]
            next_point = hull[(index + 1) % len(hull)]
            v1 = prev_point - point
            v2 = next_point - point
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_angle = float(np.dot(v1, v2) / (n1 * n2))
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)
            sharpness = max(0.0, math.pi - angle)
            distance = float(np.linalg.norm(point - center))
            score = distance * (0.35 + sharpness)
            if score > best_score:
                best_score = score
                best_point = point
        return best_point

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
        if best_score < 0.34 or (best_score - runner_up) < 0.03:
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
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return math.pi / 2.0

        base_offsets = {
            'straight': (x + w / 2.0, y + h),
            'back': (x + w / 2.0, float(y)),
            'left': (x + float(w), y + h / 2.0),
            'right': (float(x), y + h / 2.0),
        }
        base = base_offsets.get(direction)
        if base is None:
            moments = cv2.moments(contour)
            if abs(moments['m00']) < 1e-6:
                return math.pi / 2.0
            base = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])

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

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _heading_debug_line(self, box: Box2D, heading_angle_rad: float):
        cx, cy = box.center
        length = max(20.0, min(box.w, box.h) * 0.8)

        dx = math.cos(heading_angle_rad)
        dy = -math.sin(heading_angle_rad)

        base_x = cx - dx * length * 0.4
        base_y = cy - dy * length * 0.4
        tip_x = cx + dx * length * 0.6
        tip_y = cy + dy * length * 0.6

        return (
            (int(round(base_x)), int(round(base_y))),
            (int(round(tip_x)), int(round(tip_y))),
        )
