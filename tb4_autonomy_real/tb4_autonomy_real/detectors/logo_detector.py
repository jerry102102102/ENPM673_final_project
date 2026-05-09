from __future__ import annotations

import cv2
import numpy as np
import os

from tb4_autonomy_real.data_types import LogoDetection, Box2D   # Assuming these are defined in tb4_autonomy_real.data_types

try:
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
except ModuleNotFoundError:
    get_package_share_directory = None

    class PackageNotFoundError(Exception):
        pass

class LogoDetector:
    name = 'logo'   # Unique name for this detector

    def __init__(self, detect_threshold: int = 5):
        self.orb = cv2.ORB_create(nfeatures=800)   # Increase number of features to improve detection robustness

        ref_path = self._find_reference_logo_path()

        self.ref_img = cv2.imread(ref_path, 0)   # Load the logo image in grayscale
        if self.ref_img is None: 
            raise RuntimeError(f"Failed to load logo image at {ref_path}")

        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_img, None)   # Compute keypoints and descriptors for the reference logo image

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)   # Use Brute-Force matcher with Hamming distance for ORB descriptors

        self.detect_count = 0   # Counter to keep track of consecutive detections
        self.detect_threshold = max(1, int(detect_threshold))   # Number of consecutive detections required to confirm the presence of the logo

        self.min_matches = 15   # Minimum number of good matches required to consider a detection valid
        self.ratio_thresh = 0.75   # Lowe's ratio test threshold to filter out false matches
        self.color_detect_count = 0

    def _find_reference_logo_path(self):
        candidates = []
        if get_package_share_directory is not None:
            try:
                candidates.append(os.path.join(get_package_share_directory('tb4_sim'), 'textures', 'logo.png'))
            except PackageNotFoundError:
                pass

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.abspath(os.path.join(current_dir, '../../../../src/textures/logo.png')),
            os.path.abspath(os.path.join(os.getcwd(), 'src/textures/logo.png')),
        ])

        for path in candidates:
            if os.path.exists(path):
                return path
        raise RuntimeError(f"Failed to find logo.png. Checked: {candidates}")

    def detect(self, frame, context):   # Main detection method that takes a video frame and context (not used in this implementation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Convert the input frame to grayscale for feature detection

        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)   # Compute keypoints and descriptors for the input frame

        if des_frame is None:   # If no descriptors are found in the frame, try the real-video color fallback.
            self.detect_count = 0
            return self._detect_color_logo(frame)

        matches = self.bf.knnMatch(self.des_ref, des_frame, k=2)   # Find the two nearest matches for each descriptor in the reference image

        good_matches = []   # Filter matches using Lowe's ratio test to retain only good matches
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair   # m is the best match, n is the second-best match
            if m.distance < self.ratio_thresh * n.distance:   # If the best match is significantly better than the second-best match, consider it a good match
                good_matches.append(m)

        if len(good_matches) > self.min_matches:   # If there are enough good matches, proceed to find the homography
            src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)   # Get the coordinates of the matched keypoints in the reference image
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)   # Get the coordinates of the matched keypoints in the input frame

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   # Compute the homography matrix using RANSAC to filter out outliers

            if H is not None:   # If a valid homography is found, compute the bounding box of the detected logo in the input frame
                h, w = self.ref_img.shape   # Get the height and width of the reference logo image
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)   # Define the corners of the reference logo image
                dst = cv2.perspectiveTransform(pts, H)   # Transform the corners of the reference logo image to the input frame using the homography

                x, y, bw, bh = cv2.boundingRect(dst)   # Compute the bounding box of the detected logo in the input frame

                self.detect_count += 1

                if self.detect_count >= self.detect_threshold:   # If the logo has been detected in enough consecutive frames, return a LogoDetection object with the bounding box and confidence
                    box = Box2D(x, y, bw, bh)   # Create a Box2D object with the coordinates and dimensions of the bounding box
                    return LogoDetection(box=box, confidence=len(good_matches))   # Return a LogoDetection object with the bounding box and confidence based on the number of good matches
                return None

        self.detect_count = 0
        return self._detect_color_logo(frame)

    def _detect_color_logo(self, frame):
        """Fallback for low-res real video where ORB matches are too weak.

        The UMD logo board is a white sign with red/yellow/black colored logo
        content. This fallback first finds white rectangular sign candidates,
        then verifies that the interior contains enough red/yellow/dark pixels.
        """
        colored_cluster = self._detect_logo_from_colored_cluster(frame)
        if colored_cluster is not None:
            return colored_cluster

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 120), (180, 75, 255))
        white_mask[:int(height * 0.20), :] = 0
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = 0.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 900 or area > width * height * 0.35:
                continue
            aspect = w / float(max(1, h))
            if aspect < 0.65 or aspect > 1.8:
                continue
            if x < width * 0.35:
                continue

            roi = frame[y:y + h, x:x + w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red = cv2.inRange(roi_hsv, (0, 55, 55), (12, 255, 255)) | cv2.inRange(roi_hsv, (165, 55, 55), (180, 255, 255))
            yellow = cv2.inRange(roi_hsv, (12, 45, 70), (42, 255, 255))
            dark = cv2.inRange(roi_hsv, (0, 0, 0), (180, 120, 90))
            red_yellow_ratio = cv2.countNonZero(red | yellow) / float(max(1, area))
            colored = cv2.countNonZero(red | yellow | dark)
            color_ratio = colored / float(max(1, area))
            if color_ratio < 0.045 or red_yellow_ratio < 0.020:
                continue
            center_score = 1.0 - min(1.0, abs((x + w / 2.0) - width * 0.75) / max(1.0, width * 0.45))
            score = color_ratio * 100.0 + center_score + min(1.0, area / 4000.0)
            if score > best_score:
                best_score = score
                best_box = Box2D(int(x), int(y), int(w), int(h))

        if best_box is None:
            self.color_detect_count = 0
            return None

        self.color_detect_count += 1
        if self.color_detect_count < self.detect_threshold:
            return None
        self.detect_count = self.color_detect_count
        return LogoDetection(box=best_box, confidence=float(best_score))

    def _detect_logo_from_colored_cluster(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = (
            cv2.inRange(hsv, (0, 55, 55), (12, 255, 255))
            | cv2.inRange(hsv, (165, 55, 55), (180, 255, 255))
        )
        yellow = cv2.inRange(hsv, (12, 45, 70), (42, 255, 255))
        mask = red | yellow
        mask[:int(height * 0.28), :] = 0
        mask[:, :int(width * 0.55)] = 0
        mask[int(height * 0.86):, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 80.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w > width * 0.30 or h > height * 0.28:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            if cx < width * 0.64 or cy < height * 0.32 or cy > height * 0.78:
                continue

            pad_x = int(max(w * 1.20, 22))
            pad_y = int(max(h * 1.10, 20))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(width, x + w + pad_x)
            y1 = min(height, y + h + pad_y)
            bw = x1 - x0
            bh = y1 - y0
            if x0 < width * 0.50:
                continue
            if bw < 35 or bh < 35:
                continue
            aspect = bw / float(max(1, bh))
            if aspect < 0.45 or aspect > 1.65:
                continue

            roi = frame[y0:y1, x0:x1]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white = cv2.inRange(roi_hsv, (0, 0, 115), (180, 90, 255))
            white_ratio = cv2.countNonZero(white) / float(max(1, bw * bh))
            if white_ratio < 0.18:
                continue
            score = area + 250.0 * white_ratio + 100.0 * (cx / max(1.0, width))
            if score > best_score:
                best_score = score
                best_box = Box2D(int(x0), int(y0), int(bw), int(bh))

        if best_box is None:
            return None

        self.color_detect_count += 1
        if self.color_detect_count < self.detect_threshold:
            return None
        self.detect_count = self.color_detect_count
        return LogoDetection(box=best_box, confidence=float(best_score))
