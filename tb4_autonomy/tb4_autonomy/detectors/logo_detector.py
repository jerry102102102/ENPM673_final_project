from __future__ import annotations

import cv2
import numpy as np
import os

from tb4_autonomy.data_types import LogoDetection, Box2D   # Assuming these are defined in tb4_autonomy.data_types

try:
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
except ModuleNotFoundError:
    get_package_share_directory = None

    class PackageNotFoundError(Exception):
        pass

class LogoDetector:
    name = 'logo'   # Unique name for this detector

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=800)   # Increase number of features to improve detection robustness

        ref_path = self._find_reference_logo_path()

        self.ref_img = cv2.imread(ref_path, 0)   # Load the logo image in grayscale
        if self.ref_img is None: 
            raise RuntimeError(f"Failed to load logo image at {ref_path}")

        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_img, None)   # Compute keypoints and descriptors for the reference logo image

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)   # Use Brute-Force matcher with Hamming distance for ORB descriptors

        self.detect_count = 0   # Counter to keep track of consecutive detections
        self.detect_threshold = 5   # Number of consecutive detections required to confirm the presence of the logo

        self.min_matches = 15   # Minimum number of good matches required to consider a detection valid
        self.ratio_thresh = 0.75   # Lowe's ratio test threshold to filter out false matches

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

        if des_frame is None:   # If no descriptors are found in the frame, reset the detection count and return None
            self.detect_count = 0
            return None

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
        return None
