"""Quick test: run ArrowDetector pipeline on the sample image."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tb4_autonomy"))

import cv2
from tb4_autonomy.detectors.arrow_detector import ArrowDetector, ArrowDetectorConfig

IMG_PATH = os.path.join(os.path.dirname(__file__), "tb4_spawn_down_pitch_camera.jpg")

frame = cv2.imread(IMG_PATH)
if frame is None:
    print(f"ERROR: cannot read {IMG_PATH}")
    sys.exit(1)

print(f"Image shape: {frame.shape}")

detector = ArrowDetector()
result = detector.detect(frame, context=None)

if result is None:
    print("Detection result: None (no arrow detected)")
else:
    print(f"Direction      : {result.direction}")
    print(f"Raw direction  : {result.raw_direction}")
    print(f"Confidence     : {result.confidence:.3f}")
    print(f"Is stable      : {result.is_stable}")
    print(f"Box            : x={result.box.x} y={result.box.y} w={result.box.w} h={result.box.h}")
    print(f"Area ratio     : {result.area_ratio:.4f}")
    print(f"Black px ratio : {result.black_pixel_ratio:.4f}")
    print(f"Center error   : {result.center_error_px:.1f} px")
    print(f"Corners        : {result.corners}")

    # Save debug images
    if result.warped_debug_image is not None:
        cv2.imwrite("/root/test/ENPM673-Final-Project-Simulation/debug_warped.jpg", result.warped_debug_image)
        print("Saved debug_warped.jpg")
    if result.mask_debug_image is not None:
        cv2.imwrite("/root/test/ENPM673-Final-Project-Simulation/debug_mask.jpg", result.mask_debug_image)
        print("Saved debug_mask.jpg")

    # Draw detection on original image
    vis = frame.copy()
    bx, by, bw, bh = result.box.x, result.box.y, result.box.w, result.box.h
    cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    cv2.putText(vis, f"{result.direction} ({result.confidence:.2f})",
                (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("/root/test/ENPM673-Final-Project-Simulation/debug_detection.jpg", vis)
    print("Saved debug_detection.jpg")
