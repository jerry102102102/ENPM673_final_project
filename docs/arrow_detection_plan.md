Arrow Detection and Local Navigation Plan
Goal

Implement a traditional OpenCV-based arrow detection and local navigation module for the TurtleBot4 final project.

The robot must:

Detect black arrow signs printed on white paper.
Draw a green bounding rectangle around the detected arrow or sign in real time.
Determine the arrow direction: LEFT, RIGHT, STRAIGHT, or BACK/END.
Align itself with the sign before reading the arrow.
Execute a relative motion command based on the arrow direction.
Avoid repeatedly triggering on the same sign.

This implementation should use traditional computer vision instead of machine learning.

High-Level Design

The system should be split into two logical parts.

1. Arrow Detection

The arrow detector receives camera frames and produces a clean detection result.

Pipeline:

Receive camera frame
Detect possible arrow sign region
Estimate sign corners
Apply homography to rectify the sign
Segment the arrow in the warped image
Classify arrow direction
Draw green bounding box or polygon
Output detection result
2. Local Navigation and Correction

The navigation controller uses the detection result to move the robot.

Pipeline:

Use bounding box center to align robot with the sign
Use bounding box size as a rough distance estimate
Stop near the sign
Read stable arrow direction
Execute relative turn using odometry or IMU
Enter cooldown to avoid repeated triggering
Core Principle

The arrow sign is a planar object.

Therefore, after detecting the sign region, apply a homography to rectify the sign into a front-facing canonical image before classifying the arrow direction.

Correct idea:

Detect sign in original camera image
Find the four sign corners
Compute homography for the sign only
Warp the sign to a canonical square image
Classify the arrow in the warped image

Wrong idea:

Apply one homography to the entire camera scene

Reason:

A single homography only works reliably for one plane. The full camera scene contains floor, wall, paper, background objects, and robot parts, so the whole image is not one single plane. The paper sign itself is planar, so homography is appropriate for that ROI.

Recommended File Structure

Adapt this structure to the existing project if needed.

src/detection/arrow_detector.py
src/detection/arrow_types.py
src/navigation/arrow_nav_controller.py
src/nodes/arrow_detector_node.py
src/nodes/arrow_navigation_node.py
src/config/arrow_detection.yaml

The detector and controller should be separated as much as possible.

The detector answers:

What arrow did I see?

The navigation controller answers:

What should the robot do with that arrow?

Detection Result Data Model

Create a clean detection output object.

ArrowDetectionResult

Fields:

detected: bool
direction: str
one of LEFT, RIGHT, STRAIGHT, BACK, UNKNOWN
confidence: float
bbox: tuple[int, int, int, int]
(x, y, w, h) in original image coordinates
corners: optional list[point]
four sign corners in original image coordinates
center: tuple[int, int]
center of bbox in original image coordinates
area: float
bbox or contour area
warped_debug_image: optional image
rectified sign image for debugging
mask_debug_image: optional image
arrow segmentation mask for debugging

The detector should return this result every frame.

Detection Pipeline
Step 1: Receive Camera Frame

Input:

BGR camera frame from ROS image topic

Output:

Current OpenCV image frame

The detector should process one frame at a time.

Step 2: Preprocess Frame

Convert the frame into a format suitable for high-contrast detection.

Recommended process:

Receive BGR frame
Resize if needed
Convert to grayscale
Apply Gaussian blur
Threshold dark regions

The arrow and possibly the border are black, so the first useful mask is a black-pixel mask.

Use either:

Option A: grayscale plus Otsu threshold
Option B: HSV threshold using low V values

Recommended starting point:

Start with grayscale plus Otsu threshold because it is simple.
Keep HSV threshold as a backup because real demo lighting may vary.

Suggested config values:

threshold_method: otsu
hsv_v_max: 80
blur_kernel: 5
Step 3: Morphological Cleanup

The threshold mask may contain noise.

Apply morphology:

open: removes tiny noise
close or dilate: connects fragmented black regions

Purpose:

Remove small background artifacts.
Connect the arrow body and printed border if fragmented.
Make contour extraction more stable.

Suggested config values:

morph_open_kernel: 3
morph_close_kernel: 5
dilate_kernel: 7
dilate_iterations: 1
Step 4: Find Candidate Sign Regions

Use contours on the cleaned black mask.

For each contour, compute:

Bounding box
Area
Aspect ratio
Black pixel density inside bbox
Position relative to image center

Filter candidates using these rules:

Area is not too small
Area is not too large
Aspect ratio is reasonable
Black pixel density is reasonable
Candidate is not mostly outside the image

Recommended initial rules:

area > 0.5% of image area
area < 40% of image area
0.5 < width / height < 2.0
black pixel ratio inside bbox is above a minimum threshold

Suggested config values:

min_area_ratio: 0.005
max_area_ratio: 0.40
min_aspect_ratio: 0.5
max_aspect_ratio: 2.0
min_black_pixel_ratio: 0.03
Step 5: Select Best Candidate

If multiple candidate regions exist, choose the most likely sign.

Scoring can use:

Larger area
Closer to image center
Reasonable aspect ratio
Strong black-pixel density
Rectangularity

Recommended simple scoring:

area_score
center_score
aspect_ratio_score
black_density_score

The selected candidate should provide:

Bbox in original image
Candidate contour
Candidate ROI

This bbox is also used for the green detection overlay.

Step 6: Estimate Four Sign Corners

Try to estimate the four corners of the paper or sign.

Preferred method:

Approximate candidate contour with a polygon.
If the polygon has four points, use these as sign corners.

Fallback method:

Use minAreaRect.
If that also fails, use normal bbox corners.

The corners must be ordered consistently:

Top-left
Top-right
Bottom-right
Bottom-left

This ordering is necessary for homography.

Step 7: Homography Rectification

Use the four sign corners from the original image and map them to a canonical front-facing rectangle.

Canonical destination points:

(0, 0)
(W, 0)
(W, H)
(0, H)

Recommended canonical size:

warp_width: 300
warp_height: 300

Output:

Warped sign image

This image should look like a front-facing paper sign.

All arrow classification should happen on this warped image.

Step 8: Remove Outer Border

The sign may contain a black printed border. This border can confuse arrow extraction.

After warping, crop away the outer margin.

Recommended:

Remove 10% to 15% from each side.

Suggested config value:

inner_crop_margin_ratio: 0.12

Input:

Warped sign image

Output:

Inner sign image

The inner image should mostly contain the arrow body.

Step 9: Segment Arrow in Warped Image

Convert the inner warped image to grayscale and threshold black pixels again.

Then find contours inside the inner crop.

The arrow should be the largest black connected component.

Process:

Take inner warped image
Convert to grayscale
Threshold black pixels
Apply morphology cleanup
Find contours
Choose largest valid contour as arrow contour

Reject the result if:

No contour is found
Largest contour area is too small
Arrow touches too much of the crop border
Shape is too fragmented

Output:

Arrow contour in warped image coordinates
Step 10: Classify Arrow Direction

Use geometry first.

Method:

Compute the centroid of the arrow contour.
Find the contour point farthest from the centroid.
Treat this farthest point as the arrow tip.
Compute vector from centroid to arrow tip.
Convert vector into direction.

Direction rules in warped image:

Tip above centroid: STRAIGHT
Tip left of centroid: LEFT
Tip right of centroid: RIGHT
Tip below centroid: BACK or END

Use angle thresholds with margin.

Recommended direction categories:

Around positive x-axis: RIGHT
Around negative x-axis: LEFT
Around upward direction: STRAIGHT
Around downward direction: BACK

Be careful that image y-axis points downward.

If the angle is ambiguous, return:

UNKNOWN
Optional Backup: Template Matching

Add this only if geometry-based classification is unstable.

Use known arrow templates:

STRAIGHT
LEFT
RIGHT
BACK

After warping and thresholding, compare the candidate arrow mask against each template.

Use this as a secondary confidence check.

Logic:

If geometry direction and template direction agree, confidence is high.
If they disagree, lower confidence or return UNKNOWN.

This keeps the system traditional CV-based while improving robustness.

Green Bounding Box Overlay

The green bounding box should be drawn on the original camera frame.

Do not only draw it on the warped image.

Draw either:

Candidate bbox, or
Polygon connecting the four detected sign corners

Recommended behavior:

Draw green polygon if corners are valid.
Otherwise draw green rectangular bbox.

Also overlay text:

ARROW: LEFT
ARROW: RIGHT
ARROW: STRAIGHT
ARROW: UNKNOWN
Temporal Smoothing

Do not accept direction from a single frame.

Maintain a short detection history.

Recommended values:

History length: 5 to 7 frames
Minimum agreement: 4 frames

Example:

Recent detections:

LEFT
LEFT
LEFT
UNKNOWN
LEFT

Stable output:

LEFT

If not enough agreement exists:

stable_direction = UNKNOWN

Suggested config values:

history_size: 5
min_stable_count: 4

This avoids random one-frame errors.

Local Navigation and Alignment

The robot does not need global orientation or full localization for this task.

It only needs local reactive navigation.

Use four signals:

Bbox center offset
Used to align the robot with the sign.
Bbox area
Used as a rough distance proxy.
Arrow direction
Used as the next relative navigation instruction.
Odometry or IMU yaw
Used to execute accurate turns.

Important idea:

Arrow direction is not used to estimate global robot heading.
Arrow direction is the next navigation instruction.

The robot should first align itself with the sign, then read the arrow, then execute a relative turn.

Navigation State Machine

Implement the behavior as a state machine.

Recommended states:

SEARCH_SIGN
ALIGN_TO_SIGN
APPROACH_SIGN
READ_ARROW
EXECUTE_TURN
COOLDOWN
State 1: SEARCH_SIGN

Purpose:

Find an arrow sign.

Behavior:

If no sign is detected, move slowly forward or rotate slowly to search.
If a sign is detected, switch to ALIGN_TO_SIGN.

Keep this conservative.

State 2: ALIGN_TO_SIGN

Purpose:

Center the sign in the camera image.

Use:

Bbox center x
Image center x

Compute:

error_x = bbox_center_x - image_center_x

Behavior:

If error_x is negative, the sign is left of center, so rotate left slowly.
If error_x is positive, the sign is right of center, so rotate right slowly.
If abs(error_x) is below threshold, the sign is centered enough, so switch to APPROACH_SIGN.

Suggested config values:

center_tolerance_px: 40
align_angular_speed: 0.2
State 3: APPROACH_SIGN

Purpose:

Move closer until the sign is large enough to read reliably.

Use bbox area as a distance proxy.

Behavior:

If bbox area is smaller than target, move forward slowly.
While moving forward, keep a small angular correction using bbox center.
If bbox area is large enough, stop and switch to READ_ARROW.

Suggested config values:

target_bbox_area_ratio: 0.08
approach_linear_speed: 0.08
approach_angular_gain: 0.002

This avoids reading arrows that are too small or noisy.

State 4: READ_ARROW

Purpose:

Stop and classify the arrow direction.

Behavior:

Stop the robot.
Collect several frames.
Run arrow detector.
Wait for stable direction.

If stable direction is:

LEFT: switch to EXECUTE_TURN with target relative yaw +90 degrees
RIGHT: switch to EXECUTE_TURN with target relative yaw -90 degrees
STRAIGHT: move forward for a short distance or switch to COOLDOWN
BACK: either stop, U-turn, or treat as end marker depending on final task definition
UNKNOWN: keep reading until timeout

Suggested config value:

read_timeout_sec: 3.0

For the first implementation, if timeout occurs:

Stop
Log UNKNOWN
Do not take risky motion
State 5: EXECUTE_TURN

Purpose:

Execute the relative turn commanded by the arrow.

Do not rely only on fixed time.

Preferred method:

Use odometry or IMU yaw.

Behavior:

Record current yaw.
Compute target yaw.
Rotate until yaw error is small.
Stop.
Switch to COOLDOWN.

Recommended turn angles:

LEFT: +90 degrees
RIGHT: -90 degrees
STRAIGHT: 0 degrees
BACK: 180 degrees or stop, depending on final design

Suggested config values:

turn_angle_left_deg: 90
turn_angle_right_deg: -90
turn_yaw_tolerance_deg: 5
turn_angular_speed: 0.3

If odometry or IMU is not available yet, implement time-based turning as a temporary fallback only.

State 6: COOLDOWN

Purpose:

Prevent repeated triggering on the same sign.

After executing a command, the same sign may still be visible.

Behavior:

Ignore arrow detections for a short time.
Move forward a short distance.
Return to SEARCH_SIGN.

Suggested config values:

cooldown_sec: 1.5
post_turn_forward_sec: 0.8
post_turn_forward_speed: 0.08

Alternative condition:

Stay in cooldown until the previous bbox disappears from view.

For the initial implementation, fixed cooldown time is acceptable.

Control Summary

Use different signals for different jobs.

Bbox Center Offset

Purpose:

Align robot with the sign.

Meaning:

Bbox center left of image center means the sign is left of the robot.
Bbox center right of image center means the sign is right of the robot.
Bbox center near image center means the robot is facing the sign well enough.
Bbox Area

Purpose:

Estimate whether the robot is close enough to read the sign.

Meaning:

Small bbox means sign is far away.
Large bbox means sign is close enough.
Arrow Direction

Purpose:

Decide next relative navigation command.

Meaning:

LEFT means turn left relative to current robot orientation.
RIGHT means turn right relative to current robot orientation.
STRAIGHT means continue forward.
BACK means U-turn or end behavior, depending on final definition.
Odometry or IMU Yaw

Purpose:

Execute accurate turns.

Meaning:

Use yaw feedback to rotate about 90 degrees instead of relying only on fixed time.
Debug Outputs

The implementation should support debug visualization.

Display or publish:

Original frame with green bbox or polygon
Threshold mask
Selected candidate ROI
Warped sign image
Arrow mask inside warped image
Final direction text
Navigation state text

Overlay useful text on the original frame:

STATE: ALIGN_TO_SIGN
DIR: LEFT
CONF: 0.86
AREA: 0.092
CENTER_ERR: -24 px

This will make demo debugging much easier.

Suggested YAML Parameters

Put these in a config file so they can be tuned quickly.

arrow_detection
threshold_method: otsu
hsv_v_max: 80
blur_kernel: 5
morph_open_kernel: 3
morph_close_kernel: 5
dilate_kernel: 7
dilate_iterations: 1
min_area_ratio: 0.005
max_area_ratio: 0.40
min_aspect_ratio: 0.5
max_aspect_ratio: 2.0
min_black_pixel_ratio: 0.03
warp_width: 300
warp_height: 300
inner_crop_margin_ratio: 0.12
history_size: 5
min_stable_count: 4
navigation
center_tolerance_px: 40
align_angular_speed: 0.2
target_bbox_area_ratio: 0.08
approach_linear_speed: 0.08
approach_angular_gain: 0.002
read_timeout_sec: 3.0
turn_angle_left_deg: 90
turn_angle_right_deg: -90
turn_angle_back_deg: 180
turn_yaw_tolerance_deg: 5
turn_angular_speed: 0.3
cooldown_sec: 1.5
post_turn_forward_sec: 0.8
post_turn_forward_speed: 0.08
Implementation Order
Phase 1: Offline or Single-Frame Detector

Build the detector first without robot motion.

Input:

Saved image or live camera frame

Output:

Green bbox
Warped sign image
Arrow direction
Debug masks

Acceptance criteria:

Can detect arrow sign in camera frame.
Can draw green bbox.
Can generate warped front-facing sign.
Can classify LEFT, RIGHT, and STRAIGHT on clear signs.
Phase 2: Live Detector Node

Connect detector to ROS camera topic.

Input:

/camera/image_raw

Output:

Annotated image window or ROS image topic
Detection result topic or logs

Acceptance criteria:

Detector runs in real time.
Green bbox is visible.
Direction text is stable when robot is still.
Debug windows or topics are available.
Phase 3: Temporal Smoothing

Add detection history.

Acceptance criteria:

Single-frame flicker does not trigger direction changes.
Stable direction only appears after repeated agreement.
UNKNOWN is returned when detections are inconsistent.
Phase 4: Alignment Controller

Use bbox center to align the robot with the sign.

Acceptance criteria:

If sign is left, robot rotates left.
If sign is right, robot rotates right.
If sign is centered, robot stops rotating and approaches.
Phase 5: Approach Controller

Use bbox area as rough distance proxy.

Acceptance criteria:

Robot approaches the sign slowly.
Robot stops when sign is large enough.
Robot does not crash into the sign.
Phase 6: Arrow Action Execution

Use stable arrow direction to execute relative action.

Acceptance criteria:

LEFT causes approximately 90-degree left turn.
RIGHT causes approximately 90-degree right turn.
STRAIGHT causes robot to continue forward.
Cooldown prevents repeated action on the same sign.
Phase 7: Integration Test

Run the complete state machine:

SEARCH_SIGN
ALIGN_TO_SIGN
APPROACH_SIGN
READ_ARROW
EXECUTE_TURN
COOLDOWN
SEARCH_SIGN

Acceptance criteria:

Robot can detect one sign, align to it, read it, turn correctly, and move on.
Failure Handling
Case 1: Sign Detected but Homography Fails

Fallback:

Use bbox crop directly.
Classify arrow without rectification.
Lower confidence.
Case 2: Direction is UNKNOWN

Fallback:

Stop and keep reading for up to read_timeout_sec.
If still unknown, rotate slightly and retry.
Case 3: Multiple Signs Detected

Fallback:

Choose the candidate with highest score.
Prefer larger and more centered candidate.
Case 4: Same Sign Triggers Repeatedly

Fallback:

Use cooldown.
Ignore detections for cooldown_sec.
Optionally wait until sign leaves frame.
Case 5: Lighting Changes Break Threshold

Fallback:

Switch from Otsu threshold to HSV V-channel threshold.
Expose threshold values in YAML.
Important Design Notes
Homography is used only after detecting the sign.
Homography is applied only to the planar sign ROI, not the full camera image.
Direction classification should happen in the warped sign image.
Green bbox should be drawn on the original camera image.
Robot does not need global localization for this task.
The robot only needs local reactive navigation.
Arrow direction is a relative navigation instruction, not global heading.
Bbox center controls alignment.
Bbox area estimates closeness.
Odometry or IMU should be used for accurate turns.
Final Expected Behavior

The final system should behave like this:

Robot searches for arrow sign.
When sign appears, robot centers it in the camera view.
Robot moves closer until the sign is readable.
Robot stops.
Detector rectifies the sign using homography.
Detector classifies the arrow direction.
Robot executes the relative command.
Robot enters cooldown.
Robot continues searching for the next sign.

This gives a simple, explainable, traditional OpenCV solution that is robust enough for the project demo and easy to defend during presentation.