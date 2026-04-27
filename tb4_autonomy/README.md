# tb4_autonomy — TurtleBot4 Perception-to-Control Framework

This ROS 2 package implements a **vision-based autonomous navigation** pipeline for the TurtleBot4 in a Webots simulation environment.  
The robot follows a course of printed arrow signs: it detects arrows with classical OpenCV, classifies their direction, and executes the corresponding turn — all without any deep-learning model.

---

## Architecture Overview

```
Camera Image ─► Detectors ─► State Machine ─► Motion Controller ─► /cmd_vel
                  │                │                  │
           ArrowDetector     AutonomyState        Twist msgs
           LogoDetector      (FSM logic)        (P-controller)
           MovingBallDetector
           HorizonDetector
```

The entire pipeline runs inside a single ROS 2 node (`vision_controller_node`).  
Each camera frame triggers: **detect → decide → act**.

### Key Design Principle

> **Perception and control are fully decoupled through data types.**  
> Detectors only produce `DetectionResults`; the state machine only reads those results; the motion controller only receives turn commands.  
> You can swap out any layer without touching the others.

---

## File Map

| File | Role | When to modify |
|------|------|----------------|
| `vision_controller_node.py` | ROS 2 node — wires everything together, subscribes to topics, publishes `/cmd_vel` | Adding new ROS topics, changing the main loop |
| `data_types.py` | Shared dataclasses (`ArrowDetection`, `DetectionResults`, `StateMachineOutput`, etc.) | Adding new detection types or fields |
| `state_machine.py` | Finite state machine (FSM) — decides *what to do* based on detections | Changing navigation logic, adding new states |
| `motion_controller.py` | Produces `Twist` messages — cruising, tracking, P-controller turning | Tuning speeds, changing turn behavior |
| `detectors/arrow_detector.py` | OpenCV arrow detection + direction classification + continuous angle | Improving arrow recognition |
| `detectors/logo_detector.py` | UMD logo detector (triggers stop) | Logo recognition improvements |
| `detectors/moving_ball_detector.py` | Moving ball detector (triggers emergency stop) | Ball detection improvements |
| `detectors/horizon_detector.py` | Horizon line detector | Horizon estimation |
| `config/autonomy.yaml` | All tunable parameters (speeds, thresholds, detector configs) | Tuning without code changes |
| `launch/autonomy.launch.py` | Launch file for the autonomy node | Changing launch arguments |

---

## State Machine Flow

```
SEARCH_SIGN ──(arrow detected)──► ALIGN_TO_SIGN ──(centered)──► APPROACH_SIGN
                                                                      │
                                                              (close enough)
                                                                      ▼
            ARROW_COOLDOWN ◄──(turn done)── EXECUTE_TURN ◄── READ_ARROW
                  │                                          (stable direction)
                  ▼
            SEARCH_SIGN  (loop continues)
```

Special interrupts:
- **Logo detected** → `LOGO_STOP` (pause 3s, then resume)
- **Moving ball** → `BALL_STOP` (freeze until clear)

---

## How Perception Connects to Control

### 1. Detectors produce structured results

Each detector returns a typed dataclass (e.g., `ArrowDetection`):

```python
@dataclass
class ArrowDetection:
    box: Box2D                    # bounding box in image coordinates
    direction: str                # 'left', 'right', 'straight', 'back', 'unknown'
    confidence: float             # 0.0 – 1.0
    center_error_px: float        # horizontal offset from image center (for tracking)
    is_stable: bool               # True if direction is consistent over N frames
    arrow_angle_rad: float        # continuous angle from tip-finding algorithm
    area_ratio: float             # bbox area / image area (proxy for distance)
    ...
```

### 2. State machine reads detections and decides action

The `StateMachine.update()` method takes `DetectionResults` and returns a `StateMachineOutput`:

```python
@dataclass(frozen=True)
class StateMachineOutput:
    state: AutonomyState          # current FSM state
    entered_state: bool           # True on state transitions
    turn_direction: str | None    # 'left', 'right', 'back' when executing a turn
    turn_angle_rad: float | None  # continuous angle for the turn (from arrow detector)
```

### 3. Motion controller executes commands

`vision_controller_node.py` maps each state to a motion command:

| State | Motion |
|-------|--------|
| `SEARCH_SIGN` | Slow forward cruise |
| `ALIGN_TO_SIGN` | Rotate to center the arrow in frame |
| `APPROACH_SIGN` | Drive forward while tracking the arrow |
| `READ_ARROW` | Stop and wait for stable direction |
| `EXECUTE_TURN` | P-controller yaw turn to `turn_angle_rad` |
| `ARROW_COOLDOWN` | Drive forward briefly to clear the sign |
| `LOGO_STOP` | Full stop for configured duration |
| `BALL_STOP` | Full stop until ball clears |

### 4. The turn interface

```python
# In motion_controller.py
motion.start_turn(direction='left', current_yaw=1.2, angle_rad=1.57)

# Each control cycle:
twist_msg, is_complete = motion.update_turn(current_yaw)
```

- `direction`: `'left'` / `'right'` / `'back'`
- `current_yaw`: current robot heading from `/odom`
- `angle_rad` (optional): if provided, uses this exact angle; otherwise falls back to config defaults (π/2 for left/right, π for back)
- Returns: `(Twist, bool)` — the velocity command and whether the turn is done

---

## Adding a New Detector

1. Create `detectors/my_detector.py` with a `detect(frame, context) -> MyDetection` method
2. Add `MyDetection` dataclass to `data_types.py`
3. Add the field to `DetectionResults`
4. Instantiate the detector in `vision_controller_node.py` and call it in the detection loop
5. Handle the new detection in `state_machine.py`

---

## Adding a New Behavior / State

1. Add the state to `AutonomyState` enum in `data_types.py`
2. Add transition logic in `state_machine.py`
3. Add the corresponding motion command mapping in `vision_controller_node.py`

---

## Configuration

All parameters are in `config/autonomy.yaml` and can be overridden at launch:

```bash
ros2 launch tb4_autonomy autonomy.launch.py dry_run:=false
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dry_run` | `true` | When true, publishes zero velocity (safe monitoring mode) |
| `cruise_linear_x` | `0.12` | Forward speed during cruise (m/s) |
| `turn_kp` | `1.4` | Proportional gain for yaw P-controller |
| `turn_angle_rad` | `1.5708` | Default left/right turn angle (π/2) |
| `turn_tolerance_rad` | `0.05` | Yaw error threshold to consider turn complete |
| `target_bbox_area_ratio` | `0.08` | Arrow must fill this fraction of frame before reading |
| `arrow_min_stable_count` | `4` | Number of consistent frames to confirm direction |

See `autonomy.yaml` for the full list including all arrow detector thresholds.

---

## Running Tests

```bash
cd /path/to/ENPM673-Final-Project-Simulation
source /opt/ros/humble/setup.bash
source install/setup.bash
python -m pytest tb4_autonomy/test/ -v
```
