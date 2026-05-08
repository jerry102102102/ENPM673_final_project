# tb4_autonomy

Vision-driven autonomy for the ENPM673 TurtleBot4 final project.

The current project no longer uses the old hard-turn arrow FSM as the main controller. The active controller is `ArrowSmoothArcController`: it drives forward with small, smooth steering corrections from the visible arrow paper, while safety detectors can override motion.

## Current Architecture

```text
ROS image/odom topics
        |
        v
vision_controller_node
        |
        +-- ArrowDetector          -> ArrowSmoothArcController
        +-- LogoDetector           -> one-time 3 second stop
        +-- MovingBallDetector     -> BALL_STOP while active
        +-- StaticObstacleDetector -> BALL_STOP while active
        +-- HorizonDetector        -> debug / ROI reference
        |
        +-- publishes cmd_vel
        +-- publishes debug image, state, and perf text
```

Everything runs inside `tb4_autonomy/tb4_autonomy/vision_controller_node.py`.

## Observations

The autonomy node observes these ROS topics:

| Input | Default In Simulation | Real Robot Default For `tb4_X` | Use |
|-------|------------------------|--------------------------------|-----|
| Image | `/camera/image_raw/image_color` | `/tb4_X/oakd/rgb/preview/image_raw` | Main camera frame for arrow, logo, ball, and horizon detectors |
| Camera info | `/camera/image_raw/camera_info` | `/tb4_X/oakd/rgb/preview/camera_info` | Camera metadata, passed through detector context |
| Odom | `/odom` | `/tb4_X/odom` | Robot yaw and XY position for smooth heading latch / finishing behavior |
| Scan | `/scan` | `/tb4_X/scan` | Available in context for safety/debug extensions |

The real robot script can auto-detect common TurtleBot4 topic variants. It also supports compressed image topics through:

```bash
--image-topic=/tb4_X/oakd/rgb/preview/image_raw/compressed
--image-is-compressed=true
```

## Control Outputs

The node publishes:

| Output | Default In Simulation | Real Robot Default For `tb4_X` | Use |
|--------|------------------------|--------------------------------|-----|
| Command velocity | `/cmd_vel` | `/tb4_X/cmd_vel` | Robot motion command |
| Debug image | `/debug/annotated_image` | `/tb4_X/debug/annotated_image` | RViz overlay with detections and controller state |
| State | `/autonomy/state` | `/tb4_X/autonomy/state` | Current high-level state |
| Perf | `/autonomy/perf` | `/tb4_X/autonomy/perf` | Detector timing summary |

`cmd_vel` can be either `geometry_msgs/msg/Twist` or `geometry_msgs/msg/TwistStamped`. The launch argument is:

```bash
cmd_vel_stamped:=true|false
```

The real robot direct launcher auto-detects the command topic type when possible. If it cannot read the type, it defaults to stamped commands because the lab TurtleBot4 teleop example uses `stamped:=true`.

## Runtime Behavior

Normal driving is handled by `ArrowSmoothArcController` in `arrow_smooth_arc_controller.py`.

The controller states are:

| State | Behavior |
|-------|----------|
| `WAIT_FOR_TARGET` | Drive slowly forward until a valid arrow paper is acquired |
| `SMOOTH_ARC_TRACK` | Follow the visible paper center and confidence-weighted continuous arrow heading |
| `PASS_TO_NEXT` | Move forward briefly after finishing the current target |
| `LOGO_STOP` | Stop once for UMD logo detection |
| `BALL_STOP` | Stop while moving/static ball detector is active |

The smooth arc control law uses:

```text
angular_target =
    center_term
  + heading_weight * heading_term
  + previous_heading_pull_term
```

Important terms:

| Term | Meaning |
|------|---------|
| `center_term` | Keeps the current paper from drifting too far left/right in the camera |
| `heading_term` | Uses the continuous arrow heading only when heading confidence is high enough |
| `previous_heading_pull_term` | Keeps the previous high-confidence arrow direction influencing the robot while the next arrow appears |
| `latched_world_yaw` | Odom-frame yaw target created from high-confidence arrow heading |

Safety priority in `vision_controller_node.py`:

1. Moving ball or static-flow ball -> `BALL_STOP`
2. UMD logo hold / confirmed logo -> `LOGO_STOP`
3. Arrow smooth arc controller -> normal motion

## Key Files

| File | Role |
|------|------|
| `tb4_autonomy/tb4_autonomy/vision_controller_node.py` | ROS node wiring observations, detectors, controller, and publishers |
| `tb4_autonomy/tb4_autonomy/arrow_smooth_arc_controller.py` | Current driving controller |
| `tb4_autonomy/tb4_autonomy/detectors/arrow_detector.py` | Paper/arrow detection and continuous heading estimate |
| `tb4_autonomy/tb4_autonomy/detectors/logo_detector.py` | UMD logo stop detector |
| `tb4_autonomy/tb4_autonomy/detectors/moving_ball_detector.py` | Color/motion ball detector |
| `tb4_autonomy/tb4_autonomy/detectors/static_ball_detector.py` | Optical-flow round obstacle detector |
| `tb4_autonomy/tb4_autonomy/detectors/horizon_detector.py` | Horizon/debug reference detector |
| `tb4_autonomy/tb4_autonomy/utils/image_tools.py` | Annotated overlay drawing |
| `tb4_autonomy/config/autonomy.yaml` | Main tuning file |
| `tb4_autonomy/launch/autonomy.launch.py` | Launch file for the controller node and RViz |
| `scripts/start_real_tb4_autonomy.sh` | Direct real robot launcher when ROS discovery is already configured |
| `scripts/run_real_tb4_autonomy.sh` | Full real robot helper with discovery setup, ping, topic checks, and optional odom reset |

## Simulation Run

Build:

```bash
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Start Webots in one terminal:

```bash
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tb4_sim tb4_launcher.py
```

Start the controller in another terminal:

```bash
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tb4_autonomy autonomy.launch.py dry_run:=false use_rviz:=true
```

Useful launch arguments:

| Argument | Use |
|----------|-----|
| `dry_run:=true` | Run perception and RViz, but publish zero velocity |
| `dry_run:=false` | Active control |
| `use_rviz:=true` | Open RViz with the debug image layout |
| `image_topic:=...` | Override camera image topic |
| `image_is_compressed:=true` | Subscribe as `sensor_msgs/msg/CompressedImage` |
| `cmd_vel_stamped:=true` | Publish `TwistStamped` instead of `Twist` |

## Real TurtleBot4 Direct Launch

Use this when you have already connected to the lab Wi-Fi and configured ROS discovery yourself.

Lab network:

```text
SSID: RAL_robots
Password: RAL2022robots
```

Common ROS environment:

```bash
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_SUPER_CLIENT=True
```

Set the discovery server for the robot you are using:

```bash
# tb4_1
export ROS_DISCOVERY_SERVER="192.168.50.61:11811;"

# tb4_2
export ROS_DISCOVERY_SERVER=";192.168.50.62:11811;"

# tb4_4
export ROS_DISCOVERY_SERVER=";;;192.168.50.64:11811;"

# tb4_5
export ROS_DISCOVERY_SERVER=";;;;192.168.50.65:11811;"
```

Then run the direct launcher:

```bash
cd ~/ENPM673_final_project
source /opt/ros/humble/setup.bash
source install/setup.bash

scripts/start_real_tb4_autonomy.sh tb4_1
scripts/start_real_tb4_autonomy.sh tb4_2
scripts/start_real_tb4_autonomy.sh tb4_4
scripts/start_real_tb4_autonomy.sh tb4_5
```

The direct launcher does not ping or restart the ROS daemon. It only chooses the selected robot namespace and launches the controller.

Defaults for `tb4_2`:

```text
image:       /tb4_2/oakd/rgb/preview/image_raw
camera_info: /tb4_2/oakd/rgb/preview/camera_info
odom:        /tb4_2/odom
scan:        /tb4_2/scan
cmd_vel:     /tb4_2/cmd_vel
debug image: /tb4_2/debug/annotated_image
state:       /tb4_2/autonomy/state
perf:        /tb4_2/autonomy/perf
```

Useful options:

| Option | Use |
|--------|-----|
| `--dry-run=true` | Perception/RViz only; no motion |
| `--use-rviz=false` | Run without RViz |
| `--reset-odom=true` | Call `/<robot>/reset_pose` before launch |
| `--auto-detect-topics=false` | Use exact namespaced defaults |
| `--cmd-vel-stamped=true` | Force stamped velocity output |
| `--image-topic=/topic` | Override camera topic |
| `--image-is-compressed=true` | Use compressed image subscription |

Examples:

```bash
scripts/start_real_tb4_autonomy.sh tb4_4 --dry-run=true
scripts/start_real_tb4_autonomy.sh tb4_4 --reset-odom=true
scripts/start_real_tb4_autonomy.sh tb4_2 --use-rviz=false
```

## Full Real Robot Helper

`scripts/run_real_tb4_autonomy.sh` is the heavier helper. It sets the ROS discovery server for `tb4_1`, `tb4_2`, `tb4_4`, or `tb4_5`, pings the robot, restarts the ROS daemon, checks topics, optionally resets odom, and then launches the same autonomy node.

Use it when you want the script to handle the connection setup:

```bash
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=true
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=false
scripts/run_real_tb4_autonomy.sh tb4_2 --launch=false
```

## RViz Overlay

RViz displays the annotated camera image. In simulation this is:

```text
/debug/annotated_image
```

On the real robot direct launcher, it is namespaced:

```text
/<robot>/debug/annotated_image
```

The overlay shows:

- arrow paper corners and continuous heading line
- current controller state and control mode
- center error, heading weight, previous-heading pull, and cmd velocity
- UMD logo detection and stop state
- moving ball / static-flow ball detection
- horizon reference line

## Tuning

Most parameters live in `tb4_autonomy/config/autonomy.yaml`.

Common controller tuning values:

| Parameter | Meaning |
|-----------|---------|
| `arrow_smooth_arc_controller.heading_scale` | Scales current arrow heading correction |
| `arrow_smooth_arc_controller.kp_heading` | Gain for current arrow heading term |
| `arrow_smooth_arc_controller.kp_center` | Gain for paper center correction |
| `arrow_smooth_arc_controller.max_angular_z` | Overall angular velocity cap |
| `arrow_smooth_arc_controller.previous_heading_pull_gain` | Strength of previous arrow direction influence |
| `arrow_smooth_arc_controller.previous_heading_pull_decay_sec` | How long previous arrow influence persists |
| `arrow_smooth_arc_controller.previous_heading_pull_max_angular_z` | Angular cap for previous heading pull |

## Running Tests

```bash
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/humble/setup.bash
source install/setup.bash
python -m pytest tb4_autonomy/test/ -v
```
