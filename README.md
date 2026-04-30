# ENPM673 Turtlebot Perception Challenge

> **📖 Autonomy package documentation:** See [`tb4_autonomy/README.md`](tb4_autonomy/README.md) for the perception-to-control architecture, how to extend the pipeline, and all configurable parameters.

## How to install Webots?

Webots R2025a is required to run this simulation.

``` shell
# Add the Cyberbotics apt repository
wget -qO- https://cyberbotics.com/Cyberbotics.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://cyberbotics.com/debian/ binary-amd64/'
sudo apt update

# Install Webots
sudo apt install webots
```

Alternatively, download the `.deb` installer directly from the [Webots releases page](https://github.com/cyberbotics/webots/releases) and install it:

``` shell
# for R2025a
wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots_2025a_amd64.deb
sudo apt install ./webots_2025a_amd64.deb
```

Also install the ROS2 Webots driver:

``` shell
sudo apt install ros-humble-webots-ros2
```

Verify the installation:

``` shell
/usr/local/webots/webots --version
```

## How to build / install the ROS2 package?

``` shell
# first checkout the git repo
git clone https://github.com/adil275/ENPM673-Final-Project-Simulation.git
cd ENPM673-Final-Project-Simulation/
# build and install the package
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Once built, how to start the WeBots simulation?

``` shell
# Terminal 1 — launch the Webots scene
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tb4_sim tb4_launcher.py
```

## How to run the autonomy controller?

``` shell
# Terminal 2 — launch the autonomy node (after the simulation is up)
source /opt/ros/humble/setup.bash
source install/setup.bash

# dry_run:=true  → monitor only, no velocity commands (default)
# dry_run:=false → active control
ros2 launch tb4_autonomy autonomy.launch.py dry_run:=false
```

## How to run on the real TurtleBot4?

Connect your computer to the robot Wi-Fi first:

``` text
SSID: RAL_robots
Password: RAL2022robots
```

Build the workspace once before going to the robot:

``` shell
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Use the helper script to select the robot, set the ROS discovery environment, check topics, reset odom, and launch our controller with the correct namespace.

For `tb4_2`:

``` shell
cd ~/jerry_workspace/ENPM673_final_project
scripts/run_real_tb4_autonomy.sh tb4_2
```

For `tb4_5`:

``` shell
cd ~/jerry_workspace/ENPM673_final_project
scripts/run_real_tb4_autonomy.sh tb4_5
```

The script sets these automatically:

``` shell
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_SUPER_CLIENT=True
export ROS_DISCOVERY_SERVER="..."
```

It also:

- pings the selected robot (`tb4_2`: `192.168.50.62`, `tb4_5`: `192.168.50.65`)
- restarts the ROS daemon
- checks `/tb4_X/odom`, `/tb4_X/cmd_vel`, `/tb4_X/scan`, and camera topics
- calls `/tb4_X/reset_pose` to reset odom
- launches `tb4_autonomy` with `/tb4_X/...` topics
- publishes stamped `/cmd_vel` when the robot expects `TwistStamped`
- shows clear error messages when discovery, topics, or reset odom fail

Recommended first run is a health check without launching autonomy:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 --launch=false
```

Safe perception-only run:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=true
```

Active control:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=false
```

If the camera topic is not auto-detected, pass it manually:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 \
  --image-topic=/tb4_2/oakd/rgb/preview/image_raw \
  --camera-info-topic=/tb4_2/oakd/rgb/preview/camera_info
```

If odom reset is unavailable but the rest of the robot is visible:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 --reset-odom=false
```

RViz is enabled by default. It displays the same `/debug/annotated_image` pipeline overlay used in simulation, including arrow tracking, UMD logo detection, and control state.

## How to stop the WeBots simulation?

Instead of closing the WeBots window, just hit control-c from the console to send an "interrupt signal" (SIGINT) to the entire chain of processes.

## How to bring up the camera image?

One way is to use `rqt`'s image viewer to display the `/camera/image_raw` topic:

``` shell
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

## How to manually drive the Turtlebot?

We can use the `teleop_twist_keyboard` program to write angular and linear speeds to the `/cmd_vel` topic.

``` shell
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

- Press `x` and `c` to reduce linear and angular speeds.
- To move around, use the keys below:

```
Moving around:
   u    i    o
   j    k    l
   m    ,    .
```

![Simulation screenshot](assets/sim_env.png)
