# ENPM673 TurtleBot4 Perception Challenge

> Autonomy package documentation: see [`tb4_autonomy/README.md`](tb4_autonomy/README.md) for the current observation topics, control outputs, controller behavior, real TurtleBot4 scripts, and tuning parameters.

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

Also install the ROS 2 Webots driver for your ROS distro:

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
git clone https://github.com/jerry102102102/ENPM673_final_project.git
cd ENPM673_final_project/
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

The controller observes camera image, camera info, odom, and scan topics. In simulation the defaults are:

``` text
/camera/image_raw/image_color
/camera/image_raw/camera_info
/odom
/scan
```

It publishes:

``` text
/cmd_vel
/debug/annotated_image
/autonomy/state
/autonomy/perf
```

See [`tb4_autonomy/README.md`](tb4_autonomy/README.md) for the real robot namespaced topics.

## How to run on the real TurtleBot4?

Connect your computer to the robot Wi-Fi first:

``` text
SSID: RAL_robots
Password: RAL2022robots
```

Build the workspace once before going to the robot:

``` shell
cd ~/ENPM673_final_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

There are two real robot scripts:

- `scripts/start_real_tb4_autonomy.sh`: direct launcher. Use this when you already configured Wi-Fi and ROS discovery yourself. It only maps topics for the selected robot and starts the controller.
- `scripts/run_real_tb4_autonomy.sh`: full helper. It sets discovery variables, pings the robot, restarts the ROS daemon, checks topics, optionally resets odom, and launches the controller.

For test day, if you set the discovery server yourself, use the direct launcher:

``` shell
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_SUPER_CLIENT=True

# choose one robot
export ROS_DISCOVERY_SERVER="192.168.50.61:11811;"       # tb4_1
export ROS_DISCOVERY_SERVER=";192.168.50.62:11811;"      # tb4_2
export ROS_DISCOVERY_SERVER=";;;192.168.50.64:11811;"    # tb4_4
export ROS_DISCOVERY_SERVER=";;;;192.168.50.65:11811;"   # tb4_5
```

Then launch:

``` shell
scripts/start_real_tb4_autonomy.sh tb4_1
scripts/start_real_tb4_autonomy.sh tb4_2
scripts/start_real_tb4_autonomy.sh tb4_4
scripts/start_real_tb4_autonomy.sh tb4_5
```

For a perception-only check:

``` shell
scripts/start_real_tb4_autonomy.sh tb4_2 --dry-run=true
```

To reset odom before launch:

``` shell
scripts/start_real_tb4_autonomy.sh tb4_2 --reset-odom=true
```

`start_real_tb4_autonomy.sh` maps topics under the selected namespace. For `tb4_2`, it uses:

``` text
/tb4_2/oakd/rgb/preview/image_raw
/tb4_2/oakd/rgb/preview/camera_info
/tb4_2/odom
/tb4_2/scan
/tb4_2/cmd_vel
/tb4_2/debug/annotated_image
```

If you want the script to handle discovery and health checks:

``` shell
scripts/run_real_tb4_autonomy.sh tb4_2 --launch=false
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=true
scripts/run_real_tb4_autonomy.sh tb4_2 --dry-run=false
```

If the camera topic is not auto-detected, pass it manually to either script:

``` shell
scripts/start_real_tb4_autonomy.sh tb4_2 \
  --image-topic=/tb4_2/oakd/rgb/preview/image_raw \
  --camera-info-topic=/tb4_2/oakd/rgb/preview/camera_info
```

RViz is enabled by default. It displays the namespaced debug image, including arrow tracking, UMD logo detection, moving/static ball detection, horizon debug line, and controller state.

## How to stop the WeBots simulation?

Instead of closing the WeBots window, just hit control-c from the console to send an "interrupt signal" (SIGINT) to the entire chain of processes.

## How to bring up the camera image?

One way is to use `rqt`'s image viewer to display the simulation camera topic:

``` shell
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw/image_color
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
