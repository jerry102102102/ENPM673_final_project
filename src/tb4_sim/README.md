# tb4_sim

Webots simulation launch package for the TurtleBot4 project.

## What This Package Starts

`tb4_sim` launches the Webots world and the simulated TurtleBot4 interface used by `tb4_autonomy`.

The autonomy controller expects these simulation topics:

```text
/camera/image_raw/image_color
/camera/image_raw/camera_info
/odom
/scan
/cmd_vel
```

## Launch

Build and source the workspace:

```bash
cd ~/jerry_workspace/ENPM673_final_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Start the Webots scene:

```bash
ros2 launch tb4_sim tb4_launcher.py
```

Then start the controller in another terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tb4_autonomy autonomy.launch.py dry_run:=false use_rviz:=true
```

## Notes

- Keep the TA-provided world assets in `src/worlds` unchanged unless the team explicitly needs a test-only mock world.
- The real robot launch path is documented in `tb4_autonomy/README.md`.
