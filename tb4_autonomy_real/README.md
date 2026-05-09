# tb4_autonomy_real

Real-robot autonomy package for the ENPM673 TurtleBot4 project.

This package intentionally mirrors the structure of `tb4_autonomy`, but uses
the real-scene arrow detector path tuned for OAK-D preview recordings. The
simulation package `tb4_autonomy` is kept as the original `jerry/main` version.

## Current Split

- `tb4_autonomy`: simulation/original package.
- `tb4_autonomy_real`: real-robot package.
- Controller and state-machine concepts are currently the same.
- The active difference is the arrow detector and its real-scene parameters.

## Launch

After building and sourcing the workspace:

```bash
ros2 launch tb4_autonomy_real autonomy.launch.py \
  dry_run:=false \
  use_rviz:=true \
  image_topic:=/tb4_1/oakd/rgb/preview/image_raw \
  camera_info_topic:=/tb4_1/oakd/rgb/preview/camera_info \
  odom_topic:=/tb4_1/odom \
  scan_topic:=/tb4_1/scan \
  cmd_vel_topic:=/tb4_1/cmd_vel \
  cmd_vel_stamped:=true \
  annotated_image_topic:=/tb4_1/debug/annotated_image
```

The root lab helpers `scripts/start_real_tb4_autonomy.sh` and
`scripts/run_real_tb4_autonomy.sh` now launch this package.

## Key Files

- `tb4_autonomy_real/detectors/arrow_detector.py`: real-scene arrow detector.
- `config/arrow_detection_real_scene.yaml`: default runtime/offline real-scene tuning.
- `config/autonomy.yaml`: runtime real-robot parameters.
- `launch/autonomy.launch.py`: real package launch entrypoint.
