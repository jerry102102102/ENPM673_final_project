#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_real_tb4_autonomy.sh tb4_1 [options]

Direct launcher for lab use after ROS discovery is already working.
It does not ping, restart ros2 daemon, reset odom, or wait for topics.

Options:
  --dry-run=true|false       Publish zero cmd_vel while running perception. Default: false
  --use-rviz=true|false      Launch RViz with debug image. Default: true
  --image-topic=/topic       Camera image topic. Default: /<robot>/oakd/rgb/preview/image_raw
  --image-is-compressed=true|false
                             Whether image topic is sensor_msgs/CompressedImage. Default: false
  --camera-info-topic=/topic Camera info topic. Default: /<robot>/oakd/rgb/preview/camera_info
  --odom-topic=/topic        Odom topic. Default: /<robot>/odom
  --cmd-vel-topic=/topic     Cmd vel topic. Default: /<robot>/cmd_vel
  --scan-topic=/topic        Scan topic. Default: /<robot>/scan

Example:
  scripts/start_real_tb4_autonomy.sh tb4_1
EOF
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

bool_value() {
  case "$1" in
    true|True|TRUE|1|yes|YES|y) echo true ;;
    false|False|FALSE|0|no|NO|n) echo false ;;
    *) fail "Expected boolean true/false, got '$1'" ;;
  esac
}

source_ros_setup() {
  local setup_file="$1"
  [[ -f "$setup_file" ]] || fail "Missing $setup_file"
  set +u
  # shellcheck source=/dev/null
  source "$setup_file"
  set -u
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

[[ $# -ge 1 ]] || { usage; exit 2; }
robot="$1"
shift

case "$robot" in
  tb4_1|tb4_2|tb4_4|tb4_5) ;;
  *) fail "Unknown robot '$robot'. Expected tb4_1, tb4_2, tb4_4, or tb4_5." ;;
esac

ns="/$robot"
dry_run=false
use_rviz=true
image_topic="$ns/oakd/rgb/preview/image_raw"
image_is_compressed=false
image_is_compressed_override=""
camera_info_topic="$ns/oakd/rgb/preview/camera_info"
odom_topic="$ns/odom"
cmd_vel_topic="$ns/cmd_vel"
scan_topic="$ns/scan"
debug_image_topic="$ns/debug/annotated_image"
state_topic="$ns/autonomy/state"
perf_topic="$ns/autonomy/perf"

for arg in "$@"; do
  case "$arg" in
    --dry-run=*) dry_run="$(bool_value "${arg#*=}")" ;;
    --use-rviz=*) use_rviz="$(bool_value "${arg#*=}")" ;;
    --image-topic=*) image_topic="${arg#*=}" ;;
    --image-is-compressed=*) image_is_compressed_override="$(bool_value "${arg#*=}")" ;;
    --camera-info-topic=*) camera_info_topic="${arg#*=}" ;;
    --odom-topic=*) odom_topic="${arg#*=}" ;;
    --cmd-vel-topic=*) cmd_vel_topic="${arg#*=}" ;;
    --scan-topic=*) scan_topic="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option '$arg'" ;;
  esac
done

if [[ -n "$image_is_compressed_override" ]]; then
  image_is_compressed="$image_is_compressed_override"
elif [[ "$image_topic" == */compressed ]]; then
  image_is_compressed=true
else
  image_is_compressed=false
fi

cd "$repo_root"

source_ros_setup /opt/ros/humble/setup.bash
source_ros_setup install/setup.bash

cmd_type="$(ros2 topic type "$cmd_vel_topic" 2>/dev/null || true)"
cmd_vel_stamped=true
if [[ "$cmd_type" == "geometry_msgs/msg/Twist" ]]; then
  cmd_vel_stamped=false
elif [[ "$cmd_type" == "geometry_msgs/msg/TwistStamped" ]]; then
  cmd_vel_stamped=true
elif [[ -n "$cmd_type" ]]; then
  echo "[WARN] Unknown cmd_vel type '$cmd_type'; defaulting to TwistStamped." >&2
fi

rviz_config="$repo_root/tb4_autonomy/config/rviz.rviz"
rviz_config_tmp=""
if [[ "$use_rviz" == true ]]; then
  rviz_config_tmp="$(mktemp --suffix=_${robot}_rviz.rviz)"
  trap 'rm -f "$rviz_config_tmp"' EXIT
  sed "s#Value: /debug/annotated_image#Value: $debug_image_topic#g" \
    "$rviz_config" > "$rviz_config_tmp"
  rviz_config="$rviz_config_tmp"
fi

echo "[INFO] Starting autonomy directly with:"
echo "[INFO]   image_topic=$image_topic"
echo "[INFO]   image_is_compressed=$image_is_compressed"
echo "[INFO]   camera_info_topic=$camera_info_topic"
echo "[INFO]   odom_topic=$odom_topic"
echo "[INFO]   scan_topic=$scan_topic"
echo "[INFO]   cmd_vel_topic=$cmd_vel_topic type=${cmd_type:-unknown} stamped=$cmd_vel_stamped"
echo "[INFO]   debug_image_topic=$debug_image_topic"

exec ros2 launch tb4_autonomy autonomy.launch.py \
  dry_run:="$dry_run" \
  use_rviz:="$use_rviz" \
  image_topic:="$image_topic" \
  image_is_compressed:="$image_is_compressed" \
  camera_info_topic:="$camera_info_topic" \
  odom_topic:="$odom_topic" \
  scan_topic:="$scan_topic" \
  cmd_vel_topic:="$cmd_vel_topic" \
  cmd_vel_stamped:="$cmd_vel_stamped" \
  annotated_image_topic:="$debug_image_topic" \
  rviz_config:="$rviz_config" \
  state_topic:="$state_topic" \
  perf_topic:="$perf_topic"
