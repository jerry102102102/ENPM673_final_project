#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_real_tb4_autonomy.sh tb4_1 [options]
  scripts/start_real_tb4_autonomy.sh tb4_2 [options]
  scripts/start_real_tb4_autonomy.sh tb4_4 [options]
  scripts/start_real_tb4_autonomy.sh tb4_5 [options]

Direct launcher for lab use after ROS discovery is already working.
It does not configure ROS_DISCOVERY_SERVER, ping, or restart ros2 daemon.
Set the RAL_robots Wi-Fi / ROS discovery environment yourself first.

Options:
  --dry-run=true|false          Publish zero cmd_vel while running perception. Default: false
  --use-rviz=true|false         Launch RViz with debug image. Default: true
  --auto-detect-topics=true|false
                                Use visible ROS topics when available. Default: true
  --reset-odom=true|false       Call /<robot>/reset_pose before launch. Default: false
  --image-topic=/topic          Camera image topic. Default: /<robot>/oakd/rgb/preview/image_raw
  --image-is-compressed=true|false
                                Whether image topic is sensor_msgs/CompressedImage. Default: auto
  --camera-info-topic=/topic    Camera info topic. Default: /<robot>/oakd/rgb/preview/camera_info
  --odom-topic=/topic           Odom topic. Default: /<robot>/odom
  --cmd-vel-topic=/topic        Cmd vel topic. Default: /<robot>/cmd_vel
  --cmd-vel-stamped=true|false  Publish TwistStamped instead of Twist. Default: auto, then true
  --scan-topic=/topic           Scan topic. Default: /<robot>/scan

Examples:
  scripts/start_real_tb4_autonomy.sh tb4_1
  scripts/start_real_tb4_autonomy.sh tb4_4 --reset-odom=true
  scripts/start_real_tb4_autonomy.sh tb4_2 --use-rviz=false
EOF
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
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

topic_list_file=""

refresh_topic_list() {
  [[ -n "$topic_list_file" ]] || return 1
  ros2 topic list > "$topic_list_file" 2>/dev/null
}

topic_exists() {
  local topic="$1"
  [[ -n "$topic_list_file" && -f "$topic_list_file" ]] || return 1
  grep -qx -- "$topic" "$topic_list_file"
}

first_existing_topic() {
  local candidate
  for candidate in "$@"; do
    if topic_exists "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

maybe_autodetect_topic() {
  local fallback="$1"
  shift
  if [[ "$auto_detect_topics" != true ]]; then
    echo "$fallback"
    return 0
  fi
  first_existing_topic "$@" || echo "$fallback"
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
auto_detect_topics=true
reset_odom=false
image_topic="$ns/oakd/rgb/preview/image_raw"
image_is_compressed=false
image_is_compressed_override=""
camera_info_topic="$ns/oakd/rgb/preview/camera_info"
odom_topic="$ns/odom"
cmd_vel_topic="$ns/cmd_vel"
cmd_vel_stamped_override=""
scan_topic="$ns/scan"
debug_image_topic="$ns/debug/annotated_image"
state_topic="$ns/autonomy/state"
perf_topic="$ns/autonomy/perf"

for arg in "$@"; do
  case "$arg" in
    --dry-run=*) dry_run="$(bool_value "${arg#*=}")" ;;
    --use-rviz=*) use_rviz="$(bool_value "${arg#*=}")" ;;
    --auto-detect-topics=*) auto_detect_topics="$(bool_value "${arg#*=}")" ;;
    --reset-odom=*) reset_odom="$(bool_value "${arg#*=}")" ;;
    --image-topic=*) image_topic="${arg#*=}" ;;
    --image-is-compressed=*) image_is_compressed_override="$(bool_value "${arg#*=}")" ;;
    --camera-info-topic=*) camera_info_topic="${arg#*=}" ;;
    --odom-topic=*) odom_topic="${arg#*=}" ;;
    --cmd-vel-topic=*) cmd_vel_topic="${arg#*=}" ;;
    --cmd-vel-stamped=*) cmd_vel_stamped_override="$(bool_value "${arg#*=}")" ;;
    --scan-topic=*) scan_topic="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option '$arg'" ;;
  esac
done

cd "$repo_root"

source_ros_setup /opt/ros/humble/setup.bash
source_ros_setup install/setup.bash

topic_list_file="$(mktemp)"
rviz_config_tmp=""
trap 'rm -f "$topic_list_file" "$rviz_config_tmp"' EXIT

if [[ "$auto_detect_topics" == true ]]; then
  if refresh_topic_list && [[ -s "$topic_list_file" ]]; then
    image_topic="$(maybe_autodetect_topic "$image_topic" \
      "$ns/oakd/rgb/preview/image_raw" \
      "$ns/oakd/rgb/preview/image_raw/compressed" \
      "$ns/oakd/rgb/image_raw" \
      "$ns/oakd/rgb/image_raw/compressed" \
      "$ns/oakd/rgb/preview/image_color" \
      "$ns/camera/image_raw/image_color" \
      "$ns/camera/image_raw" \
      "$ns/color/image")"

    odom_topic="$(maybe_autodetect_topic "$odom_topic" \
      "$ns/odom" \
      "$ns/wheel_odom" \
      "$ns/mobile_base/odom" \
      "/odom")"

    cmd_vel_topic="$(maybe_autodetect_topic "$cmd_vel_topic" \
      "$ns/cmd_vel" \
      "$ns/mobile_base/cmd_vel" \
      "$ns/diffdrive_controller/cmd_vel_unstamped" \
      "/cmd_vel")"

    scan_topic="$(maybe_autodetect_topic "$scan_topic" \
      "$ns/scan" \
      "$ns/rplidar/scan" \
      "/scan")"

    image_parent="$(dirname "$image_topic")"
    if [[ "$image_topic" == */compressed || "$image_topic" == */compressedDepth || "$image_topic" == */theora || "$image_topic" == */zstd ]]; then
      image_parent="$(dirname "$image_parent")"
    fi
    camera_info_topic="$(maybe_autodetect_topic "$camera_info_topic" \
      "$image_parent/camera_info" \
      "$ns/oakd/rgb/preview/camera_info" \
      "$ns/oakd/rgb/camera_info" \
      "$ns/camera/image_raw/camera_info" \
      "$ns/camera/camera_info")"
  else
    warn "Could not read ros2 topic list; using namespace defaults for $robot."
  fi
fi

if [[ -n "$image_is_compressed_override" ]]; then
  image_is_compressed="$image_is_compressed_override"
elif [[ "$image_topic" == */compressed ]]; then
  image_is_compressed=true
else
  image_is_compressed=false
fi

cmd_type="$(ros2 topic type "$cmd_vel_topic" 2>/dev/null || true)"
if [[ -n "$cmd_vel_stamped_override" ]]; then
  cmd_vel_stamped="$cmd_vel_stamped_override"
else
  cmd_vel_stamped=true
  if [[ "$cmd_type" == "geometry_msgs/msg/Twist" ]]; then
    cmd_vel_stamped=false
  elif [[ "$cmd_type" == "geometry_msgs/msg/TwistStamped" ]]; then
    cmd_vel_stamped=true
  elif [[ -n "$cmd_type" ]]; then
    warn "Unknown cmd_vel type '$cmd_type'; defaulting to TwistStamped."
  else
    warn "Could not read cmd_vel type for $cmd_vel_topic; defaulting to TwistStamped."
  fi
fi

if [[ "$reset_odom" == true ]]; then
  reset_service="$ns/reset_pose"
  info "Resetting odom via $reset_service"
  ros2 service call "$reset_service" irobot_create_msgs/srv/ResetPose "{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}" >/tmp/"${robot}_reset_pose.log" 2>&1 \
    || fail "reset_pose call failed. See /tmp/${robot}_reset_pose.log"
fi

rviz_config="$repo_root/tb4_autonomy/config/rviz.rviz"
if [[ "$use_rviz" == true ]]; then
  rviz_config_tmp="$(mktemp --suffix=_${robot}_rviz.rviz)"
  sed "s#Value: /debug/annotated_image#Value: $debug_image_topic#g" \
    "$rviz_config" > "$rviz_config_tmp"
  rviz_config="$rviz_config_tmp"
fi

info "Starting autonomy directly for $robot with:"
info "  image_topic=$image_topic"
info "  image_is_compressed=$image_is_compressed"
info "  camera_info_topic=$camera_info_topic"
info "  odom_topic=$odom_topic"
info "  scan_topic=$scan_topic"
info "  cmd_vel_topic=$cmd_vel_topic type=${cmd_type:-unknown} stamped=$cmd_vel_stamped"
info "  debug_image_topic=$debug_image_topic"

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
