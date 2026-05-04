#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_real_tb4_autonomy.sh tb4_1 [options]
  scripts/run_real_tb4_autonomy.sh tb4_2 [options]
  scripts/run_real_tb4_autonomy.sh tb4_4 [options]
  scripts/run_real_tb4_autonomy.sh tb4_5 [options]

Options:
  --dry-run=true|false          Publish zero cmd_vel while running perception. Default: false
  --use-rviz=true|false         Launch RViz with /debug/annotated_image. Default: true
  --reset-odom=true|false       Call /<robot>/reset_pose before launch. Default: true
  --launch=true|false           Start tb4_autonomy after health checks. Default: true
  --image-topic=/topic          Override auto-detected camera image topic.
  --image-is-compressed=true|false
                                  Override whether image topic is sensor_msgs/CompressedImage.
                                  Default: auto-detect from image topic suffix.
  --camera-info-topic=/topic    Override auto-detected camera info topic.
  --odom-topic=/topic           Override auto-detected odom topic.
  --cmd-vel-topic=/topic        Override auto-detected cmd_vel topic.
  --scan-topic=/topic           Override auto-detected scan topic.

Examples:
  scripts/run_real_tb4_autonomy.sh tb4_2
  scripts/run_real_tb4_autonomy.sh tb4_5 --dry-run=true --launch=false
EOF
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

source_ros_setup() {
  local setup_file="$1"
  [[ -f "$setup_file" ]] || fail "Missing $setup_file"

  # ROS setup files can reference unset AMENT_* variables. Temporarily disable
  # nounset so `set -u` in this wrapper does not break Humble setup sourcing.
  set +u
  # shellcheck source=/dev/null
  source "$setup_file"
  set -u
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

topic_exists() {
  local topic="$1"
  grep -qx -- "$topic" "$TOPIC_LIST_FILE"
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

refresh_topic_list() {
  ros2 topic list > "$TOPIC_LIST_FILE" 2> "$TOPIC_ERROR_FILE"
}

print_visible_topics() {
  warn "Visible topics containing '$robot', 'odom', 'cmd_vel', 'scan', or camera names:"
  grep -En "($robot|odom|cmd_vel|scan|oakd|camera|image)" "$TOPIC_LIST_FILE" >&2 || sed -n '1,120p' "$TOPIC_LIST_FILE" >&2
}

wait_for_topic_list() {
  local tries=12
  local idx
  for idx in $(seq 1 "$tries"); do
    if ros2 topic list > "$TOPIC_LIST_FILE" 2> "$TOPIC_ERROR_FILE" && [[ -s "$TOPIC_LIST_FILE" ]]; then
      return 0
    fi
    sleep 1
  done
  warn "ros2 topic list did not return topics. Last error:"
  sed -n '1,80p' "$TOPIC_ERROR_FILE" >&2 || true
  return 1
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
  tb4_1)
    robot_ip="192.168.50.61"
    discovery_server="192.168.50.61:11811;"
    ;;
  tb4_2)
    robot_ip="192.168.50.62"
    discovery_server=";192.168.50.62:11811;"
    ;;
  tb4_4)
    robot_ip="192.168.50.64"
    discovery_server=";;;192.168.50.64:11811;"
    ;;
  tb4_5)
    robot_ip="192.168.50.65"
    discovery_server=";;;;192.168.50.65:11811;"
    ;;
  *)
    usage
    fail "Unknown robot '$robot'. Expected tb4_1, tb4_2, tb4_4, or tb4_5."
    ;;
esac

dry_run=false
use_rviz=true
reset_odom=true
do_launch=true
image_topic_override=""
image_is_compressed_override=""
camera_info_topic_override=""
odom_topic_override=""
cmd_vel_topic_override=""
scan_topic_override=""

for arg in "$@"; do
  case "$arg" in
    --dry-run=*) dry_run="$(bool_value "${arg#*=}")" ;;
    --use-rviz=*) use_rviz="$(bool_value "${arg#*=}")" ;;
    --reset-odom=*) reset_odom="$(bool_value "${arg#*=}")" ;;
    --launch=*) do_launch="$(bool_value "${arg#*=}")" ;;
    --image-topic=*) image_topic_override="${arg#*=}" ;;
    --image-is-compressed=*) image_is_compressed_override="$(bool_value "${arg#*=}")" ;;
    --camera-info-topic=*) camera_info_topic_override="${arg#*=}" ;;
    --odom-topic=*) odom_topic_override="${arg#*=}" ;;
    --cmd-vel-topic=*) cmd_vel_topic_override="${arg#*=}" ;;
    --scan-topic=*) scan_topic_override="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option '$arg'" ;;
  esac
done

cd "$repo_root"

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_SUPER_CLIENT=True
export ROS_DISCOVERY_SERVER="$discovery_server"

info "Robot: $robot"
info "IP: $robot_ip"
info "ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
info "Checking Wi-Fi reachability with ping..."
ping -c 2 -W 2 "$robot_ip" >/dev/null || fail "Cannot ping $robot_ip. Check Wi-Fi RAL_robots and robot power."

source_ros_setup /opt/ros/humble/setup.bash

[[ -f install/setup.bash ]] || fail "Missing install/setup.bash. Run: colcon build --symlink-install"
source_ros_setup install/setup.bash

info "Restarting ROS daemon..."
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null

TOPIC_LIST_FILE="$(mktemp)"
TOPIC_ERROR_FILE="$(mktemp)"
RVIZ_CONFIG_FILE=""
trap 'rm -f "$TOPIC_LIST_FILE" "$TOPIC_ERROR_FILE" "$RVIZ_CONFIG_FILE"' EXIT

wait_for_topic_list || fail "ROS discovery failed. Check ROS env vars, discovery server, and robot network."

ns="/$robot"
reset_service="$ns/reset_pose"
state_topic="$ns/autonomy/state"
perf_topic="$ns/autonomy/perf"
debug_image_topic="$ns/debug/annotated_image"

if [[ "$use_rviz" == true ]]; then
  default_rviz_config="$repo_root/tb4_autonomy/config/rviz.rviz"
  [[ -f "$default_rviz_config" ]] || fail "Missing RViz config $default_rviz_config"
  RVIZ_CONFIG_FILE="$(mktemp --suffix=_${robot}_rviz.rviz)"
  sed "s#Value: /debug/annotated_image#Value: $debug_image_topic#g" \
    "$default_rviz_config" > "$RVIZ_CONFIG_FILE"
fi

if [[ -n "$cmd_vel_topic_override" ]]; then
  cmd_vel_topic="$cmd_vel_topic_override"
else
  cmd_vel_topic="$(first_existing_topic \
    "$ns/cmd_vel" \
    "$ns/mobile_base/cmd_vel" \
    "$ns/diffdrive_controller/cmd_vel_unstamped" \
    "/cmd_vel" \
  )" || {
    print_visible_topics
    fail "Could not auto-detect cmd_vel topic for $robot. Pass --cmd-vel-topic=/topic."
  }
fi

if [[ -n "$odom_topic_override" ]]; then
  odom_topic="$odom_topic_override"
else
  odom_topic=""
  for _ in $(seq 1 10); do
    refresh_topic_list || true
    odom_topic="$(first_existing_topic \
      "$ns/odom" \
      "$ns/wheel_odom" \
      "$ns/mobile_base/odom" \
      "/odom" \
    )" && break
    sleep 1
  done
  if [[ -z "$odom_topic" ]]; then
    print_visible_topics
    fail "Could not auto-detect odom topic for $robot. Pass --odom-topic=/topic if the robot uses a different name."
  fi
fi

if [[ -n "$scan_topic_override" ]]; then
  scan_topic="$scan_topic_override"
else
  scan_topic="$(first_existing_topic \
    "$ns/scan" \
    "$ns/rplidar/scan" \
    "/scan" \
  )" || {
    scan_topic="$ns/scan"
    warn "Could not auto-detect scan topic. Continuing with $scan_topic because camera control may still run."
  }
fi

if [[ -n "$image_topic_override" ]]; then
  image_topic="$image_topic_override"
else
  image_topic="$(first_existing_topic \
    "$ns/oakd/rgb/preview/image_raw" \
    "$ns/oakd/rgb/image_raw" \
    "$ns/oakd/rgb/preview/image_raw/compressed" \
    "$ns/oakd/rgb/image_raw/compressed" \
    "$ns/oakd/rgb/preview/image_color" \
    "$ns/camera/image_raw/image_color" \
    "$ns/camera/image_raw" \
    "$ns/color/image" \
  )" || fail "Could not auto-detect camera image topic for $robot. Pass --image-topic=/topic."
fi

if [[ -n "$image_is_compressed_override" ]]; then
  image_is_compressed="$image_is_compressed_override"
elif [[ "$image_topic" == */compressed ]]; then
  image_is_compressed=true
else
  image_is_compressed=false
fi

if [[ -n "$camera_info_topic_override" ]]; then
  camera_info_topic="$camera_info_topic_override"
else
  image_parent="$(dirname "$image_topic")"
  if [[ "$image_topic" == */compressed || "$image_topic" == */compressedDepth || "$image_topic" == */theora || "$image_topic" == */zstd ]]; then
    image_parent="$(dirname "$image_parent")"
  fi
  camera_info_topic="$(first_existing_topic \
    "$image_parent/camera_info" \
    "$ns/oakd/rgb/preview/camera_info" \
    "$ns/oakd/rgb/camera_info" \
    "$ns/camera/image_raw/camera_info" \
    "$ns/camera/camera_info" \
  )" || fail "Could not auto-detect camera info topic for $robot. Pass --camera-info-topic=/topic."
fi

cmd_type="$(ros2 topic type "$cmd_vel_topic" 2>/dev/null || true)"
cmd_vel_stamped=true
if [[ "$cmd_type" == "geometry_msgs/msg/Twist" ]]; then
  cmd_vel_stamped=false
elif [[ "$cmd_type" == "geometry_msgs/msg/TwistStamped" ]]; then
  cmd_vel_stamped=true
else
  warn "Could not confirm cmd_vel type for $cmd_vel_topic; defaulting to TwistStamped because TA teleop uses stamped:=true."
fi

info "Health check summary:"
info "  image_topic=$image_topic"
info "  image_is_compressed=$image_is_compressed"
info "  camera_info_topic=$camera_info_topic"
info "  odom_topic=$odom_topic"
info "  scan_topic=$scan_topic"
info "  cmd_vel_topic=$cmd_vel_topic type=${cmd_type:-unknown} stamped=$cmd_vel_stamped"

if [[ "$reset_odom" == true ]]; then
  info "Waiting for reset service $reset_service..."
  if ! ros2 service list | grep -qx -- "$reset_service"; then
    warn "reset service $reset_service not visible yet; waiting up to 8 seconds"
  fi
  if ros2 service list | grep -qx -- "$reset_service" || timeout 8 bash -lc "source /opt/ros/humble/setup.bash && source '$repo_root/install/setup.bash' && until ros2 service list | grep -qx -- '$reset_service'; do sleep 1; done"; then
    info "Resetting odom via $reset_service"
    ros2 service call "$reset_service" irobot_create_msgs/srv/ResetPose "{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}" >/tmp/"${robot}_reset_pose.log" 2>&1 \
      || fail "reset_pose call failed. See /tmp/${robot}_reset_pose.log"
  else
    fail "Missing reset service $reset_service. Use --reset-odom=false to skip."
  fi
fi

if [[ "$do_launch" != true ]]; then
  info "--launch=false selected; health check completed without starting autonomy."
  exit 0
fi

info "Launching tb4_autonomy for $robot"
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
  rviz_config:="${RVIZ_CONFIG_FILE:-$repo_root/tb4_autonomy/config/rviz.rviz}" \
  state_topic:="$state_topic" \
  perf_topic:="$perf_topic"
