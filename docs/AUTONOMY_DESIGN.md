# ENPM673 Final Project Autonomy Design

這份文件說明我們在助教提供的 Webots/TurtleBot4 simulation 上，應該如何加入自己的作業程式。重點是先把 ROS2 package、控制框架、RViz/debug 監控介面建立好，之後每位組員只要在指定 detector 或 controller 區塊補演算法。

## 1. 專案分工原則

助教提供的 `tb4_sim` package 只負責：

- 啟動 Webots simulation。
- 載入 TurtleBot4 robot。
- 提供相機、LiDAR、IMU、odom 等 ROS topics。
- 提供 `/cmd_vel` 控制介面。
- 啟動 Webots 裡的 moving ball controller。

我們自己的作業程式不要寫在 `build/` 或 `install/` 下面。那些都是 colcon build 出來的產物，會被重新產生。

建議新增一個自己的 ROS2 package：

```text
ENPM673-Final-Project-Simulation/
  src/                  # 助教提供的 tb4_sim package，盡量不改
  tb4_autonomy/         # 我們自己的作業 package
```

## 2. 我們要新增的 Package

Package 名稱建議：

```text
tb4_autonomy
```

建議檔案結構：

```text
tb4_autonomy/
  package.xml
  setup.py
  setup.cfg
  resource/tb4_autonomy
  launch/
    autonomy.launch.py
    sim_autonomy.launch.py
  config/
    autonomy.yaml
    rviz.rviz
  tb4_autonomy/
    __init__.py
    vision_controller_node.py
    state_machine.py
    motion_controller.py
    data_types.py
    detectors/
      __init__.py
      arrow_detector.py
      logo_detector.py
      moving_ball_detector.py
      horizon_detector.py
    utils/
      __init__.py
      image_tools.py
      geometry.py
  test/
    test_state_machine.py
    test_motion_controller.py
    test_detectors_contract.py
```

第一版可以先只跑一個 ROS node：`vision_controller_node`。Node 裡面再用 Python classes 拆分 detectors、state machine、motion controller。

## 3. ROS Topics and Interfaces

### Inputs

主要 input topics：

```text
/camera/image_raw/image_color   sensor_msgs/msg/Image
/camera/image_raw/camera_info   sensor_msgs/msg/CameraInfo
/odom                           nav_msgs/msg/Odometry
/scan                           sensor_msgs/msg/LaserScan
/imu                            sensor_msgs/msg/Imu
/tf
/tf_static
/clock
```

最重要的是相機：

```text
/camera/image_raw/image_color
```

雖然 xacro 裡的 base topic 是 `/camera/image_raw`，但 Webots ROS2 driver 實際 publish 的 image topic 是 `/camera/image_raw/image_color`。

### Outputs

主要 output topics：

```text
/cmd_vel                        geometry_msgs/msg/Twist
/debug/annotated_image          sensor_msgs/msg/Image
/autonomy/state                 std_msgs/msg/String
/autonomy/perf                  std_msgs/msg/String
```

控制機器人只需要 publish `/cmd_vel`：

```text
linear.x   # 前進/後退速度
angular.z  # 左右轉角速度
```

底層 diff drive controller 已由助教 launch file 處理。

## 4. 主 Node 設計

主 node：

```text
vision_controller_node.py
```

負責 ROS I/O：

```text
Subscribe:
  /camera/image_raw/image_color
  /camera/image_raw/camera_info
  /odom
  /scan

Publish:
  /cmd_vel
  /debug/annotated_image
  /autonomy/state
  /autonomy/perf
```

建議執行流程：

```text
image_callback:
  1. Convert ROS Image to OpenCV image.
  2. Run detector interfaces.
  3. Draw overlays for arrow, logo, moving ball, and horizon.
  4. Publish /debug/annotated_image.
  5. Store latest detection result.

control_timer_callback, 10-20 Hz:
  1. Read latest detections.
  2. Read latest odom/yaw.
  3. Update state machine.
  4. Compute Twist command.
  5. Publish /cmd_vel.
  6. Publish /autonomy/state.
```

這樣相機 callback 負責 perception/debug image，控制 timer 負責穩定發 `/cmd_vel`。

## 5. Detector Interfaces

每個 detector 應該提供相同風格的 `detect()` 介面。第一版可以先回傳 `None` 或 placeholder result。

```python
result = detector.detect(frame, context)
```

`frame` 是 OpenCV BGR image。`context` 可包含 camera info、odom、上一幀結果、時間戳等資訊。

建議結果資料型別集中放在：

```text
tb4_autonomy/data_types.py
```

可以先用 Python dataclass：

```python
@dataclass
class Box2D:
    x: int
    y: int
    w: int
    h: int

@dataclass
class ArrowDetection:
    box: Box2D
    direction: str  # "left", "right", "straight", "end", "unknown"
    confidence: float

@dataclass
class LogoDetection:
    box: Box2D
    confidence: float

@dataclass
class BallDetection:
    box: Box2D
    moving: bool
    ttc: float | None

@dataclass
class HorizonDetection:
    p1: tuple[int, int]
    p2: tuple[int, int]
    confidence: float
```

### Arrow Detector

File:

```text
tb4_autonomy/detectors/arrow_detector.py
```

負責 Task 1：

- 偵測黑色箭頭紙。
- 判斷箭頭方向。
- 回傳綠色 bounding box 要畫的位置。
- 回傳方向給 state machine 做轉彎。

### Logo Detector

File:

```text
tb4_autonomy/detectors/logo_detector.py
```

負責 Task 2：

- 偵測 UMD Terrapin logo。
- 回傳紅色 bounding box。
- 觸發 `LOGO_STOP` 狀態，停 3 秒再恢復。

### Moving Ball Detector

File:

```text
tb4_autonomy/detectors/moving_ball_detector.py
```

負責 Task 3：

- 偵測 moving ball。
- 回傳黃色 bounding box。
- 標籤文字為 `MOVING`。
- 估計 TTC。
- 觸發 `BALL_STOP` 狀態，直到球離開路徑。

### Horizon Detector

File:

```text
tb4_autonomy/detectors/horizon_detector.py
```

負責 Task 4：

- 每一幀都要輸出 horizon line。
- Overlay 必須全程可見。
- 可以使用 arrow bounding rectangle 作為 ROI。

## 6. State Machine

File:

```text
tb4_autonomy/state_machine.py
```

建議狀態：

```text
IDLE
CRUISE
TRACK_ARROW
EXECUTE_TURN
LOGO_STOP
BALL_STOP
FINISHED
```

優先權：

```text
BALL_STOP > LOGO_STOP > EXECUTE_TURN > TRACK_ARROW > CRUISE
```

基本規則：

- `BALL_STOP`: 看到 moving ball 或 TTC 太小時，立即停車。球清掉後回到前一個 navigation state。
- `LOGO_STOP`: 看到 logo 時停 3 秒。時間到後繼續原任務。
- `EXECUTE_TURN`: arrow detector 給出方向後，使用 odom yaw 做 closed-loop turn。
- `TRACK_ARROW`: 用 arrow 在畫面中的位置做 alignment 或 approach。
- `CRUISE`: 沒有特殊任務時慢速前進。
- `IDLE`: debug 或 dry-run 模式使用，速度為 0。

## 7. Motion Controller

File:

```text
tb4_autonomy/motion_controller.py
```

負責產生 `geometry_msgs/msg/Twist`。

建議提供：

```python
stop()
cruise()
track_x_error(error_px, image_width)
start_turn(direction, current_yaw)
update_turn(current_yaw)
```

轉彎不要只用 `sleep()`。應該用 `/odom` 的 yaw 判斷是否轉到目標角度，例如左轉 90 度或右轉 90 度。

## 8. RViz and Debug Visualization

作業要求 live bounding boxes and overlays，所以一定要 publish debug image。

Debug topic：

```text
/debug/annotated_image
```

每一幀應該畫：

- Current state。
- Frame processing time。
- Arrow green box。
- Logo red box and label。
- Moving ball yellow box and `MOVING` label。
- Horizon line。

RViz config 建議包含：

```text
Image:
  /debug/annotated_image

TF:
  /tf
  /tf_static

Odometry:
  /odom

LaserScan:
  /scan
```

也可以用 `rqt_image_view` 快速看：

```bash
ros2 run rqt_image_view rqt_image_view /debug/annotated_image
```

## 9. Execution Commands

啟動助教 simulation：

```zsh
cd /root/test/ENPM673-Final-Project-Simulation
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export USER=$(whoami)
export USERNAME=$(whoami)
ros2 launch tb4_sim tb4_launcher.py
```

之後啟動我們自己的 autonomy node：

```zsh
source /opt/ros/humble/setup.zsh
source /root/test/ENPM673-Final-Project-Simulation/install/setup.zsh
ros2 launch tb4_autonomy autonomy.launch.py
```

如果做 `sim_autonomy.launch.py`，可以一個 launch 同時啟動 simulation、autonomy node、RViz：

```zsh
ros2 launch tb4_autonomy sim_autonomy.launch.py
```

## 10. Tests

測試先不依賴 Webots。第一版測這些：

```text
test_state_machine.py:
  - logo detection triggers LOGO_STOP.
  - LOGO_STOP returns to CRUISE after 3 seconds.
  - ball detection has highest priority.
  - turn complete returns to CRUISE.

test_motion_controller.py:
  - stop() returns zero Twist.
  - cruise() returns configured speed.
  - track_x_error() turns correct direction.
  - update_turn() finishes near target yaw.

test_detectors_contract.py:
  - every detector has detect().
  - empty frame does not crash.
  - placeholder results can be drawn by overlay code.
```

Build and test:

```zsh
cd /root/test/ENPM673-Final-Project-Simulation
colcon build --symlink-install
source install/setup.zsh
pytest tb4_autonomy/test
```

## 11. Work Ownership

建議每位組員負責一個 detector，但共用同一套 ROS framework：

```text
Task 1 owner:
  tb4_autonomy/detectors/arrow_detector.py

Task 2 owner:
  tb4_autonomy/detectors/logo_detector.py

Task 3 owner:
  tb4_autonomy/detectors/moving_ball_detector.py

Task 4 owner:
  tb4_autonomy/detectors/horizon_detector.py

Integration owner:
  tb4_autonomy/vision_controller_node.py
  tb4_autonomy/state_machine.py
  tb4_autonomy/motion_controller.py
  launch and RViz config
```

每個 task owner 應該只改自己的 detector 檔案和必要的 config。不要直接在 detector 裡 publish `/cmd_vel`，控制統一交給 state machine 和 motion controller。

## 12. First Milestone

第一階段完成標準：

- `tb4_autonomy` package 可以 build。
- `vision_controller_node` 可以啟動。
- 可以 subscribe `/camera/image_raw/image_color`。
- 可以 publish `/debug/annotated_image`。
- 可以 publish `/cmd_vel`，但 dry-run 模式要能停用 motion。
- RViz 可以看到 annotated image、odom、scan、TF。
- State machine 和 motion controller 有 unit tests。
- 四個 detector 先是 placeholder，但介面固定。

完成這階段後，之後只要逐一補 CV 演算法，不需要重構 ROS 架構。
