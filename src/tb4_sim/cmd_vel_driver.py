import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy


def _float_property(properties, name, default):
    try:
        return float(properties.get(name, default))
    except (TypeError, ValueError):
        return default


def _string_property(properties, name, default):
    value = properties.get(name, default)
    return str(value) if value is not None else default


class CmdVelDriver:
    def init(self, webots_node, properties):
        if not rclpy.ok():
            rclpy.init(args=None)

        self._robot = webots_node.robot
        self._basic_timestep_s = self._robot.getBasicTimeStep() / 1000.0
        self._node = rclpy.create_node('tb4_cmd_vel_driver')

        self._cmd_vel_topic = _string_property(properties, 'cmdVelTopic', '/cmd_vel')
        self._odom_topic = _string_property(properties, 'odomTopic', '/odom')
        self._wheel_separation = _float_property(properties, 'wheelSeparation', 0.25)
        self._wheel_radius = _float_property(properties, 'wheelRadius', 0.036)
        self._max_wheel_speed = _float_property(properties, 'maxWheelSpeed', 20.0)
        self._cmd_timeout_s = _float_property(properties, 'cmdTimeout', 0.5)

        self._left_motor = self._get_motor(
            _string_property(properties, 'leftMotor', 'left_wheel_joint'),
            ['left_wheel_joint', 'left wheel motor', 'left_wheel', 'left wheel'],
        )
        self._right_motor = self._get_motor(
            _string_property(properties, 'rightMotor', 'right_wheel_joint'),
            ['right_wheel_joint', 'right wheel motor', 'right_wheel', 'right wheel'],
        )
        for motor in (self._left_motor, self._right_motor):
            if motor is not None:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)

        self._target = Twist()
        self._last_cmd_time = None
        self._last_cmd_log_time = -1.0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self._node.create_subscription(Twist, self._cmd_vel_topic, self._cmd_vel_callback, 10)
        self._odom_pub = self._node.create_publisher(Odometry, self._odom_topic, 10)
        self._node.get_logger().info(
            f'tb4 cmd_vel driver ready: {self._cmd_vel_topic} -> wheels, odom={self._odom_topic}'
        )

    def _get_motor(self, preferred_name, fallback_names):
        for name in [preferred_name] + [item for item in fallback_names if item != preferred_name]:
            try:
                return self._robot.getDevice(name)
            except Exception:
                continue
        available = self._device_names()
        self._node.get_logger().error(
            f'Unable to find wheel motor {preferred_name!r}. Available devices: {available}'
        )
        return None

    def _device_names(self):
        names = []
        try:
            for index in range(self._robot.getNumberOfDevices()):
                names.append(self._robot.getDeviceByIndex(index).getName())
        except Exception:
            return 'unknown'
        return ', '.join(names)

    def _cmd_vel_callback(self, msg):
        self._target = msg
        self._last_cmd_time = self._robot.getTime()
        if (
            abs(msg.linear.x) > 1e-6 or abs(msg.angular.z) > 1e-6
        ) and self._last_cmd_time - self._last_cmd_log_time > 1.0:
            self._last_cmd_log_time = self._last_cmd_time
            self._node.get_logger().info(
                f'cmd_vel linear.x={msg.linear.x:.3f} angular.z={msg.angular.z:.3f}'
            )

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)

        now = self._robot.getTime()
        dt = self._basic_timestep_s
        if self._last_cmd_time is None or now - self._last_cmd_time > self._cmd_timeout_s:
            linear_x = 0.0
            angular_z = 0.0
        else:
            linear_x = self._target.linear.x
            angular_z = self._target.angular.z

        left_speed = (linear_x - angular_z * self._wheel_separation / 2.0) / self._wheel_radius
        right_speed = (linear_x + angular_z * self._wheel_separation / 2.0) / self._wheel_radius
        left_speed = max(-self._max_wheel_speed, min(self._max_wheel_speed, left_speed))
        right_speed = max(-self._max_wheel_speed, min(self._max_wheel_speed, right_speed))

        if self._left_motor is not None:
            self._left_motor.setVelocity(left_speed)
        if self._right_motor is not None:
            self._right_motor.setVelocity(right_speed)

        self._integrate_odom(linear_x, angular_z, dt)
        self._publish_odom(linear_x, angular_z)

    def _integrate_odom(self, linear_x, angular_z, dt):
        self._yaw = math.atan2(
            math.sin(self._yaw + angular_z * dt),
            math.cos(self._yaw + angular_z * dt),
        )
        self._x += linear_x * math.cos(self._yaw) * dt
        self._y += linear_x * math.sin(self._yaw) * dt

    def _publish_odom(self, linear_x, angular_z):
        odom = Odometry()
        odom.header.stamp = self._node.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.z = math.sin(self._yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self._yaw / 2.0)
        odom.twist.twist.linear.x = linear_x
        odom.twist.twist.angular.z = angular_z
        self._odom_pub.publish(odom)
