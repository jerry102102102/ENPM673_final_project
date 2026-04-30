import os
import re
import tempfile
import launch
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import webots_ros2_driver.webots_launcher as webots_launcher_module
from webots_ros2_driver.urdf_spawner import URDFSpawner, get_webots_driver_node
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def prefer_ros_python_environment():
    os.environ.pop('PYTHONHOME', None)
    path_entries = [
        entry for entry in os.environ.get('PATH', '').split(':')
        if entry and not entry.startswith('/home/linuxbrew/.linuxbrew')
    ]
    preferred_entries = ['/usr/bin', '/bin', '/usr/local/sbin', '/usr/local/bin', '/usr/sbin', '/sbin']
    clean_entries = []
    for entry in preferred_entries + path_entries:
        if entry not in clean_entries:
            clean_entries.append(entry)
    os.environ['PATH'] = ':'.join(clean_entries)

    if os.environ.get('PYTHONPATH'):
        os.environ['PYTHONPATH'] = ':'.join(
            entry for entry in os.environ['PYTHONPATH'].split(':')
            if entry and not entry.startswith('/home/linuxbrew/.linuxbrew')
        )


def prefer_linux_webots_when_available():
    if os.path.exists('/usr/local/webots/webots'):
        os.environ['WEBOTS_HOME'] = '/usr/local/webots'
        os.environ['ROS2_WEBOTS_HOME'] = '/usr/local/webots'
        webots_launcher_module.is_wsl = lambda: False


def write_robot_description_file(robot_description):
    with tempfile.NamedTemporaryFile(mode='w', suffix='_turtlebot4_webots.urdf', delete=False) as file:
        file.write(robot_description)
        return file.name


def get_ros2_nodes(*args):
    package_dir = get_package_share_directory('tb4_sim')
    tb4_xacro_path = os.path.join(package_dir, 'resource', 'tb4_webots.xacro')
    tb4_description = xacro.process_file(
        tb4_xacro_path,
        mappings={'name': 'turtlebot4'}
    ).toxml()
    tb4_description = re.sub(r'<robot(\s|>)', r'<robot name="turtlebot4"\1', tb4_description, count=1)
    tb4_description_path = write_robot_description_file(tb4_description)

    spawn_URDF_tb4 = URDFSpawner(
        name='turtlebot4',
        robot_description=tb4_description,
        relative_path_prefix=os.path.join(package_dir, 'resource'),
        translation='6.66 0.327 -0.00564',
        # Face the nearest initial arrow instead of starting toward the wall.
        rotation='0 0 1 -1.57',
    )

    tb4_driver = WebotsController(
        robot_name='turtlebot4',
        parameters=[
            {
                'robot_description': tb4_description_path,
                'use_sim_time': True,
                'set_robot_state_publisher': True,
            },
        ],
    )

    # Ball robot extern controller
    ball_robot_driver = ExecuteProcess(
        cmd=[
            '/usr/bin/python3',
            os.path.join(
                get_package_share_directory('tb4_sim'),
                'controllers', 'ball_robot', 'ball_robot.py'
            )
        ],
        additional_env={
            'WEBOTS_CONTROLLER_URL': 'ipc://1234/ball_robot',
            'WEBOTS_HOME': '/usr/local/webots',
            'LD_LIBRARY_PATH': '/usr/local/webots/lib/controller',
        },
        output='screen',
    )

    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
    )

    return [
        spawn_URDF_tb4,
        footprint_publisher,
        ball_robot_driver,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessIO(
                target_action=spawn_URDF_tb4,
                on_stdout=lambda event: get_webots_driver_node(event, [tb4_driver]),
            )
        ),
    ]


def launch_webots(context, *args, **kwargs):
    prefer_ros_python_environment()
    prefer_linux_webots_when_available()
    os.environ.setdefault('USER', 'jerry')
    os.environ.setdefault('USERNAME', os.environ.get('USER', 'jerry'))
    package_share = get_package_share_directory('tb4_sim')
    world_file = LaunchConfiguration('world').perform(context)
    world_path = os.path.join(package_share, 'worlds', world_file)

    webots = WebotsLauncher(
        world=world_path,
        ros2_supervisor=True,
    )

    return [
        webots,
        webots._supervisor,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ]



def generate_launch_description():
    prefer_ros_python_environment()
    # Ensure controller plugins are findable
    ros_distro = os.environ.get('ROS_DISTRO', 'jazzy')
    ros_prefix = f'/opt/ros/{ros_distro}'
    os.environ['LD_LIBRARY_PATH'] = (
        os.path.join(ros_prefix, 'lib') + ':'
        + os.environ.get('LD_LIBRARY_PATH', '')
    )
    os.environ['AMENT_PREFIX_PATH'] = (
        ros_prefix + ':'
        + os.environ.get('AMENT_PREFIX_PATH', '')
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='house.wbt',
            description='Choose one of the world files from the tb4_sim share directory'
        ),
        OpaqueFunction(function=launch_webots),
    ] + get_ros2_nodes())
