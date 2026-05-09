import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_output_dir = os.path.join(os.getcwd(), 'run_logs', 'camera_recordings')
    recorder_node = Node(
        package='tb4_autonomy_real',
        executable='real_camera_recorder_node',
        name='real_camera_recorder_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'output_dir': LaunchConfiguration('output_dir'),
            'output_name': LaunchConfiguration('output_name'),
            'max_duration_sec': LaunchConfiguration('max_duration_sec'),
        }],
    )
    waypoint_node = Node(
        package='tb4_autonomy_real',
        executable='real_waypoint_driver_node',
        name='real_waypoint_driver_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('run_driver')),
        parameters=[{
            'use_sim_time': True,
            'waypoints': LaunchConfiguration('waypoints'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'output_dir',
            default_value=default_output_dir,
            description='Directory for recorded camera videos.',
        ),
        DeclareLaunchArgument(
            'output_name',
            default_value='',
            description='Optional mp4 filename. Defaults to a timestamped filename.',
        ),
        DeclareLaunchArgument(
            'max_duration_sec',
            default_value='0.0',
            description='Stop recording after this duration. 0 means unlimited.',
        ),
        DeclareLaunchArgument(
            'run_driver',
            default_value='true',
            description='Run the map-based waypoint driver while recording.',
        ),
        DeclareLaunchArgument(
            'waypoints',
            default_value='',
            description='Optional x,y;x,y waypoint override.',
        ),
        recorder_node,
        waypoint_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=recorder_node,
                on_exit=[EmitEvent(event=Shutdown(reason='camera recording complete'))],
            )
        ),
    ])
