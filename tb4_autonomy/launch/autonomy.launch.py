import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('tb4_autonomy')
    default_config = os.path.join(package_share, 'config', 'autonomy.yaml')
    default_rviz = os.path.join(package_share, 'config', 'rviz.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to autonomy parameter YAML file.',
        ),
        DeclareLaunchArgument(
            'dry_run',
            default_value='true',
            description='When true, publish zero cmd_vel while keeping all monitoring active.',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Start RViz with the project debug layout.',
        ),
        Node(
            package='tb4_autonomy',
            executable='vision_controller_node',
            name='vision_controller_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {'dry_run': LaunchConfiguration('dry_run')},
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', default_rviz],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
