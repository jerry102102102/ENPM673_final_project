import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('tb4_autonomy_real')
    default_config = os.path.join(package_share, 'config', 'arrow_detection_real_scene.yaml')
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
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw/image_color'),
        DeclareLaunchArgument('image_is_compressed', default_value='false'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/image_raw/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('cmd_vel_stamped', default_value='false'),
        DeclareLaunchArgument('annotated_image_topic', default_value='/debug/annotated_image'),
        DeclareLaunchArgument('state_topic', default_value='/autonomy/state'),
        DeclareLaunchArgument('perf_topic', default_value='/autonomy/perf'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Start RViz with the project debug layout.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='RViz config file to load when use_rviz is true.',
        ),
        Node(
            package='tb4_autonomy_real',
            executable='real_vision_controller_node',
            name='real_vision_controller_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'dry_run': LaunchConfiguration('dry_run'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'image_is_compressed': LaunchConfiguration('image_is_compressed'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                    'cmd_vel_stamped': LaunchConfiguration('cmd_vel_stamped'),
                    'annotated_image_topic': LaunchConfiguration('annotated_image_topic'),
                    'state_topic': LaunchConfiguration('state_topic'),
                    'perf_topic': LaunchConfiguration('perf_topic'),
                },
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
