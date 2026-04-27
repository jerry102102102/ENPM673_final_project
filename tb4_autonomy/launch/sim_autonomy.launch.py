import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    autonomy_share = get_package_share_directory('tb4_autonomy')
    tb4_sim_share = get_package_share_directory('tb4_sim')

    tb4_launch = os.path.join(tb4_sim_share, 'launch', 'tb4_launcher.py')
    autonomy_launch = os.path.join(autonomy_share, 'launch', 'autonomy.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='house.wbt',
            description='Webots world file from tb4_sim/worlds.',
        ),
        DeclareLaunchArgument(
            'dry_run',
            default_value='true',
            description='Set false to allow the autonomy node to publish motion commands.',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz with the project debug layout.',
        ),
        SetEnvironmentVariable(
            name='USER',
            value=EnvironmentVariable('USER', default_value='root'),
        ),
        SetEnvironmentVariable(
            name='USERNAME',
            value=EnvironmentVariable('USERNAME', default_value='root'),
        ),
        SetEnvironmentVariable(name='WEBOTS_HOME', value='/usr/local/webots'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tb4_launch),
            launch_arguments={'world': LaunchConfiguration('world')}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(autonomy_launch),
            launch_arguments={
                'dry_run': LaunchConfiguration('dry_run'),
                'use_rviz': LaunchConfiguration('use_rviz'),
            }.items(),
        ),
    ])
