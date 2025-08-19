from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os
import sys


def generate_launch_description():
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # Robot ip address
    os.environ['MROS_AGENT_IP'] = '10.192.1.2'

    # Get URDF file path
    urdf_file = PathJoinSubstitution(
        [FindPackageShare("robot_description"), "pointfoot/" + robot_type + "/urdf", "robot.urdf"]
    )

    rviz_config_file = get_package_share_directory('robot_visualization') + '/rviz/pointfoot.rviz'

    # Load the URDF file
    robot_description = Command(["xacro ", urdf_file, " ",])

    return LaunchDescription([
        DeclareLaunchArgument(
            "bridge_mros2ros",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "mros2ros_include",
            default_value="/tf;/tf_static;/odom;/VisualizeStatePointFoot",
        ),
        Node(
            package='mrosbridger',
            executable='mrosbridger',
            output='screen',
            parameters=[
                {
                    "bridge_mros2ros": LaunchConfiguration("bridge_mros2ros"),
                    "mros2ros_include": LaunchConfiguration("mros2ros_include"),
                },
            ],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                {
                    "robot_description": robot_description,
                },
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_file]
        ),
    ])