from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rosmaster_a1_e2e_vision")
    default_config = os.path.join(package_share, "config", "runtime.yaml")

    return LaunchDescription(
        [
            Node(
                package="rosmaster_a1_e2e_vision",
                executable="e2e_driver_node",
                name="e2e_visual_driver",
                output="screen",
                parameters=[default_config],
            )
        ]
    )
