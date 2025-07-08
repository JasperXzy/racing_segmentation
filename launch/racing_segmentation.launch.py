# racing_segmentation/launch/racing_segmentation.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='racing_segmentation',
            executable='racing_segmentation',
            name='racing_segmentation',
            output='screen',
            parameters=[
            ],
            remappings=[
            ]
        )
    ])
