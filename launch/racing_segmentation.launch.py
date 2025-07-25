# racing_segmentation/launch/racing_segmentation.launch.py

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    launch_actions = []

    segmentation_node = Node(
        package='racing_segmentation',
        executable='racing_segmentation',
        name='racing_segmentation',
        output='screen',
        parameters=[
        ],
        remappings=[
        ]
    )
    launch_actions.append(segmentation_node)

    return LaunchDescription(launch_actions)
