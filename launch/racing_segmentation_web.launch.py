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

    websocket_node = Node(
        package='websocket',
        executable='websocket',
        name='websocket',
        output='screen',
        parameters=[
            {"image_topic": "/image"},
            {"image_type": "mjpeg"},
            {"only_show_image": False},
            {"output_fps": 0},
            {"smart_topic": "/racing_segmentation"}, 
            {"channel": 0}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    launch_actions.append(websocket_node)

    return LaunchDescription(launch_actions)
