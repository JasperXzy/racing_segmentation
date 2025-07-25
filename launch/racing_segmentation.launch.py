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

    if os.getenv('WEB_SHOW') == 'True':
        print("\n'WEB_SHOW' is set to True, launching websocket_node for segmentation.\n")
        
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
    else:
        print("\n'WEB_SHOW' is not set to True, skipping websocket_node.\n")

    return LaunchDescription(launch_actions)
