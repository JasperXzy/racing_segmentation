import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory


def generate_launch_description():
    image_width_launch_arg = DeclareLaunchArgument(
        "dnn_sample_image_width", default_value=TextSubstitution(text="640")
    )
    image_height_launch_arg = DeclareLaunchArgument(
        "dnn_sample_image_height", default_value=TextSubstitution(text="480")
    )

    web_show = os.getenv('WEB_SHOW')
    print("web_show is ", web_show)

    jpeg_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_encode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'shared_mem',
            'codec_out_mode': 'ros',
            'codec_sub_topic': '/hbmem_img',
            'codec_pub_topic': '/image_jpeg',
        }.items()
    )

    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image_jpeg',
            'websocket_smart_topic': '/racing_segmentation'
        }.items()
    )

    racing_segmentation_node = Node(
        package='racing_segmentation',
        executable='racing_segmentation',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )

    if web_show == "TRUE":
        return LaunchDescription([
            racing_segmentation_node,
            jpeg_codec_node,
            web_node
        ])
    else:
        return LaunchDescription([
            racing_segmentation_node
        ])
