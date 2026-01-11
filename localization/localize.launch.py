from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam',
            name='visual_slam_node',
            parameters=[{
                'enable_rectified_pose': True,
                'denoise_input_images': False,
            }],
            output='screen'
        ),
        # Node configuration for cuVGL would go here
    ])
