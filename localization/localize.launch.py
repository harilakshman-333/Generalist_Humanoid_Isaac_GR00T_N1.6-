# localize.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        namespace='visual_slam',
        parameters=[{
            'enable_rectified_pose': True,
            'denoise_input_images': False,
            'rectified_images': True,
            'enable_debug_mode': False,
            'debug_dump_path': '/tmp/cuvslam',
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            'enable_slam_visualization': True,
        }],
        remappings=[
            ('stereo_camera/left/image_rect', '/camera/left/image_rect'),
            ('stereo_camera/right/image_rect', '/camera/right/image_rect'),
            ('stereo_camera/left/camera_info', '/camera/left/camera_info'),
            ('stereo_camera/right/camera_info', '/camera/right/camera_info'),
        ]
    )

    container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='visual_slam',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen'
    )

    return LaunchDescription([container])
