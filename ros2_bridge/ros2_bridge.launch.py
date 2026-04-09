# ros2_bridge.launch.py
# Launch file for the complete G1 ROS 2 communication stack.
#
# Starts (in order):
#   1. g1_bridge_node     — state machine + velocity controller
#   2. safety_monitor     — hardware limit watchdog
#   3. vla_inference_node — VLA camera→action pipeline (optional, if ROS-enabled)
#
# Usage:
#   ros2 launch ros2_bridge ros2_bridge.launch.py
#   ros2 launch ros2_bridge ros2_bridge.launch.py mock:=true
#   ros2 launch ros2_bridge ros2_bridge.launch.py vla_mock:=true ros_domain_id:=42

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    GroupAction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    # ------------------------------------------------------------------
    # Launch Arguments
    # ------------------------------------------------------------------
    args = [
        DeclareLaunchArgument(
            "mock",
            default_value="true",
            description="Run in mock mode (no hardware required)",
        ),
        DeclareLaunchArgument(
            "vla_mock",
            default_value="true",
            description="Run VLA node with mock model (no real weights needed)",
        ),
        DeclareLaunchArgument(
            "ros_domain_id",
            default_value="0",
            description="ROS_DOMAIN_ID for network isolation",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="Logging level: debug | info | warn | error",
        ),
        DeclareLaunchArgument(
            "enable_vla",
            default_value="true",
            description="Launch VLA inference node alongside bridge",
        ),
    ]

    # ------------------------------------------------------------------
    # Node: G1 Bridge (state machine + navigation controller)
    # ------------------------------------------------------------------
    bridge_node = Node(
        package="ros2_bridge",
        executable="g1_bridge_node",
        name="g1_bridge",
        namespace="g1",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        parameters=[
            {"use_sim_time": False},
            {"mock_mode": LaunchConfiguration("mock")},
        ],
        remappings=[
            # Localization input
            ("/visual_slam/tracking/odometry", "/visual_slam/tracking/odometry"),
            # VLA brain input
            ("/vla/target_pose", "/vla/target_pose"),
            # Robot command outputs
            ("/g1/cmd_vel",        "/g1/cmd_vel"),
            ("/g1/joint_commands", "/g1/joint_commands"),
            ("/g1/e_stop",         "/g1/e_stop"),
        ],
    )

    # ------------------------------------------------------------------
    # Node: Safety Monitor (100 Hz watchdog)
    # ------------------------------------------------------------------
    safety_node = Node(
        package="ros2_bridge",
        executable="safety_monitor",
        name="g1_safety_monitor",
        namespace="g1",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        parameters=[
            {"use_sim_time": False},
        ],
        remappings=[
            ("/g1/joint_states",   "/g1/joint_states"),
            ("/g1/imu",            "/g1/imu"),
            ("/g1/cmd_vel",        "/g1/cmd_vel"),
            ("/g1/e_stop",         "/g1/e_stop"),
            ("/g1/safety_status",  "/g1/safety_status"),
        ],
    )

    # ------------------------------------------------------------------
    # Node: VLA Inference (optional — can run in separate container)
    # ------------------------------------------------------------------
    vla_node = Node(
        package="gr00t_model",
        executable="run_inference",
        name="vla_inference",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_vla")),
        parameters=[
            {"use_sim_time": False},
            {"mock_mode": LaunchConfiguration("vla_mock")},
            {"ros_enabled": True},
            {"infer_hz": 5.0},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=[
            ("/vla/target_pose", "/vla/target_pose"),
        ],
    )

    # ------------------------------------------------------------------
    # Startup Info
    # ------------------------------------------------------------------
    startup_log = LogInfo(
        msg="[G1 Bridge] Launching ROS 2 bridge stack..."
              " | mock={mock} | vla_mock={vla_mock} | domain_id={ros_domain_id}".format(
                  mock="$(var mock)",
                  vla_mock="$(var vla_mock)",
                  ros_domain_id="$(var ros_domain_id)",
              )
    )

    # ------------------------------------------------------------------
    # Launch Order: safety first, then bridge, then VLA (with 1s delay)
    # ------------------------------------------------------------------
    return LaunchDescription(
        args
        + [
            startup_log,
            safety_node,    # Start safety monitor first
            bridge_node,    # Then bridge
            TimerAction(    # VLA starts 1s later (model load time)
                period=1.0,
                actions=[vla_node],
            ),
        ]
    )
