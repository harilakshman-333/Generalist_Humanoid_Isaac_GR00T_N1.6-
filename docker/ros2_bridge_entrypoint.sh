#!/bin/bash
# ros2_bridge_entrypoint.sh
# Sources ROS 2 and the colcon workspace before executing the CMD.

set -e

# Source ROS 2 base
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Source the built workspace (if it exists)
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

exec "$@"
