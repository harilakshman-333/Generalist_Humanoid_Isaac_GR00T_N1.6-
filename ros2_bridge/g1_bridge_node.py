# g1_bridge_node.py
# ROS 2 Communication Bridge — the "Nervous System" of the G1 stack.
#
# This node sits between all four subsystems and coordinates them:
#
#   [Localization / VSLAM]     → /visual_slam/tracking/odometry
#   [VLA Brain]                → /vla/target_pose
#   [RL Policy velocity cmd]   ← /g1/cmd_vel
#   [Motor joint targets]      ← /g1/joint_commands
#   [Emergency stop]          ↔ /g1/e_stop  (bidirectional)
#
# State Machine:
#   IDLE → LOCALIZING → NAVIGATING → MANIPULATING → E_STOP
#              ↑___________________|
#
# Run inside the ros2_bridge container:
#   ros2 run ros2_bridge g1_bridge_node

from __future__ import annotations
import math
import time
import threading
import logging
from enum import Enum, auto
from typing import Optional

# ROS 2 imports — gracefully degrade if not available (for offline testing)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import (
        PoseStamped, TwistStamped, Twist, Vector3
    )
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState, Imu
    from std_msgs.msg import Bool, String, Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object   # type: ignore

log = logging.getLogger("g1_bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] g1_bridge — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# G1 Hardware Limits (from Unitree G1 spec)
# ---------------------------------------------------------------------------
G1_MAX_LIN_VEL_MPS   = 1.0     # m/s — walking speed cap
G1_MAX_ANG_VEL_RADPS = 1.0     # rad/s — turning speed cap
G1_MAX_STEP_HEIGHT_M = 0.20    # m — stair climbing height cap
WATCHDOG_TIMEOUT_S   = 0.5     # seconds — E-STOP if no cmd received


# ---------------------------------------------------------------------------
# Robot State Machine
# ---------------------------------------------------------------------------
class RobotState(Enum):
    IDLE         = auto()
    LOCALIZING   = auto()   # Waiting for valid pose from VSLAM
    NAVIGATING   = auto()   # Walking to VLA-specified target
    MANIPULATING = auto()   # Arm in active use
    E_STOP       = auto()   # Emergency stop — all motion halted


# ---------------------------------------------------------------------------
# Bridge Node
# ---------------------------------------------------------------------------
class G1BridgeNode(Node if ROS_AVAILABLE else object):
    """
    Central coordination node for the Unitree G1.
    Implements the state machine, velocity command generation,
    and joint command forwarding.
    """

    def __init__(self):
        if ROS_AVAILABLE:
            super().__init__("g1_bridge_node")
            self._init_ros()
        else:
            log.warning("ROS 2 not available — bridge running in offline/debug mode.")

        # State
        self.state: RobotState = RobotState.IDLE
        self.current_pose: Optional[dict] = None      # {x, y, z, yaw}
        self.target_pose: Optional[dict] = None       # {x, y, z, gripper_open}
        self.vla_action_type: str = "navigate"
        self._last_cmd_time: float = time.time()
        self._e_stop_active: bool = False
        self._lock = threading.Lock()

        log.info(f"G1BridgeNode initialised | state={self.state.name}")

    # ------------------------------------------------------------------
    # ROS 2 Setup
    # ------------------------------------------------------------------
    def _init_ros(self):
        # QoS — best effort for high-rate sensor data, reliable for commands
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self.sub_odom = self.create_subscription(
            Odometry,
            "/visual_slam/tracking/odometry",
            self._on_odometry,
            sensor_qos,
        )
        self.sub_vla = self.create_subscription(
            PoseStamped,
            "/vla/target_pose",
            self._on_vla_target,
            reliable_qos,
        )
        self.sub_estop_in = self.create_subscription(
            Bool,
            "/g1/e_stop",
            self._on_e_stop_signal,
            reliable_qos,
        )
        self.sub_joint_states = self.create_subscription(
            JointState,
            "/g1/joint_states",
            self._on_joint_states,
            sensor_qos,
        )
        self.sub_imu = self.create_subscription(
            Imu,
            "/g1/imu",
            self._on_imu,
            sensor_qos,
        )

        # --- Publishers ---
        self.pub_cmd_vel = self.create_publisher(Twist, "/g1/cmd_vel", 10)
        self.pub_joint_cmds = self.create_publisher(
            Float32MultiArray, "/g1/joint_commands", 10
        )
        self.pub_estop = self.create_publisher(Bool, "/g1/e_stop", reliable_qos)
        self.pub_state = self.create_publisher(String, "/g1/state", 10)

        # --- Control loop timer: 50 Hz ---
        self.control_timer = self.create_timer(0.02, self._control_loop)

        # --- Watchdog timer: checks for command staleness ---
        self.watchdog_timer = self.create_timer(0.1, self._watchdog)

        log.info("ROS 2 subscribers and publishers initialised.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_odometry(self, msg: "Odometry"):
        """Update current robot pose from VSLAM."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        # Extract yaw from quaternion
        yaw = self._quat_to_yaw(orient.x, orient.y, orient.z, orient.w)
        with self._lock:
            self.current_pose = {"x": pos.x, "y": pos.y, "z": pos.z, "yaw": yaw}
            # Transition: if we were waiting for localization, move to NAVIGATING
            if self.state == RobotState.LOCALIZING:
                log.info("Pose acquired — transitioning LOCALIZING → NAVIGATING")
                self.state = RobotState.NAVIGATING

    def _on_vla_target(self, msg: "PoseStamped"):
        """Receive high-level target from VLA brain."""
        with self._lock:
            self.target_pose = {
                "x": msg.pose.position.x,
                "y": msg.pose.position.y,
                "z": msg.pose.position.z,
            }
            self._last_cmd_time = time.time()
            log.info(
                f"VLA target received: "
                f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})"
            )

    def _on_e_stop_signal(self, msg: "Bool"):
        """External E-STOP signal (from safety_monitor or operator)."""
        if msg.data:
            self._trigger_e_stop("External E-STOP signal received")
        else:
            with self._lock:
                if self.state == RobotState.E_STOP:
                    log.info("E-STOP cleared by external signal.")
                    self._e_stop_active = False
                    self.state = RobotState.IDLE

    def _on_joint_states(self, msg: "JointState"):
        """Receive joint state feedback for monitoring."""
        # Forwarded to safety_monitor via /g1/joint_states (shared topic)
        # Bridge only logs if debug needed
        pass

    def _on_imu(self, msg: "Imu"):
        """Monitor IMU for tilt — trigger E-STOP if robot is falling."""
        # Gravity vector
        lin_accel = msg.linear_acceleration
        # Magnitude of horizontal acceleration (should be ~0 when upright)
        horizontal_g = math.sqrt(lin_accel.x**2 + lin_accel.y**2)
        vertical_g   = abs(lin_accel.z)
        # If mostly horizontal acceleration from gravity → robot is tilted badly
        if vertical_g < 3.0 and horizontal_g > 7.0:
            self._trigger_e_stop(
                f"IMU fall detected: h_g={horizontal_g:.1f} v_g={vertical_g:.1f}"
            )

    # ------------------------------------------------------------------
    # Control Loop (50 Hz)
    # ------------------------------------------------------------------
    def _control_loop(self):
        """Main control tick — runs at 50 Hz."""
        with self._lock:
            state = self.state
            pose  = self.current_pose
            target = self.target_pose

        if state == RobotState.E_STOP:
            self._publish_zero_velocity()
            return

        if state == RobotState.IDLE:
            # Transition to LOCALIZING once we start receiving data
            if pose is not None:
                with self._lock:
                    self.state = RobotState.NAVIGATING
            else:
                with self._lock:
                    self.state = RobotState.LOCALIZING
            return

        if state == RobotState.LOCALIZING:
            self._publish_zero_velocity()
            return

        if state in (RobotState.NAVIGATING, RobotState.MANIPULATING):
            if target is None or pose is None:
                self._publish_zero_velocity()
                return
            cmd = self._compute_velocity_command(pose, target)
            self._publish_cmd_vel(cmd)

        # Publish state for monitoring
        if ROS_AVAILABLE:
            msg = String()
            msg.data = state.name
            self.pub_state.publish(msg)

    # ------------------------------------------------------------------
    # Velocity Command Generation
    # ------------------------------------------------------------------
    def _compute_velocity_command(self, pose: dict, target: dict) -> dict:
        """
        Simple proportional navigation controller.

        Computes the velocity command to drive the robot toward the target
        position. The RL policy will then translate this into joint torques.

        Returns: {vx, vy, vyaw} — all in robot base frame, clamped to G1 limits.
        """
        dx = target["x"] - pose["x"]
        dy = target["y"] - pose["y"]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.05:   # 5 cm threshold — consider arrived
            log.info("Target reached.")
            with self._lock:
                if self.state == RobotState.NAVIGATING:
                    self.state = RobotState.IDLE
                self.target_pose = None
            return {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}

        # Angle to target in world frame
        angle_to_target = math.atan2(dy, dx)
        # Heading error (difference between current yaw and target direction)
        yaw_error = self._wrap_angle(angle_to_target - pose["yaw"])

        # Proportional gains
        kv   = 0.6    # linear gain
        kyaw = 1.0    # angular gain

        # Forward velocity (only when heading is roughly right)
        heading_ok = abs(yaw_error) < math.pi / 4
        vx = min(kv * distance, G1_MAX_LIN_VEL_MPS) if heading_ok else 0.0
        vy = 0.0   # No sideways walking for now (can enable for crab-walk)
        vyaw = max(-G1_MAX_ANG_VEL_RADPS, min(kyaw * yaw_error, G1_MAX_ANG_VEL_RADPS))

        return {"vx": vx, "vy": vy, "vyaw": vyaw}

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _publish_cmd_vel(self, cmd: dict):
        if not ROS_AVAILABLE:
            log.debug(f"[cmd_vel] vx={cmd['vx']:.3f} vy={cmd['vy']:.3f} vyaw={cmd['vyaw']:.3f}")
            return
        msg = Twist()
        msg.linear.x  = float(cmd["vx"])
        msg.linear.y  = float(cmd["vy"])
        msg.angular.z = float(cmd["vyaw"])
        self.pub_cmd_vel.publish(msg)

    def _publish_zero_velocity(self):
        self._publish_cmd_vel({"vx": 0.0, "vy": 0.0, "vyaw": 0.0})

    # ------------------------------------------------------------------
    # E-STOP
    # ------------------------------------------------------------------
    def _trigger_e_stop(self, reason: str):
        with self._lock:
            if self.state == RobotState.E_STOP:
                return
            log.error(f"E-STOP TRIGGERED: {reason}")
            self.state = RobotState.E_STOP
            self._e_stop_active = True

        self._publish_zero_velocity()
        if ROS_AVAILABLE:
            msg = Bool()
            msg.data = True
            self.pub_estop.publish(msg)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _watchdog(self):
        """Trigger E-STOP if VLA has not sent a command recently."""
        with self._lock:
            state = self.state
            last_cmd = self._last_cmd_time

        if state not in (RobotState.NAVIGATING, RobotState.MANIPULATING):
            return
        if time.time() - last_cmd > WATCHDOG_TIMEOUT_S:
            self._trigger_e_stop(
                f"Watchdog timeout — no VLA command for >{WATCHDOG_TIMEOUT_S}s"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """Extract yaw (rotation about Z) from quaternion."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    if not ROS_AVAILABLE:
        log.error("rclpy not found. Install ROS 2 Humble or run inside ros2_bridge container.")
        return

    rclpy.init()
    node = G1BridgeNode()
    log.info("G1 Bridge Node spinning...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        log.info("Shutting down G1 Bridge Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
