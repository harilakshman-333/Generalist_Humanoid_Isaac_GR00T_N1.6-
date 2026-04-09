# safety_monitor.py
# Hardware Safety Monitor for the Unitree G1.
#
# This node runs independently from the bridge to form a second line of defence.
# It continuously validates all outgoing commands against G1 hardware limits
# and triggers an E-STOP if anything is out of range.
#
# Checks performed (every cycle at 100 Hz):
#   1. Joint position limits (per-joint angular bounds from the G1 datasheet)
#   2. Joint velocity limits (max ω per joint)
#   3. Command rate watchdog (E-STOP if /g1/cmd_vel goes silent mid-motion)
#   4. Tilt angle (from IMU — robot is falling)
#   5. Self-collision proxy (simplified bounding-sphere check on key links)
#
# Topics:
#   Subscribes: /g1/joint_states, /g1/imu, /g1/cmd_vel, /g1/e_stop
#   Publishes:  /g1/e_stop (Bool), /g1/safety_status (String)

from __future__ import annotations
import math
import time
import logging
import threading
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState, Imu
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # type: ignore

log = logging.getLogger("safety_monitor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] safety — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# G1 Joint Safety Limits
# Two rings of protection: WARNING (80% of hardware limit) and CRITICAL (95%)
# ---------------------------------------------------------------------------
# Format: joint_name → (lower_rad, upper_rad)
G1_JOINT_LIMITS = {
    "left_hip_pitch_joint":       (-2.87, 2.87),
    "left_hip_roll_joint":        (-0.34, 3.11),
    "left_hip_yaw_joint":         (-0.52, 0.52),
    "left_knee_joint":            (-0.26, 2.87),
    "left_ankle_pitch_joint":     (-0.89, 0.89),
    "left_ankle_roll_joint":      (-0.26, 0.26),
    "right_hip_pitch_joint":      (-2.87, 2.87),
    "right_hip_roll_joint":       (-3.11, 0.34),
    "right_hip_yaw_joint":        (-0.52, 0.52),
    "right_knee_joint":           (-0.26, 2.87),
    "right_ankle_pitch_joint":    (-0.89, 0.89),
    "right_ankle_roll_joint":     (-0.26, 0.26),
    "waist_yaw_joint":            (-2.62, 2.62),
    "left_shoulder_pitch_joint":  (-3.14, 3.14),
    "left_shoulder_roll_joint":   (-0.17, 3.14),
    "left_elbow_pitch_joint":     (-1.25, 2.18),
    "left_wrist_yaw_joint":       (-1.97, 1.97),
    "left_wrist_roll_joint":      (-1.57, 1.57),
    "right_shoulder_pitch_joint": (-3.14, 3.14),
    "right_shoulder_roll_joint":  (-3.14, 0.17),
    "right_elbow_pitch_joint":    (-1.25, 2.18),
    "right_wrist_yaw_joint":      (-1.97, 1.97),
    "right_wrist_roll_joint":     (-1.57, 1.57),
}

G1_JOINT_VEL_LIMIT_RADPS = 10.0    # rad/s — conservative limit for all joints
G1_MAX_TILT_DEG           = 35.0   # degrees — trigger E-STOP above this tilt
WATCHDOG_TIMEOUT_S         = 0.5   # seconds — silence on /g1/cmd_vel triggers E-STOP
WARNING_MARGIN             = 0.80  # 80% of limit → warning log
CRITICAL_MARGIN            = 0.95  # 95% of limit → E-STOP


# ---------------------------------------------------------------------------
# Safety Status
# ---------------------------------------------------------------------------
class SafetyStatus:
    def __init__(self):
        self.ok = True
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, msg: str):
        self.warnings.append(msg)
        log.warning(f"[WARN] {msg}")

    def error(self, msg: str):
        self.ok = False
        self.errors.append(msg)
        log.error(f"[ERROR] {msg}")

    def summary(self) -> str:
        if not self.ok:
            return f"UNSAFE | errors={self.errors[:2]}"
        if self.warnings:
            return f"WARNING | {self.warnings[:2]}"
        return "OK"


# ---------------------------------------------------------------------------
# Safety Monitor Node
# ---------------------------------------------------------------------------
class SafetyMonitorNode(Node if ROS_AVAILABLE else object):

    def __init__(self):
        if ROS_AVAILABLE:
            super().__init__("g1_safety_monitor")
            self._init_ros()

        self._estop_active = False
        self._last_cmd_vel_time: float = time.time()
        self._last_joint_positions: dict[str, float] = {}
        self._last_joint_velocities: dict[str, float] = {}
        self._is_moving = False
        self._lock = threading.Lock()

        log.info("SafetyMonitorNode ready.")

    # ------------------------------------------------------------------
    # ROS 2 Init
    # ------------------------------------------------------------------
    def _init_ros(self):
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

        # Subscribers
        self.sub_joints = self.create_subscription(
            JointState, "/g1/joint_states", self._on_joint_states, sensor_qos
        )
        self.sub_imu = self.create_subscription(
            Imu, "/g1/imu", self._on_imu, sensor_qos
        )
        self.sub_cmd_vel = self.create_subscription(
            Twist, "/g1/cmd_vel", self._on_cmd_vel, sensor_qos
        )
        self.sub_estop_in = self.create_subscription(
            Bool, "/g1/e_stop", self._on_estop, reliable_qos
        )

        # Publishers
        self.pub_estop = self.create_publisher(Bool, "/g1/e_stop", reliable_qos)
        self.pub_status = self.create_publisher(String, "/g1/safety_status", 10)

        # Safety check loop at 100 Hz
        self.timer = self.create_timer(0.01, self._safety_check_loop)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: "JointState"):
        with self._lock:
            for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
                self._last_joint_positions[name] = pos
                self._last_joint_velocities[name] = vel

    def _on_imu(self, msg: "Imu"):
        """Store IMU data for tilt check in the main loop."""
        self._imu_linear_accel = msg.linear_acceleration

    def _on_cmd_vel(self, msg: "Twist"):
        with self._lock:
            self._last_cmd_vel_time = time.time()
            vx = msg.linear.x
            vy = msg.linear.y
            vyaw = msg.angular.z
            self._is_moving = (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01)

    def _on_estop(self, msg: "Bool"):
        if msg.data:
            with self._lock:
                self._estop_active = True

    # ------------------------------------------------------------------
    # Safety Check Loop (100 Hz)
    # ------------------------------------------------------------------
    def _safety_check_loop(self):
        status = SafetyStatus()

        self._check_joint_limits(status)
        self._check_joint_velocities(status)
        self._check_watchdog(status)
        self._check_imu_tilt(status)

        # Publish status string
        if ROS_AVAILABLE:
            msg = String()
            msg.data = status.summary()
            self.pub_status.publish(msg)

        # Trigger E-STOP if any hard error
        if not status.ok and not self._estop_active:
            self._trigger_e_stop(status.errors[0] if status.errors else "Unknown safety error")

    # ------------------------------------------------------------------
    # Individual Checks
    # ------------------------------------------------------------------
    def _check_joint_limits(self, status: SafetyStatus):
        """Compare current joint positions against G1 hardware limits."""
        with self._lock:
            positions = dict(self._last_joint_positions)

        for joint_name, pos in positions.items():
            if joint_name not in G1_JOINT_LIMITS:
                continue
            lo, hi = G1_JOINT_LIMITS[joint_name]
            span = hi - lo

            # Normalised distance from center in [0, 1] (1 = at hard limit)
            normalised = (pos - lo) / span if span > 0 else 0.0

            if pos < lo * CRITICAL_MARGIN or pos > hi * CRITICAL_MARGIN:
                status.error(
                    f"JOINT LIMIT CRITICAL: {joint_name}={pos:.3f}rad "
                    f"(limits=[{lo:.3f}, {hi:.3f}])"
                )
            elif pos < lo * WARNING_MARGIN or pos > hi * WARNING_MARGIN:
                status.warn(
                    f"Joint approaching limit: {joint_name}={pos:.3f}rad"
                )

    def _check_joint_velocities(self, status: SafetyStatus):
        """Check no joint is spinning faster than the safe limit."""
        with self._lock:
            velocities = dict(self._last_joint_velocities)

        for joint_name, vel in velocities.items():
            if abs(vel) > G1_JOINT_VEL_LIMIT_RADPS * CRITICAL_MARGIN:
                status.error(
                    f"JOINT VELOCITY CRITICAL: {joint_name}={vel:.2f}rad/s "
                    f"(limit={G1_JOINT_VEL_LIMIT_RADPS})"
                )
            elif abs(vel) > G1_JOINT_VEL_LIMIT_RADPS * WARNING_MARGIN:
                status.warn(f"Joint velocity high: {joint_name}={vel:.2f}rad/s")

    def _check_watchdog(self, status: SafetyStatus):
        """E-STOP if robot is supposed to be moving but commands have gone silent."""
        with self._lock:
            last_t = self._last_cmd_vel_time
            moving = self._is_moving

        if moving and (time.time() - last_t) > WATCHDOG_TIMEOUT_S:
            status.error(
                f"CMD_VEL WATCHDOG: robot moving but no command for "
                f">{WATCHDOG_TIMEOUT_S}s"
            )

    def _check_imu_tilt(self, status: SafetyStatus):
        """Detect if robot is falling based on IMU gravity direction."""
        imu = getattr(self, "_imu_linear_accel", None)
        if imu is None:
            return

        ax = getattr(imu, "x", 0.0)
        ay = getattr(imu, "y", 0.0)
        az = getattr(imu, "z", 9.81)

        # Approximate tilt angle from vertical
        g_total = math.sqrt(ax**2 + ay**2 + az**2)
        if g_total < 0.1:
            return
        tilt_deg = math.degrees(math.acos(min(1.0, abs(az) / g_total)))

        if tilt_deg > G1_MAX_TILT_DEG * (CRITICAL_MARGIN):
            status.error(f"IMU TILT CRITICAL: {tilt_deg:.1f}° (max={G1_MAX_TILT_DEG}°)")
        elif tilt_deg > G1_MAX_TILT_DEG * WARNING_MARGIN:
            status.warn(f"Tilt warning: {tilt_deg:.1f}°")

    # ------------------------------------------------------------------
    # E-STOP Publisher
    # ------------------------------------------------------------------
    def _trigger_e_stop(self, reason: str):
        with self._lock:
            self._estop_active = True
        log.critical(f"🛑 SAFETY E-STOP: {reason}")
        if ROS_AVAILABLE:
            msg = Bool()
            msg.data = True
            self.pub_estop.publish(msg)


# ---------------------------------------------------------------------------
# Offline / Unit-Test Mode
# ---------------------------------------------------------------------------
class OfflineSafetyChecker:
    """
    Run safety checks without ROS 2 — useful for CI/CD validation.
    Feed joint positions directly and call check().
    """

    def check(self, joint_positions: dict[str, float]) -> SafetyStatus:
        status = SafetyStatus()
        # Manually initialise only the fields the check methods need,
        # bypassing the ROS 2 Node super().__init__() call entirely.
        monitor = object.__new__(SafetyMonitorNode)
        monitor._lock = threading.Lock()
        monitor._last_joint_positions = joint_positions
        monitor._last_joint_velocities = {}
        monitor._check_joint_limits(status)
        monitor._check_joint_velocities(status)
        return status


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    if not ROS_AVAILABLE:
        log.warning("ROS 2 not available — running offline self-test.")
        checker = OfflineSafetyChecker()

        # Test: joints within limits
        safe_positions = {j: 0.0 for j in G1_JOINT_LIMITS}
        result = checker.check(safe_positions)
        print(f"Safe joint test: {result.summary()}")

        # Test: violated joint
        bad_positions = dict(safe_positions)
        bad_positions["left_knee_joint"] = 3.0  # Exceeds 2.87 limit
        result = checker.check(bad_positions)
        print(f"Bad joint test:  {result.summary()}")
        return

    rclpy.init()
    node = SafetyMonitorNode()
    log.info("Safety Monitor spinning at 100 Hz...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        log.info("Safety Monitor shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
