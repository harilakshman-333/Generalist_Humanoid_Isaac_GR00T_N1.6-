# mock_g1_node.py
# Mock Unitree G1 Robot — Software-Only Hardware Simulator
#
# This node mimics the ROS 2 interface of the Unitree SDK 2, allowing the
# full software stack (bridge + safety + VLA) to be validated WITHOUT
# connecting to the physical robot.
#
# What it simulates:
#   ✓ Joint state feedback (23 joints) with Gaussian sensor noise
#   ✓ IMU data (accelerometer + gyroscope with drift)
#   ✓ Stereo camera image publication (synthetic gradient frames)
#   ✓ Odometry integration from velocity commands
#   ✓ Command logging to mock_robot/logs/
#   ✓ Simple physics: tracks commanded velocity, applies momentum
#
# Topics published (mimicking real G1 SDK):
#   /g1/joint_states        — sensor_msgs/JointState
#   /g1/imu                 — sensor_msgs/Imu
#   /camera/left/image_rect — sensor_msgs/Image
#   /camera/left/camera_info
#   (Simulated odometry is published to /visual_slam/tracking/odometry
#    so the bridge's LOCALIZING → NAVIGATING transition fires)
#
# Topics subscribed:
#   /g1/cmd_vel         — geometry_msgs/Twist  (from bridge)
#   /g1/joint_commands  — std_msgs/Float32MultiArray (from bridge)
#   /g1/e_stop          — std_msgs/Bool

from __future__ import annotations
import os
import math
import time
import json
import random
import logging
import threading
from datetime import datetime
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, Float32MultiArray, Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # type: ignore

log = logging.getLogger("mock_g1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] mock_g1 — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# G1 Joint Names (23 DoF — must match bridge expectations)
# ---------------------------------------------------------------------------
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_pitch_joint",
    "left_wrist_yaw_joint",   "left_wrist_roll_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_elbow_pitch_joint",
    "right_wrist_yaw_joint",  "right_wrist_roll_joint",
]

# G1 approximate default (stand-still) joint positions in radians
G1_DEFAULT_JOINT_POS = [
    0.0, 0.0, 0.0,   # left hip
    0.3, -0.2, 0.0,  # left knee-ankle
    0.0, 0.0, 0.0,   # right hip
    0.3, -0.2, 0.0,  # right knee-ankle
    0.0,             # waist yaw
    0.0, 0.2, 0.5, 0.0, 0.0,   # left arm
    0.0, -0.2, 0.5, 0.0, 0.0,  # right arm
]

# Noise standard deviation per sensor type
JOINT_POS_NOISE_STD  = 0.002   # rad
JOINT_VEL_NOISE_STD  = 0.01    # rad/s
IMU_ACCEL_NOISE_STD  = 0.05    # m/s²
IMU_GYRO_NOISE_STD   = 0.005   # rad/s
IMU_DRIFT_RATE       = 0.0001  # rad/s per second (gyro drift)

# Camera intrinsics (simulated RealSense D435i parameters)
CAM_WIDTH, CAM_HEIGHT = 640, 480
CAM_FX, CAM_FY        = 615.0, 615.0
CAM_CX, CAM_CY        = 320.0, 240.0


# ---------------------------------------------------------------------------
# Simple Robot Dynamics (2D unicycle model)
# ---------------------------------------------------------------------------
class MockRobotDynamics:
    """
    Integrates velocity commands into a pose estimate.
    Models the G1 walking as a unicycle (forward + angular velocity).
    """

    def __init__(self):
        self.x    = 0.0     # m — world frame
        self.y    = 0.0     # m — world frame
        self.z    = 0.78    # m — pelvis height (G1 standing)
        self.yaw  = 0.0     # rad
        self.vx   = 0.0     # m/s — current velocity (smoothed)
        self.vy   = 0.0
        self.vyaw = 0.0
        self._lock = threading.Lock()
        self.last_update = time.time()

    def set_cmd_vel(self, vx: float, vy: float, vyaw: float):
        """Apply low-pass filter on velocity commands (momentum)."""
        alpha = 0.3   # smoothing factor
        with self._lock:
            self.vx   = alpha * vx   + (1 - alpha) * self.vx
            self.vy   = alpha * vy   + (1 - alpha) * self.vy
            self.vyaw = alpha * vyaw + (1 - alpha) * self.vyaw

    def step(self):
        """Integrate one timestep."""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        with self._lock:
            # Rotate velocity to world frame then integrate
            cos_yaw = math.cos(self.yaw)
            sin_yaw = math.sin(self.yaw)
            self.x   += (cos_yaw * self.vx - sin_yaw * self.vy) * dt
            self.y   += (sin_yaw * self.vx + cos_yaw * self.vy) * dt
            self.yaw += self.vyaw * dt
            # Wrap yaw
            while self.yaw >  math.pi: self.yaw -= 2 * math.pi
            while self.yaw < -math.pi: self.yaw += 2 * math.pi

    def get_pose(self) -> dict:
        with self._lock:
            return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw,
                    "vx": self.vx, "vy": self.vy, "vyaw": self.vyaw}


# ---------------------------------------------------------------------------
# Command Logger
# ---------------------------------------------------------------------------
class CommandLogger:
    """Persists received commands to a JSONL file for offline analysis."""

    def __init__(self, log_dir: str = "/workspace/mock_robot/logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"commands_{ts}.jsonl")
        self._f = open(self._path, "w")
        log.info(f"Command log → {self._path}")

    def log(self, cmd_type: str, data: dict):
        entry = {"t": time.time(), "type": cmd_type, **data}
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Mock G1 Node
# ---------------------------------------------------------------------------
class MockG1Node(Node if ROS_AVAILABLE else object):

    def __init__(self):
        if ROS_AVAILABLE:
            super().__init__("mock_g1_node")
            self._init_ros()

        self.dynamics = MockRobotDynamics()
        self.joint_positions = list(G1_DEFAULT_JOINT_POS)
        self.joint_targets   = list(G1_DEFAULT_JOINT_POS)
        self.estop = False
        self.cmd_logger = CommandLogger()
        self._gyro_drift = [0.0, 0.0, IMU_DRIFT_RATE]  # drift on Z axis
        self._step_count = 0

        log.info("MockG1Node ready — simulating Unitree G1 hardware.")

    # ------------------------------------------------------------------
    # ROS 2 Setup
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

        # Subscribers (receiving commands from the bridge)
        self.sub_cmd_vel = self.create_subscription(
            Twist, "/g1/cmd_vel", self._on_cmd_vel, sensor_qos
        )
        self.sub_joint_cmds = self.create_subscription(
            Float32MultiArray, "/g1/joint_commands", self._on_joint_commands, sensor_qos
        )
        self.sub_estop = self.create_subscription(
            Bool, "/g1/e_stop", self._on_estop, reliable_qos
        )

        # Publishers (simulating G1 sensor feedback)
        self.pub_joint_states = self.create_publisher(JointState, "/g1/joint_states", 10)
        self.pub_imu          = self.create_publisher(Imu,        "/g1/imu",         10)
        self.pub_odom         = self.create_publisher(Odometry,   "/visual_slam/tracking/odometry", 10)
        self.pub_cam_image    = self.create_publisher(Image,      "/camera/left/image_rect", 10)
        self.pub_cam_info     = self.create_publisher(CameraInfo, "/camera/left/camera_info", 10)

        # Simulation loop: 50 Hz (matching bridge control rate)
        self.sim_timer = self.create_timer(0.02, self._sim_step)

        # Camera at 30 Hz (real RealSense rate)
        self.cam_timer = self.create_timer(1.0/30.0, self._publish_camera)

        log.info("ROS 2 mock publishers/subscribers ready.")

    # ------------------------------------------------------------------
    # Command Callbacks
    # ------------------------------------------------------------------
    def _on_cmd_vel(self, msg: "Twist"):
        if self.estop:
            return
        vx   = msg.linear.x
        vy   = msg.linear.y
        vyaw = msg.angular.z
        self.dynamics.set_cmd_vel(vx, vy, vyaw)
        self.cmd_logger.log("cmd_vel", {"vx": vx, "vy": vy, "vyaw": vyaw})

    def _on_joint_commands(self, msg: "Float32MultiArray"):
        if self.estop:
            return
        targets = msg.data
        if len(targets) == len(self.joint_targets):
            self.joint_targets = list(targets)
            self.cmd_logger.log("joint_cmd", {"targets": list(targets[:6])})  # log first 6

    def _on_estop(self, msg: "Bool"):
        if msg.data:
            self.estop = True
            self.dynamics.set_cmd_vel(0.0, 0.0, 0.0)
            log.warning("E-STOP received — mock halting all motion.")
            self.cmd_logger.log("e_stop", {"active": True})

    # ------------------------------------------------------------------
    # Simulation Step (50 Hz)
    # ------------------------------------------------------------------
    def _sim_step(self):
        self._step_count += 1

        # 1. Integrate dynamics
        if not self.estop:
            self.dynamics.step()

        # 2. Move joints toward targets (simple PD)
        kp = 5.0 * 0.02   # proportional gain × dt
        for i in range(len(self.joint_positions)):
            error = self.joint_targets[i] - self.joint_positions[i]
            self.joint_positions[i] += kp * error + random.gauss(0, JOINT_POS_NOISE_STD)

        # 3. Publish all sensor data
        self._publish_joint_states()
        self._publish_imu()
        self._publish_odometry()

        if self._step_count % 50 == 0:   # 1 Hz status log
            pose = self.dynamics.get_pose()
            log.info(
                f"[Mock] pos=({pose['x']:.2f},{pose['y']:.2f}) "
                f"yaw={math.degrees(pose['yaw']):.1f}° "
                f"vx={pose['vx']:.2f}m/s  estop={self.estop}"
            )

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _publish_joint_states(self):
        if not ROS_AVAILABLE:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = G1_JOINT_NAMES
        msg.position = [
            p + random.gauss(0, JOINT_POS_NOISE_STD)
            for p in self.joint_positions
        ]
        msg.velocity = [
            random.gauss(0, JOINT_VEL_NOISE_STD)
            for _ in self.joint_positions
        ]
        msg.effort = [0.0] * len(self.joint_positions)
        self.pub_joint_states.publish(msg)

    def _publish_imu(self):
        if not ROS_AVAILABLE:
            return
        pose = self.dynamics.get_pose()
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        # Simulate accelerometer: gravity vector rotated by tilt (G1 upright = [0,0,9.81])
        msg.linear_acceleration.x = random.gauss(0,   IMU_ACCEL_NOISE_STD)
        msg.linear_acceleration.y = random.gauss(0,   IMU_ACCEL_NOISE_STD)
        msg.linear_acceleration.z = random.gauss(9.81, IMU_ACCEL_NOISE_STD)

        # Simulate gyroscope: angular velocity + drift
        msg.angular_velocity.x = random.gauss(0, IMU_GYRO_NOISE_STD) + self._gyro_drift[0]
        msg.angular_velocity.y = random.gauss(0, IMU_GYRO_NOISE_STD) + self._gyro_drift[1]
        msg.angular_velocity.z = (
            pose["vyaw"]
            + random.gauss(0, IMU_GYRO_NOISE_STD)
            + self._gyro_drift[2]
        )

        self.pub_imu.publish(msg)

    def _publish_odometry(self):
        if not ROS_AVAILABLE:
            return
        pose = self.dynamics.get_pose()
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id  = "base_link"

        msg.pose.pose.position.x = pose["x"]
        msg.pose.pose.position.y = pose["y"]
        msg.pose.pose.position.z = pose["z"]

        # Quaternion from yaw
        cy = math.cos(pose["yaw"] * 0.5)
        sy = math.sin(pose["yaw"] * 0.5)
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy

        msg.twist.twist.linear.x  = pose["vx"]
        msg.twist.twist.linear.y  = pose["vy"]
        msg.twist.twist.angular.z = pose["vyaw"]

        self.pub_odom.publish(msg)

    def _publish_camera(self):
        """Publish a synthetic gradient image to simulate the forward camera."""
        if not ROS_AVAILABLE:
            return
        try:
            import numpy as np
            t = time.time()

            # Synthetic scene: moving gradient + edge clues
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            # Horizontal gradient (blue-red)
            frame[:, :, 0] = np.linspace(50, 200, CAM_WIDTH, dtype=np.uint8)
            frame[:, :, 2] = np.linspace(200, 50, CAM_WIDTH, dtype=np.uint8)
            # Add some vertical bands that shift with time (simulates motion)
            offset = int(t * 30) % CAM_WIDTH
            frame[:, offset:min(offset+20, CAM_WIDTH), 1] = 200  # green bar

            # Flatten to ROS Image
            msg = Image()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "left_camera_optical_frame"
            msg.height   = CAM_HEIGHT
            msg.width    = CAM_WIDTH
            msg.encoding = "rgb8"
            msg.step     = CAM_WIDTH * 3
            msg.data     = frame.tobytes()
            self.pub_cam_image.publish(msg)

            # Camera Info
            info = CameraInfo()
            info.header = msg.header
            info.width  = CAM_WIDTH
            info.height = CAM_HEIGHT
            info.k = [CAM_FX, 0.0, CAM_CX,
                      0.0, CAM_FY, CAM_CY,
                      0.0, 0.0, 1.0]
            info.distortion_model = "plumb_bob"
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.pub_cam_info.publish(info)

        except ImportError:
            pass   # numpy not available — skip camera


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    if not ROS_AVAILABLE:
        log.error("rclpy not found. Run inside the ros2_bridge Docker container.")
        log.info("Offline simulation test:")
        dyn = MockRobotDynamics()
        dyn.set_cmd_vel(0.5, 0.0, 0.2)
        for i in range(10):
            dyn.step()
            p = dyn.get_pose()
            print(f"  step {i}: x={p['x']:.3f} y={p['y']:.3f} yaw={math.degrees(p['yaw']):.1f}°")
        return

    rclpy.init()
    node = MockG1Node()
    log.info("Mock G1 Node spinning — all sensor topics active.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        log.info("Mock G1 shutting down.")
    finally:
        node.cmd_logger.close()
        node.destroy_node()
        rclpy.shutdown()


def _offline_integration_test():
    """Step the unicycle model with a fixed dt so output is non-trivial."""
    dyn = MockRobotDynamics()
    dyn.set_cmd_vel(0.5, 0.0, 0.2)
    dt = 0.02   # 50 Hz timestep
    for i in range(10):
        import math
        cos_yaw = math.cos(dyn.yaw)
        sin_yaw = math.sin(dyn.yaw)
        dyn.x   += (cos_yaw * dyn.vx - sin_yaw * dyn.vy) * dt
        dyn.y   += (sin_yaw * dyn.vx + cos_yaw * dyn.vy) * dt
        dyn.yaw += dyn.vyaw * dt
        p = dyn.get_pose()
        print(f"  step {i:02d}: x={p['x']:.3f} y={p['y']:.3f} yaw={math.degrees(p['yaw']):.1f}°")


if __name__ == "__main__":
    main()
