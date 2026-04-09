# g1_env_cfg.py
# Unitree G1 Specific Isaac Lab Environment Configuration.
# This replaces the generic "Isaac-Humanoid-v0" with a task tuned to G1's
# 23-DoF joint layout, mass distribution, and sensor suite.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# G1 Joint Definitions (23 DoF)
# Reference: Unitree G1 technical spec / URDF joint names
# ---------------------------------------------------------------------------
G1_JOINT_NAMES: List[str] = [
    # Left Leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right Leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Left Arm (5)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    # Right Arm (5)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
]

# G1 joint position limits [lower, upper] in radians
# Derived from Unitree G1 URDF / hardware spec
G1_JOINT_LIMITS = {
    "left_hip_pitch_joint":    (-2.87, 2.87),
    "left_hip_roll_joint":     (-0.34, 3.11),
    "left_hip_yaw_joint":      (-0.52, 0.52),
    "left_knee_joint":         (-0.26, 2.87),
    "left_ankle_pitch_joint":  (-0.89, 0.89),
    "left_ankle_roll_joint":   (-0.26, 0.26),
    "right_hip_pitch_joint":   (-2.87, 2.87),
    "right_hip_roll_joint":    (-3.11, 0.34),
    "right_hip_yaw_joint":     (-0.52, 0.52),
    "right_knee_joint":        (-0.26, 2.87),
    "right_ankle_pitch_joint": (-0.89, 0.89),
    "right_ankle_roll_joint":  (-0.26, 0.26),
    "waist_yaw_joint":         (-2.62, 2.62),
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


# ---------------------------------------------------------------------------
# Observation Space Configuration
# ---------------------------------------------------------------------------
@dataclass
class G1ObservationCfg:
    """Defines what the policy sees at each timestep."""

    # Base state (6)
    base_lin_vel: bool = True        # [vx, vy, vz] in base frame — 3 dims
    base_ang_vel: bool = True        # [wx, wy, wz] in base frame — 3 dims
    # Orientation (3)
    projected_gravity: bool = True   # gravity vector projected on base — 3 dims
    # Commands (3)
    velocity_commands: bool = True   # [vx_cmd, vy_cmd, yaw_cmd] — 3 dims
    # Joint state (46)
    joint_pos: bool = True           # 23 joints × 1 = 23 dims (relative to default)
    joint_vel: bool = True           # 23 joints × 1 = 23 dims
    # History (23)
    last_action: bool = True         # Previous action — 23 dims

    @property
    def total_dims(self) -> int:
        return 3 + 3 + 3 + 3 + 23 + 23 + 23  # = 81


# ---------------------------------------------------------------------------
# Action Space Configuration
# ---------------------------------------------------------------------------
@dataclass
class G1ActionCfg:
    """Maps network output to joint position targets."""

    num_actions: int = 23            # One target per joint
    action_scale: float = 0.5       # Scale factor applied to raw network output (rad)
    clip_actions: float = 100.0     # Clip after scaling (large = effectively no clip)


# ---------------------------------------------------------------------------
# Reward Function Configuration
# ---------------------------------------------------------------------------
@dataclass
class G1RewardCfg:
    """Weighted reward terms for locomotion training.

    Positive rewards encourage desired behaviour; negative weights are penalties.
    Weights are tuned for 512-environment parallel training at 200Hz sim rate.
    """

    # --- Positive terms ---
    # Track commanded linear velocity (primary objective)
    lin_vel_tracking_weight: float = 1.5
    lin_vel_tracking_exp_coef: float = -4.0  # exp(-coef * error^2)

    # Track commanded yaw rate
    ang_vel_tracking_weight: float = 0.75
    ang_vel_tracking_exp_coef: float = -4.0

    # Alive bonus (keep the robot up)
    alive_weight: float = 0.5

    # --- Negative / penalty terms ---
    # Penalise z-axis linear velocity (bouncing)
    lin_vel_z_penalty: float = -2.0

    # Penalise roll/pitch angular velocity (wobbling)
    ang_vel_xy_penalty: float = -0.05

    # Penalise joint acceleration (smooth motion)
    joint_acc_penalty: float = -2.5e-7

    # Penalise large actions (energy efficiency)
    action_rate_penalty: float = -0.01

    # Penalise joint positions near limits (safety margin)
    joint_limit_penalty: float = -1.0

    # Penalise ground contact with anything but feet
    undesired_contact_penalty: float = -1.0

    # Penalise feet slipping
    feet_slip_penalty: float = -0.1


# ---------------------------------------------------------------------------
# Termination Configuration
# ---------------------------------------------------------------------------
@dataclass
class G1TerminationCfg:
    """Conditions that end an episode early."""

    # Fall detection: end if base height < threshold (G1 pelvis ~ 0.78m nominal)
    base_height_min: float = 0.35   # metres — below this = fallen

    # Max episode length
    max_episode_length_s: float = 20.0   # seconds


# ---------------------------------------------------------------------------
# Domain Randomisation Reference (used in domain_rand.py)
# ---------------------------------------------------------------------------
@dataclass
class G1DomainRandCfg:
    """Ranges for randomisation — applied at each episode reset."""

    # Rigid body mass perturbation (kg, additive delta, sampled uniformly)
    mass_delta_range: tuple = (-2.0, 2.0)

    # Joint friction scaling (multiplicative)
    joint_friction_range: tuple = (0.7, 1.3)

    # Motor strength scaling (multiplicative — simulates actuator wear)
    motor_strength_range: tuple = (0.9, 1.1)

    # Ground friction
    ground_friction_range: tuple = (0.3, 1.5)

    # Ground restitution (bounciness)
    ground_restitution_range: tuple = (0.0, 0.4)

    # Push perturbation on the base (for robustness to external forces)
    push_force_range_N: tuple = (0.0, 150.0)   # peak Newtons
    push_interval_s: float = 8.0               # average seconds between pushes


# ---------------------------------------------------------------------------
# Top-Level G1 Environment Config (passed to Isaac Lab)
# ---------------------------------------------------------------------------
@dataclass
class G1EnvCfg:
    """Master config object for the G1 RL training environment."""

    # Scene
    num_envs: int = 512
    env_spacing: float = 2.5        # metres between environment origins
    sim_dt: float = 0.005           # 200 Hz physics

    # Control
    decimation: int = 4             # Policy runs at 50 Hz (200 / 4)

    # Sub-configs
    observations: G1ObservationCfg = field(default_factory=G1ObservationCfg)
    actions: G1ActionCfg = field(default_factory=G1ActionCfg)
    rewards: G1RewardCfg = field(default_factory=G1RewardCfg)
    terminations: G1TerminationCfg = field(default_factory=G1TerminationCfg)
    domain_rand: G1DomainRandCfg = field(default_factory=G1DomainRandCfg)

    # Asset path — override with local URDF if downloaded
    robot_usd_path: str = "{ISAAC_ASSETS}/Robots/Unitree/G1/g1.usd"

    # Velocity command ranges for training
    lin_vel_x_range: tuple = (-1.0, 1.0)    # m/s
    lin_vel_y_range: tuple = (-0.5, 0.5)    # m/s
    ang_vel_yaw_range: tuple = (-1.0, 1.0)  # rad/s


# ---------------------------------------------------------------------------
# Default instance (import this in train_policy.py)
# ---------------------------------------------------------------------------
G1_ENV_CFG = G1EnvCfg()
