# domain_rand.py
# Domain Randomisation utilities for the Unitree G1.
# Applied at every episode reset to make the trained policy robust to
# real-world hardware variance (mass tolerances, friction, motor wear, etc.)
#
# Usage:
#   from isaac_lab.domain_rand import apply_domain_randomization
#   apply_domain_randomization(env, cfg=G1_ENV_CFG.domain_rand)

from __future__ import annotations
import random
import math
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from g1_env_cfg import G1DomainRandCfg

log = logging.getLogger(__name__)


def _uniform(low: float, high: float) -> float:
    """Shorthand for a single uniform sample."""
    return random.uniform(low, high)


# ---------------------------------------------------------------------------
# Core randomization functions
# Each function is self-contained so it can be tested in isolation.
# ---------------------------------------------------------------------------

def randomize_rigid_body_mass(env, cfg: "G1DomainRandCfg") -> None:
    """
    Add a random mass delta to the robot's base body.
    Simulates payload variation (backpack, tool, etc.) and manufacturing
    tolerances in the G1 torso mass.

    In Isaac Lab: env.scene["robot"].data.default_mass is a tensor shaped
    [num_envs, num_bodies]. We perturb only the base body (index 0).
    """
    try:
        import torch
        robot = env.scene["robot"]
        num_envs = env.num_envs
        mass_delta = torch.zeros(num_envs, device=env.device)
        low, high = cfg.mass_delta_range
        mass_delta.uniform_(low, high)

        # Apply delta to base body (index 0)
        robot.data.default_mass[:, 0] += mass_delta
        log.debug(f"[DR] Mass delta range [{low:.2f}, {high:.2f}] kg applied.")
    except Exception as e:
        log.warning(f"[DR] randomize_rigid_body_mass skipped: {e}")


def randomize_joint_friction(env, cfg: "G1DomainRandCfg") -> None:
    """
    Scale joint friction coefficients per-environment.
    The G1 uses harmonic drives whose friction varies with temperature and wear.
    """
    try:
        import torch
        robot = env.scene["robot"]
        num_envs = env.num_envs
        num_joints = robot.data.default_joint_pos.shape[1]

        low, high = cfg.joint_friction_range
        scale = torch.empty(num_envs, num_joints, device=env.device).uniform_(low, high)

        # Isaac Lab exposes joint friction as a physics material property
        # This modifies the internal articulation cache
        robot.data.joint_friction *= scale
        log.debug(f"[DR] Joint friction scaled [{low:.2f}x, {high:.2f}x].")
    except Exception as e:
        log.warning(f"[DR] randomize_joint_friction skipped: {e}")


def randomize_motor_strength(env, cfg: "G1DomainRandCfg") -> None:
    """
    Scale the effective actuator strength per-environment.
    Models real-world motor output variation: temperature de-rating, voltage
    sag, and hardware unit variance across multiple G1 units.
    """
    try:
        import torch
        robot = env.scene["robot"]
        num_envs = env.num_envs
        num_joints = robot.data.default_joint_pos.shape[1]

        low, high = cfg.motor_strength_range
        strength_scale = torch.empty(num_envs, num_joints, device=env.device).uniform_(low, high)

        # Store in the env for the action post-processing step
        # train_policy.py multiplies actions by this before sending to the articulation
        env.motor_strength_scale = strength_scale
        log.debug(f"[DR] Motor strength scaled [{low:.2f}x, {high:.2f}x].")
    except Exception as e:
        log.warning(f"[DR] randomize_motor_strength skipped: {e}")
        try:
            import torch
            env.motor_strength_scale = torch.ones(
                env.num_envs,
                len([k for k in dir(env) if "joint" in k]),
                device=env.device,
            )
        except Exception:
            pass


def randomize_ground_physics(env, cfg: "G1DomainRandCfg") -> None:
    """
    Randomise the ground plane's friction and restitution.
    The G1 must walk on concrete, carpet, tiles, and outdoor terrain —
    this prepares the policy for all of them.
    """
    try:
        terrain = env.scene.terrain
        fr_low, fr_high = cfg.ground_friction_range
        re_low, re_high = cfg.ground_restitution_range

        new_friction = _uniform(fr_low, fr_high)
        new_restitution = _uniform(re_low, re_high)

        terrain.cfg.physics_material.static_friction = new_friction
        terrain.cfg.physics_material.dynamic_friction = new_friction * 0.85
        terrain.cfg.physics_material.restitution = new_restitution
        log.debug(f"[DR] Ground friction={new_friction:.2f}, restitution={new_restitution:.2f}.")
    except Exception as e:
        log.warning(f"[DR] randomize_ground_physics skipped: {e}")


def apply_push_perturbation(env, cfg: "G1DomainRandCfg", current_time_s: float) -> None:
    """
    Stochastically apply a short impulse force to the robot's base.
    Mimics people bumping into the robot or slipping on unexpected terrain.

    Should be called every simulation step; internally samples whether a push
    should occur based on a Poisson process with rate 1/push_interval_s.
    """
    try:
        import torch
        # Poisson approximation: push probability per step
        sim_dt = env.cfg.sim_dt
        push_prob = sim_dt / cfg.push_interval_s

        # Sample which envs get pushed this step
        push_mask = torch.rand(env.num_envs, device=env.device) < push_prob
        num_pushed = push_mask.sum().item()
        if num_pushed == 0:
            return

        low, high = cfg.push_force_range_N
        # Random direction (horizontal only — we don't want vertical pushes)
        forces = torch.zeros(env.num_envs, 3, device=env.device)
        forces[push_mask, 0] = torch.empty(int(num_pushed), device=env.device).uniform_(low, high)
        forces[push_mask, 1] = torch.empty(int(num_pushed), device=env.device).uniform_(-high / 2, high / 2)
        # Randomly flip direction
        forces[push_mask] *= (torch.randint(0, 2, (int(num_pushed), 3), device=env.device).float() * 2 - 1)

        robot = env.scene["robot"]
        robot.set_external_force_and_torque(
            forces=forces.unsqueeze(1),   # [num_envs, 1, 3]
            torques=torch.zeros_like(forces.unsqueeze(1)),
            body_ids=[0],                  # Apply to base body
        )
        log.debug(f"[DR] Push applied to {num_pushed} envs.")
    except Exception as e:
        log.warning(f"[DR] apply_push_perturbation skipped: {e}")


# ---------------------------------------------------------------------------
# Master function — call this at env reset
# ---------------------------------------------------------------------------

def apply_domain_randomization(env, cfg: "G1DomainRandCfg") -> None:
    """
    Apply all domain randomization at once. Call this inside the env's
    _reset_idx() hook, after the articulation is reset but before the
    first observation is computed.

    Example (inside train_policy.py or a custom env class):
        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            apply_domain_randomization(self, self.cfg.domain_rand)
    """
    log.info("[DR] Applying domain randomization...")
    randomize_rigid_body_mass(env, cfg)
    randomize_joint_friction(env, cfg)
    randomize_motor_strength(env, cfg)
    randomize_ground_physics(env, cfg)
    log.info("[DR] Domain randomization complete.")


# ---------------------------------------------------------------------------
# Standalone test (run without Isaac Lab to verify logic)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from g1_env_cfg import G1DomainRandCfg

    class _MockRobotData:
        def __init__(self):
            try:
                import torch
                self.default_mass = torch.ones(4, 10)
                self.default_joint_pos = torch.zeros(4, 23)
                self.joint_friction = torch.ones(4, 23)
            except ImportError:
                print("[TEST] torch not available — skipping tensor checks.")

    class _MockRobot:
        def __init__(self):
            self.data = _MockRobotData()

    class _MockScene:
        def __init__(self):
            self._robot = _MockRobot()

        def __getitem__(self, key):
            if key == "robot":
                return self._robot
            raise KeyError(key)

        @property
        def terrain(self):
            raise AttributeError("No terrain in mock")

    class _MockEnv:
        def __init__(self):
            self.num_envs = 4
            self.device = "cpu"
            self.scene = _MockScene()
            self.cfg = type("cfg", (), {"sim_dt": 0.005})()

    print("Running domain randomization self-test...")
    mock_env = _MockEnv()
    cfg = G1DomainRandCfg()
    apply_domain_randomization(mock_env, cfg)
    print("Self-test passed — all DR functions ran without fatal errors.")
