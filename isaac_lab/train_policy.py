# train_policy.py
# Whole-Body RL training for the Unitree G1 using Isaac Lab + RSL-RL (PPO).
#
# Pipeline:
#   1. Launch Isaac Sim (must happen before all other imports)
#   2. Build the G1-specific environment from g1_env_cfg.py
#   3. Apply domain randomisation on every episode reset
#   4. Train a PPO policy with RSL-RL
#   5. Save checkpoints to /workspace/isaac_lab/logs/<task>/
#
# Run inside the isaac-lab container:
#   docker compose run --rm isaac-lab \
#       python3 /workspace/isaac_lab/train_policy.py \
#       --num_envs 512 --max_iterations 3000

from __future__ import annotations
import argparse
import sys
import os

# ---------------------------------------------------------------------------
# 1. Isaac Sim MUST be launched before any omni.* imports
# ---------------------------------------------------------------------------
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError as e:
    print(f"[ERROR] Isaac Lab not found: {e}")
    print("        Make sure you are running inside the isaac-lab Docker container.")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Train Unitree G1 Whole-Body RL Policy")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-G1-v0",
    help="Isaac Lab task name. Falls back to generic humanoid if G1 task not registered.",
)
parser.add_argument("--num_envs",       type=int,   default=512,    help="Parallel environments")
parser.add_argument("--seed",           type=int,   default=42,     help="Random seed")
parser.add_argument("--max_iterations", type=int,   default=3000,   help="PPO iterations")
parser.add_argument("--checkpoint",     type=str,   default=None,   help="Resume from checkpoint path")
parser.add_argument("--no_domain_rand", action="store_true",        help="Disable domain randomization")
parser.add_argument("--log_interval",   type=int,   default=50,     help="Log stats every N iterations")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[INFO] Launching Isaac Sim (headless={args.headless})...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2. Safe to import omni / lab modules now
# ---------------------------------------------------------------------------
import gymnasium as gym
import torch

try:
    import omni.isaac.lab_tasks  # Registers all built-in tasks
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab_tasks.utils import parse_env_cfg
except ImportError as e:
    print(f"[ERROR] Isaac Lab tasks not found: {e}")
    simulation_app.close()
    sys.exit(1)

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("[ERROR] 'rsl_rl' not found. Install it: pip install rsl-rl")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Load G1-specific config
# ---------------------------------------------------------------------------
# g1_env_cfg is a pure-Python dataclass — safe to import after AppLauncher
sys.path.insert(0, os.path.dirname(__file__))
from g1_env_cfg import G1_ENV_CFG, G1EnvCfg

if not args.no_domain_rand:
    from domain_rand import apply_domain_randomization

# ---------------------------------------------------------------------------
# 4. RSL-RL PPO Hyperparameters (tuned for G1 locomotion)
# ---------------------------------------------------------------------------
PPO_CONFIG = {
    "seed": args.seed,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "policy": {
        "init_noise_std": 1.0,
        # Larger networks for 81-dim obs → 23-dim action
        "actor_hidden_dims":  [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "clip_param":           0.2,
        "desired_kl":           0.01,
        "entropy_coef":         0.01,
        "gamma":                0.99,
        "lam":                  0.95,
        "learning_rate":        1.0e-3,
        "max_grad_norm":        1.0,
        "num_learning_epochs":  5,
        "num_mini_batches":     4,
        "schedule":             "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef":      1.0,
    },
    "runner": {
        "policy_class_name":     "ActorCritic",
        "algorithm_class_name":  "PPO",
        "num_steps_per_env":     24,    # Rollout length (steps at 50Hz policy = 0.48s)
        "max_iterations":        args.max_iterations,
        "save_interval":         200,   # Save checkpoint every N iterations
        "experiment_name":       f"g1_{args.task}",
        "run_name":              "",
        "resume":                args.checkpoint is not None,
        "load_run":              args.checkpoint or "",
        "checkpoint":            -1,
    },
}


# ---------------------------------------------------------------------------
# 5. Build Environment
# ---------------------------------------------------------------------------
def build_env() -> gym.Env:
    """
    Attempt to create the G1-specific Isaac Lab task.
    Falls back to the generic humanoid task if the G1 task is not yet
    registered (e.g., running with an older Isaac Lab version).
    """
    # Override num_envs from CLI
    G1_ENV_CFG.num_envs = args.num_envs

    try:
        print(f"[INFO] Creating task: {args.task}")
        env_cfg = parse_env_cfg(args.task, use_gpu=not args.cpu, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        print(f"[INFO] G1 task '{args.task}' created successfully.")
        return env
    except Exception as e:
        print(f"[WARN] G1 task failed: {e}")
        print("[WARN] Falling back to generic 'Isaac-Humanoid-v0'...")
        print("[WARN] Domain randomization will still be applied as a wrapper.")
        env_cfg = parse_env_cfg("Isaac-Humanoid-v0", use_gpu=not args.cpu, num_envs=args.num_envs)
        env = gym.make("Isaac-Humanoid-v0", cfg=env_cfg)
        return env


# ---------------------------------------------------------------------------
# 6. Domain Randomization wrapper
# ---------------------------------------------------------------------------
class DomainRandWrapper(gym.Wrapper):
    """
    Wraps any Isaac Lab environment to inject domain randomization
    on each episode reset. Transparent to the RSL-RL runner.
    """

    def __init__(self, env: gym.Env, dr_cfg: "G1EnvCfg"):
        super().__init__(env)
        self.dr_cfg = dr_cfg
        print("[INFO] DomainRandWrapper active — randomizing mass, friction, motors, ground.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            apply_domain_randomization(self.env.unwrapped, self.dr_cfg.domain_rand)
        except Exception as e:
            print(f"[WARN] Domain randomization failed at reset: {e}")
        return obs, info

    def step(self, action):
        # Apply push perturbation every step
        try:
            from domain_rand import apply_push_perturbation
            t = getattr(self.env.unwrapped, "episode_length_buf", None)
            current_time = (t.float().mean().item() * G1_ENV_CFG.sim_dt) if t is not None else 0.0
            apply_push_perturbation(self.env.unwrapped, self.dr_cfg.domain_rand, current_time)
        except Exception:
            pass

        # Scale actions by per-env motor strength (if DR applied)
        strength = getattr(self.env.unwrapped, "motor_strength_scale", None)
        if strength is not None and isinstance(action, torch.Tensor):
            action = action * strength

        return self.env.step(action)


# ---------------------------------------------------------------------------
# 7. Main Training Loop
# ---------------------------------------------------------------------------
def main():
    log_dir = os.path.join("/workspace/isaac_lab/logs", args.task.replace("-", "_"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logs → {log_dir}")
    print(f"[INFO] Seed={args.seed} | Envs={args.num_envs} | Iterations={args.max_iterations}")
    print(f"[INFO] Domain Rand={'DISABLED' if args.no_domain_rand else 'ENABLED'}")

    env = build_env()

    if not args.no_domain_rand:
        env = DomainRandWrapper(env, G1_ENV_CFG)

    runner = OnPolicyRunner(env, PPO_CONFIG, log_dir=log_dir, device=PPO_CONFIG["device"])

    if args.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)

    print(f"\n{'='*60}")
    print(f"  Starting G1 Whole-Body RL Training (PPO)")
    print(f"  Task       : {args.task}")
    print(f"  Obs dims   : {G1_ENV_CFG.observations.total_dims}")
    print(f"  Action dims: {G1_ENV_CFG.actions.num_actions}")
    print(f"  Policy Hz  : {1.0 / (G1_ENV_CFG.sim_dt * G1_ENV_CFG.decimation):.0f}")
    print(f"{'='*60}\n")

    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted — saving final checkpoint...")
        runner.save(os.path.join(log_dir, "interrupted_checkpoint.pt"))

    print("[INFO] Training complete.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
