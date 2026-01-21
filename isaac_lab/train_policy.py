# train_policy.py
import argparse
import sys
import os

# 1. Launch Isaac Sim First (Must happen before other imports)
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError as e:
    print(f"Error importing Isaac Lab: {e}")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Train Whole-Body RL Policy with Isaac Lab")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v1", help="Name of the task to train")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless if not specified, to save VRAM
# args.headless = True 

print(f"[INFO] Launching Isaac Sim app (Headless: {args.headless})...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. Import Logic Packages (Safe now)
import gymnasium as gym
import torch
import omni.isaac.lab_tasks  # Register all tasks
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils import parse_env_cfg

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("[ERROR] 'rsl_rl' library not found. Please ensure it is installed in the container.")
    simulation_app.close()
    sys.exit(1)

def main():
    print(f"[INFO] Setting up task: {args.task}")
    
    # Load Environment Config
    try:
        env_cfg = parse_env_cfg(args.task, use_gpu=not args.cpu, num_envs=args.num_envs)
        # Create Environment
        env = gym.make(args.task, cfg=env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create environment '{args.task}': {e}")
        print("Falling back to 'Isaac-Humanoid-v0' just in case...")
        env_cfg = parse_env_cfg("Isaac-Humanoid-v0", use_gpu=not args.cpu, num_envs=args.num_envs)
        env = gym.make("Isaac-Humanoid-v0", cfg=env_cfg)

    # Basic PPO Configuration (Standard for Humanoids)
    # Usually this is loaded from a yaml, but we define a robust default here.
    ppo_config = {
        "seed": args.seed,
        "device": "cuda", # Assuming GPU
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        }
    }
    
    log_dir = os.path.join("/workspace/isaac_lab/logs", args.task)
    print(f"[INFO] Logs will be saved to: {log_dir}")

    # Initialize RSL-RL Runner
    runner = OnPolicyRunner(env, ppo_config, log_dir=log_dir, device="cuda")
    
    print("[INFO] Starting REAL Training (PPO)...")
    print(f"[INFO] Max Iterations: {args.max_iterations}")

    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
    
    print("[INFO] Training complete. Saving final model...")
    # RSL-RL saves checkpoints automatically, but we ensure cleanliness
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
