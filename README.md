# Unitree G1 Humanoid Generalist: Sim-to-Real Implementation

This project implements a complete Sim-to-Real workflow for the Unitree G1 humanoid robot. It bridges the NVIDIA Isaac GR00T framework with Unitree's native **UnifoLM** (Universal Foundation Model for Legged Manipulation) and utilizes both **Isaac Lab** and **mjlab** (MuJoCo) for robust policy training.

## Project Architecture

The system is composed of four primary components:

1.  **Whole-Body Reinforcement Learning (Isaac/mjlab)**: Training robust locomotion and manipulation policies. We support both **Isaac Lab** (Isaac Sim 4.5.0) and **mjlab** (MuJoCo-Warp) to tackle Sim-to-Real domain gaps effectively.
2.  **World-Model-Action & Navigation (UnifoLM-WMA / COMPASS)**: A hybrid procedural data generation pipeline and world model for creating diverse navigation datasets for training policies.
3.  **Vision-Language-Action (UnifoLM-VLA)**: A runtime interface for Unitree's native 8B parameter `UnifoLM-VLA-Base` model, replacing generic VLA models to perform spatial-semantic reasoning optimized for the G1 hardware.
4.  **Vision-Based Localization (Isaac ROS)**: A strictly versioned **Isaac ROS** stack (ROS 2 Humble) running Visual SLAM (VSLAM) for high-accuracy estimation ($100\text{Hz}$ pose) in real-world environments.

## Getting Started

This project is fully containerized. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### 1. Build the Environment
```bash
docker compose build
```

### 2. Run Components

**Train Whole-Body Policy**
```bash
# Using Isaac Lab
docker compose run --rm isaac-lab python3 /workspace/isaac_lab/train_policy.py
```
*(Note: mjlab integration scripts are under active development in the same directory.)*

**Generate Synthetic Data**
```bash
docker compose run --rm isaac-lab python3 /workspace/compass_nav/generate_data.py
```

**Launch Localization Stack**
```bash
docker compose up localization
```

**Run VLA Inference Server**
```bash
docker compose run --rm gr00t python3 /workspace/gr00t_model/run_inference.py
```

## Structure

- `isaac_lab/`: Policy training scripts and environment configurations (Isaac Lab & mjlab).
- `compass_nav/`: Synthetic data generation logic (WMA/COMPASS).
- `localization/`: ROS 2 launch files and configuration for Visual SLAM.
- `gr00t_model/`: Inference interface for the UnifoLM / GR00T VLA models.
- `docker/`: Dockerfiles pinning specific simulation and ROS 2 versions for reproducibility.

## Requirements

- Linux (Ubuntu 22.04 recommended)
- NVIDIA GPU (RTX series recommended)
- Docker & Docker Compose
