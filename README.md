# NVIDIA Isaac GR00T N1.6 Sim-to-Real Implementation

This project implements a complete Sim-to-Real workflow for humanoid robots, replicating the architecture described in the NVIDIA Isaac GR00T N1.6 technical blog. It integrates high-fidelity simulation, synthetic data generation, and vision-based localization into a containerized pipeline.

## Project Architecture

The system is composed of four primary components:

1.  **Whole-Body Reinforcement Learning**: Leveraging **Isaac Lab** (based on Isaac Sim 4.5.0) to train robust locomotion and manipulation policies using the RSL-RL library.
2.  **Synthetic Navigation Data (COMPASS)**: A procedural data generation pipeline in Isaac Sim to create diverse navigation datasets for training COMPASS policies.
3.  **Vision-Based Localization**: A strictly versioned **Isaac ROS** stack (ROS 2 Humble) running Visual SLAM (VSLAM) for high-accuracy estimation in real-world environments.
4.  **VLA Model Integration**: A runtime interface for the **GR00T N1.6** Vision-Language-Action model to perform high-level reasoning and instruction following.

## Getting Started

This project is fully containerized. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### 1. Build the Environment
```bash
docker compose build
```

### 2. Run Components

**Train Whole-Body Policy (Isaac Lab)**
```bash
docker compose run --rm isaac-lab python3 /workspace/isaac_lab/train_policy.py
```

**Generate Navigation Data (COMPASS)**
```bash
docker compose run --rm isaac-lab python3 /workspace/compass_nav/generate_data.py
```

**Launch Localization Stack**
```bash
docker compose up localization
```

## Structure

- `isaac_lab/`: Policy training scripts and environment configurations.
- `compass_nav/`: Synthetic data generation logic.
- `localization/`: ROS 2 launch files and configuration for VSLAM.
- `gr00t_model/`: Inference interface for the VLA model.
- `docker/`: Dockerfiles pinning specific Isaac Sim and ROS 2 versions for reproducibility.

## Requirements

- Linux (Ubuntu 22.04 recommended)
- NVIDIA GPU (RTX series recommended for Isaac Sim)
- Docker & Docker Compose
