# Technical Explanation: Unitree G1 Humanoid Generalist

This document provides a deep dive into the technical architecture, algorithms, and integration strategies used in the "Humanoid Generalist" project for the Unitree G1.

---

## 1. System Overview

The project implements a **Sim-to-Real** pipeline designed to enable a Unitree G1 humanoid to perform both agile locomotion and complex manipulation. It bridges deep reinforcement learning (RL) for low-level control with vision-language-action models (VLA) for high-level reasoning.

### Key Technologies:
- **Simulation**: NVIDIA Isaac Sim 4.5.0 / Isaac Lab & MuJoCo (mjlab).
- **Communication**: ROS 2 Humble.
- **RL Framework**: RSL-RL (PPO implementation).
- **VLA Brain**: Unitree UnifoLM-VLA / NVIDIA GR00T-N1.6.
- **Localization**: Isaac ROS Visual SLAM.

---

## 2. Component Architecture

The system follows a modular, containerized architecture that decouples the "brain" from the "body."

### A. Whole-Body Reinforcement Learning (`isaac_lab/`)
The locomotion and whole-body coordination are trained using **Proximal Policy Optimization (PPO)** within Isaac Lab.
- **Observation Space (81 dimensions)**: Includes base linear/angular velocities, gravity vector, joint positions/velocities, previous actions, and navigation commands.
- **Action Space (23 dimensions)**: Target joint positions for all degrees of freedom (DoFs) of the G1.
- **Control Frequency**: 50 Hz policy decimation from a 200 Hz simulator.
- **Domain Randomization**: Crucial for sim-to-real transfer. We randomize body mass, joint friction, motor strength, and apply random push perturbations (up to 150N) during training to ensure robustness.

### B. Vision-Language-Action (VLA) Brain (`gr00t_model/`)
High-level task execution is handled by a VLA model (GR00T-N1.6 or UnifoLM-VLA).
- **Inference Loop**: Runs at ~5 Hz. It consumes RGB frames from the robot's cameras and natural language instructions.
- **Output**: Generates target poses (`target_xyz`, `target_rpy`) and gripper states.
- **Integration**: Exposed via a FastAPI server, allowing the ROS 2 bridge to query the brain asynchronously.

### C. ROS 2 Bridge & Safety Stack (`ros2_bridge/`)
This is the "nervous system" of the robot.
- **State Machine**: Manages transitions between `IDLE`, `LOCALIZING`, `NAVIGATING`, and `MANIPULATING`.
- **Safety Monitor (100 Hz)**: A critical watchdog that monitors joint limits, velocity spikes, and IMU tilt. It can trigger a hardware E-STOP if any safety boundary is breached.
- **Command Watchdog**: Prevents "runaway" behavior by stopping the robot if communication from the VLA or RL policy is lost for more than 500ms.

### D. Localization (`localization/`)
Uses the **Isaac ROS VSLAM** stack.
- **100 Hz Pose Estimation**: Provides high-frequency odometry by fusing visual features from RealSense cameras with IMU data.
- **Spatial Alignment**: Maps the robot's local coordinate system to the global mission space.

---

## 3. Data Integration Strategy

### UnifoLM-WBT-Dataset (Real-World Grounding)
The project integrates the **UnifoLM-WBT-Dataset**, which contains high-quality real-world whole-body teleoperation data.
1. **Dynamics Alignment**: Real-world torque and velocity distributions from this dataset are used to calibrate the domain randomization parameters in `isaac_lab/domain_rand.py`.
2. **Imitation Learning**: The RL policy is pre-trained or refined using behavior cloning (BC) on the WBT expert demonstrations to accelerate the learning of complex tasks like folding clothes or cleaning.
3. **VLA Fine-tuning**: The native 8B UnifoLM-VLA is fine-tuned on G1-specific manipulation sequences from the dataset to improve real-world spatial reasoning.

---

## 4. Development Workflow

### Simulation vs. Mocking
To enable rapid development without physical hardware:
1. **Isaac Lab**: Used for heavy RL training and high-fidelity physics validation.
2. **Mock Robot**: A lightweight ROS 2 node that simulates the G1 SDK. it provides fake joint states and IMU data, allowing full-stack validation of the bridge and brain.
3. **mjlab**: An alternative simulation path using MuJoCo-Warp for cross-verifying policy performance in a different physics engine.

### Deployment Path
1. **Validation**: All code is checked via `docker compose build` and syntax self-tests.
2. **Hardware Connection**: Once the G1 arrives, the `mock_robot` is replaced by the real Unitree SDK 2 bridge.
3. **Model Loading**: weights are pulled from Hugging Face, and `MOCK_MODE` is disabled.

---

## 5. Directory Structure Reference

- `/isaac_lab`: RL training logic and environment configs.
- `/gr00t_model`: VLA inference server and weight management.
- `/ros2_bridge`: Safety monitors and hardware abstraction.
- `/localization`: Camera and SLAM launch configurations.
- `/mock_robot`: Software-only G1 hardware simulator.
- `/tasks`: Detailed integration roadmaps.
