# Walkthrough: Unitree G1 Humanoid Generalist Setup

I have initialized the project structure for replicating the NVIDIA Isaac GR00T workflow, enhanced with **Unitree's UnifoLM** architecture and **mjlab** for the **Unitree G1**, fully containerized with Docker.

## Directory Structure

The project is located at `/home/kwalker96/humanoid_generalist` and organized as follows:

```
humanoid_generalist/
├── docker/                 # Docker configuration
│   ├── Dockerfile.isaac_lab
│   └── Dockerfile.localization
├── isaac_lab/              # Whole-Body RL & mjlab
│   └── train_policy.py     # Base script for PPO training
├── compass_nav/            # Synthetic Data Generation & UnifoLM-WMA
│   └── generate_data.py
├── localization/           # Vision-Based Localization (ROS 2)
│   └── localize.launch.py
├── gr00t_model/            # VLA Integration (UnifoLM-VLA / GR00T)
│   └── run_inference.py
├── docker-compose.yml      # Orchestration
└── README.md               # Documentation
```

## Docker Services

The `docker-compose.yml` defines three main services:

- **isaac-lab**: For running Whole-Body RL training (Isaac Sim/mjlab) and data generation.
    - Base Image: `nvcr.io/nvidia/isaac-sim:4.5.0`
- **localization**: For running the Isaac ROS visual SLAM and localization stack.
    - Base Image: `nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble_f247dd1051869171c3fc53bb35f6b907`
    - Package: `ros-humble-isaac-ros-visual-slam` 
- **gr00t**: For running the VLA model inference.

### 1. Whole-Body RL (Isaac Lab / mjlab)
- **Script**: `isaac_lab/train_policy.py`
- **Architecture**: Capable of using `omni.isaac.lab.app.AppLauncher` for Isaac Sim, but will be extended to support the MuJoCo-backed `mjlab` API for better Sim-to-Real transfer on the G1.
- **Run Command**: `docker compose run --rm isaac-lab python3 /workspace/isaac_lab/train_policy.py`

### 2. Synthetic Data & World Models (WMA/COMPASS)
- **Script**: `compass_nav/generate_data.py`
- **Architecture**: A procedural generation script designed to be enhanced by the `UnifoLM-WMA-0` Simulation Engine. Generates navigation metadata and expert trajectories.
- **Run Command**: `docker compose run --rm isaac-lab python3 /workspace/compass_nav/generate_data.py`

### 3. Visual Localization (Isaac ROS)
- **Launch File**: `localization/localize.launch.py`
- **Architecture**: Configures the `isaac_ros_visual_slam` ComposableNode to provide $100\text{Hz}$ pose estimation from real Realsense/stereo cameras.
- **Run Command**: `docker compose up localization`

### 4. VLA Integration (UnifoLM-VLA)
- **Script**: `gr00t_model/run_inference.py`
- **Architecture**: Python-based inference loop designed for HuggingFace transformers. Configured to load Unitree's native `UnifoLM-VLA-Base` (8B), avoiding the friction of generic GR00T models. Process multimodal inputs (Image+Text) and outputs spatial actions.
- **Run Command**: `docker compose run --rm gr00t python3 /workspace/gr00t_model/run_inference.py`

### 5. System Architecture

#### Technical Dependency Explanation

1.  **Isaac Lab / mjlab (RL Base)**:
    - **role**: It is the "muscle memory" factory. It runs offline to produce the **Whole-Body Policy** that converts high-level commands (velocity) into low-level motor torques. `mjlab` ensures MuJoCo physics fidelity for the G1.

2.  **UnifoLM-WMA / COMPASS (Navigation Data)**:
    - **role**: It generates massive synthetic datasets to train a robust Navigation Policy. This policy handles "getting from A to B" without colliding. Can leverage the `UnifoLM_WBT_Dataset`.

3.  **Isaac ROS (Visual Perception)**:
    - **role**: The "eyes" and "inner ear". It processes raw camera streams to tell the robot *where it is* (Pose), decoupling hardware drivers from pure inference.

4.  **UnifoLM-VLA (High-Level Brain)**:
    - **role**: The "commander". It takes multimodal input (Language + Vision) and outputs high-level intents or spatial manipulation targets. Naturally optimized for legged systems.

### 6. Next Steps (Algorithm Development)

1.  **Clone `mjlab` Toolchain**:
    - Update the learning environment to support MuJoCo backends via the `mjlab` API.
2.  **Download UnifoLM Models (`gr00t_model`)**:
    - Update `gr00t_model/run_inference.py` to point to `unitreerobotics/UnifoLM-VLA-Base` on Hugging Face.
3.  **Real-World Prep**:
    - Build drivers to map the G1's raw sensor inputs to ROS 2 topics matching `isaac_ros_visual_slam`.
