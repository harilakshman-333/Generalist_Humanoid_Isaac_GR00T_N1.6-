# Walkthrough: Humanoid Generalist Project Setup

I have initialized the project structure for replicating the NVIDIA Isaac GR00T N1.6 Sim-to-Real workflow, fully containerized with Docker.

## Directory Structure

The project is located at `/home/kwalker96/humanoid_generalist` and organized as follows:

```
humanoid_generalist/
├── docker/                 # Docker configuration
│   ├── Dockerfile.isaac_lab
│   └── Dockerfile.localization
├── isaac_lab/              # Whole-Body RL & COMPASS
│   └── train_policy.py
├── compass_nav/            # Synthetic Data Generation
│   └── generate_data.py
├── localization/           # Vision-Based Localization (ROS 2)
│   └── localize.launch.py
├── gr00t_model/            # VLA Integration
│   └── run_inference.py
├── docker-compose.yml      # Orchestration
└── README.md               # Documentation
```

## Docker Services

The `docker-compose.yml` defines three main services:

- **isaac-lab**: For running Whole-Body RL training and COMPASS data generation.
    - Base Image: `nvcr.io/nvidia/isaac-sim:4.5.0`
- **localization**: For running the Isaac ROS visual SLAM and localization stack.
    - Base Image: `nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble_f247dd1051869171c3fc53bb35f6b907`
    - Package: `ros-humble-isaac-ros-visual-slam` (replaces visual-localization)
- **gr00t**: For running the VLA model inference.

### 1. Whole-Body RL (Isaac Lab)
- **Script**: `isaac_lab/train_policy.py`
- **Architecture**: Uses `omni.isaac.lab.app.AppLauncher` to bootstrap the simulation context in headless mode. Configured to load the `Isaac-Velocity-Flat-G1-v1` task environment.
- **Run Command**: `docker compose run --rm isaac-lab python3 /workspace/isaac_lab/train_policy.py`

### 2. Synthetic Data (COMPASS)
- **Script**: `compass_nav/generate_data.py`
- **Architecture**: Procedural generation script leveraging Isaac Sim's Python API. Generates navigation metadata and is extendable to save full sensor trajectories.
- **Run Command**: `docker compose run --rm isaac-lab python3 /workspace/compass_nav/generate_data.py`

### 3. Visual Localization (Isaac ROS)
- **Launch File**: `localization/localize.launch.py`
- **Architecture**: Configures the `isaac_ros_visual_slam` ComposableNode.
    - **Parameters**: `enable_rectified_pose=True`, `base_frame='base_link'`, `enable_slam_visualization=True`.
    - **Remappings**: Maps `stereo_camera/left/image_rect` -> `/camera/left/image_rect` (standard ROS 2 topics).
- **Run Command**: `docker compose up localization`

### 4. GR00T VLA Integration
- **Script**: `gr00t_model/run_inference.py`
- **Architecture**: Python-based inference loop designed for HuggingFace transformers. Includes structure for loading `nvidia/GR00T-N1.6-3B` and processing multimodal (Image+Text) inputs.
- **Run Command**: `docker compose run --rm gr00t python3 /workspace/gr00t_model/run_inference.py`

### 5. System Architecture


![System Architecture - Isaac GR00T N1.6 Sim-to-Real Pipeline](/home/kwalker96/.gemini/antigravity/brain/5ed2ace2-32aa-4536-8387-62957c08d313/isaac_gr00t_architecture_1768653237064.png)

#### Technical Dependency Explanation

1.  **Isaac Lab (RL Base)**:
    - **dependency**: Relies on **Isaac Sim 4.5.0** (PhysX 5.0) for high-fidelity physics.
    - **role**: It is the "muscle memory" factory. It runs offline to produce the **Whole-Body Policy** (a neural network) that converts high-level commands (velocity) into low-level motor torques.

2.  **COMPASS (Navigation Data)**:
    - **dependency**: Relies on **Isaac Sim** for procedural environment generation.
    - **role**: It generates massive synthetic datasets to train a robust **Navigation Policy**. This policy handles "getting from A to B" without colliding, feeding velocity commands to the Whole-Body Policy.

3.  **Isaac ROS (Visual Perception)**:
    - **dependency**: Relies on **CUDA-X** and **ROS 2 Humble**. Strictly decoupled from the simulation Python environment to avoid dependency hell.
    - **role**: The "eyes" and "inner ear". It processes raw camera streams to tell the robot *where it is* (Pose) and *what is around it* (Occupancy Map), which is critical input for the Navigation Policy.

4.  **GR00T (High-Level Brain)**:
    - **dependency**: Relies on **PyTorch** and **Transformers**.
    - **role**: The "commander". It takes multimodal input (Language + Vision) and outputs the high-level intent (e.g., "Go to the kitchen"). It doesn't drive the motors directly; it directs the Navigation Stack.

### 6. Next Steps (Algorithm Development)

Now that the infrastructure is set up, your focus shifts from "Engineer" to "Researcher".

1.  **RL Policy Training (`isaac_lab`)**:
    - Edit `isaac_lab/train_policy.py` to import your specific robot asset (USD file).
    - Design your Reward Function (e.g., +1 for upright, -5 for falling).
    - Run training overnight to generate `policy.pt`.

2.  **Navigation Dataset (`compass_nav`)**:
    - Extend `generate_data.py` to save RGB/Depth images and occupancy maps.
    - Generate 100k+ frames of navigation data.
    - Train your Navigation Policy using this data.

3.  **Real-World Prep**:
    - Ensure your physical robot has the drivers installed to publish `/camera/left/image_rect` and `/camera/right/image_rect`.
    - Apply for access to the `nvidia/GR00T-N1.6-3B` weights on HuggingFace.
