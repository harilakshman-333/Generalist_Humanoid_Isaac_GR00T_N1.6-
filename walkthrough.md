# Walkthrough: Unitree G1 Humanoid Generalist — Pre-Robot Build-Out

## What Was Built

This walkthrough documents the full software stack implemented *before* connecting to the physical Unitree G1. Every component can be validated in a software-only dry run.

---

## Directory Structure

```
humanoid_generalist/
├── docker/
│   ├── Dockerfile.isaac_lab        # Isaac Sim 4.5.0 + Isaac Lab + RSL-RL
│   ├── Dockerfile.localization     # Isaac ROS Humble + Visual SLAM
│   ├── Dockerfile.gr00t            # PyTorch + HuggingFace VLA inference
│   ├── Dockerfile.ros2_bridge      # ROS 2 Humble bridge + safety monitor
│   └── ros2_bridge_entrypoint.sh   # Sources ROS 2 workspace at container start
├── isaac_lab/
│   ├── train_policy.py             # PPO training loop (G1-specific)
│   ├── g1_env_cfg.py               # G1 23-DoF environment configuration
│   └── domain_rand.py              # Domain randomization (mass/friction/motor/push)
├── compass_nav/
│   └── generate_data.py            # Synthetic navigation data generation
├── localization/
│   └── localize.launch.py          # Isaac ROS VSLAM launch (100 Hz pose)
├── gr00t_model/
│   ├── run_inference.py            # VLA inference loop + ROS 2 publisher
│   └── vla_server.py               # FastAPI HTTP server for VLA inference
├── ros2_bridge/
│   ├── g1_bridge_node.py           # State machine + navigation controller
│   ├── safety_monitor.py           # 100 Hz hardware limit watchdog
│   └── ros2_bridge.launch.py       # Launches bridge + safety + VLA together
├── mock_robot/
│   ├── mock_g1_node.py             # Full software G1 hardware simulator
│   └── logs/                       # Command logs (JSONL, auto-created)
├── docker-compose.yml              # Five-service orchestration
└── README.md
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ROS 2 Network (ros2_net)              │
│                                                             │
│  ┌──────────────┐    /vla/target_pose    ┌───────────────┐  │
│  │  gr00t       │──────────────────────▶│ ros2_bridge   │  │
│  │  (VLA Brain) │                        │  (Bridge Node)│  │
│  │  5 Hz        │◀── /camera/left/image  │  50 Hz ctrl  │  │
│  └──────────────┘                        └───────┬───────┘  │
│                                                  │          │
│  ┌──────────────┐  /visual_slam/odometry         │          │
│  │ localization │──────────────────────────────▶│          │
│  │  (VSLAM)     │                                │          │
│  │  100 Hz      │                        /g1/cmd_vel        │
│  └──────────────┘                        /g1/joint_commands │
│                                                  │          │
│  ┌──────────────┐    /g1/joint_states   ┌────────▼───────┐  │
│  │ mock_robot   │◀──────────────────────│ safety_monitor │  │
│  │ (Mock G1)    │──────────────────────▶│  100 Hz        │  │
│  │  50 Hz sim   │    /g1/imu            └───────┬───────┘  │
│  └──────────────┘    /g1/e_stop ◀───────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Whole-Body RL (`isaac_lab/`)

| Item | Value |
|---|---|
| Algorithm | PPO (RSL-RL) |
| Observation dims | 81 (vel × 2, gravity, cmd × 2, joint pos+vel, last action) |
| Action dims | 23 (one per G1 joint) |
| Policy frequency | 50 Hz (200 Hz sim ÷ decimation 4) |
| Training envs | 512 parallel |

**Domain Randomization applied at every reset:**
- Body mass ± 2 kg
- Joint friction × [0.7, 1.3]
- Motor strength × [0.9, 1.1]
- Ground friction [0.3, 1.5]
- Random push forces up to 150 N every ~8 s

### 2. Dataset-Guided Alignment (`UnifoLM-WBT`)

The project leverages the **Unitree UnifoLM-WBT-Dataset** (released March 2026) to ground simulation results in real-world humanoid mechanics.

| Use Case | Implementation |
|---|---|
| **Expert Demos** | Using real WBT trajectories as target joint states for Imitation Learning. |
| **VLA Fine-tuning** | Refining the 8B `UnifoLM-VLA` on G1-specific manipulation data. |
| **Dynamics Tuning**| Refining `domain_rand.py` coefficients using real hardware torque/velocity distributions. |

### 3. VLA Inference (`gr00t_model/`)

| Item | Value |
|---|---|
| Model | `nvidia/GR00T-N1.6-3B` (or UnifoLM-VLA-Base) |
| Inference rate | 5 Hz |
| Input | RGB frame (640×480) + natural language instruction |
| Output | `{action_type, target_xyz, target_rpy, gripper_open}` |
| Fallback | Mock mode — synthetic output, no weights needed |
| HTTP API | `POST localhost:8000/infer` |

### 3. ROS 2 Bridge (`ros2_bridge/`)

**State Machine:**
```
IDLE → LOCALIZING → NAVIGATING → MANIPULATING
                       ↑               |
                       └───────────────┘
                    (all states) → E_STOP
```

**Safety Monitor checks (100 Hz):**
- Joint position limits (warning at 80%, E-STOP at 95% of hardware limit)
- Joint velocity > 10 rad/s → E-STOP
- Command watchdog: 500 ms silence while moving → E-STOP
- IMU tilt > 35° → E-STOP

### 4. Mock Robot (`mock_robot/`)

Simulates the full G1 SDK interface:
- 23 joint positions with Gaussian noise (σ = 0.002 rad)
- IMU: gravity vector + gyro drift
- Forward camera: synthetic gradient images at 30 Hz
- Odometry: 2D unicycle integration with momentum smoothing
- Logs all received commands to `mock_robot/logs/*.jsonl`

---

## Running the Dry-Run (No Hardware)

```bash
# Terminal 1: Start mock robot + bridge
docker compose up ros2_bridge mock_robot

# Terminal 2: Inject a fake VLA target (navigate 1 m forward)
docker exec -it <bridge_container> bash
ros2 topic pub --once /vla/target_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}}}"

# Terminal 3: Observe outputs
ros2 topic echo /g1/cmd_vel
ros2 topic echo /g1/safety_status
ros2 topic echo /g1/state

# Trigger E-STOP manually
ros2 topic pub --once /g1/e_stop std_msgs/Bool "{data: true}"
```

---

## Running VLA Inference (HTTP API, no ROS needed)

```bash
# Start VLA server in mock mode
docker compose run --rm gr00t \
  python3 /workspace/gr00t_model/vla_server.py --mock

# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Test inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Pick up the red block", "image_b64": ""}' \
  | python3 -m json.tool
```

---

## What's Next (When Robot Arrives)

| Step | Action |
|---|---|
| 1 | Download `UnifoLM-WBT-Dataset` from Hugging Face for G1-specific motion priors |
| 2 | Fine-tune `UnifoLM-VLA` using real-world manipulation clips (folding clothes, pick/place) |
| 3 | Install Unitree SDK 2 and connect to G1 via Ethernet |
| 4 | Replace `mock_robot` service with actual Unitree SDK 2 bridge |
| 5 | Set `MOCK_MODE=0` and download VLA model weights (`huggingface-cli login`) |
| 6 | Switch localization to real RealSense D435i topics |
| 7 | Run `docker compose up localization ros2_bridge gr00t` (no `mock_robot`) |
| 8 | Deploy trained RL checkpoint to onboard compute |

---

## Validation Commands

```bash
# Syntax-check all Python files
find . -name "*.py" | xargs python3 -m py_compile && echo "All OK"

# Validate docker-compose config
docker compose config --quiet && echo "Compose OK"

# Run domain randomization self-test (no Isaac Sim needed)
python3 isaac_lab/domain_rand.py

# Run safety monitor offline self-test (no ROS needed)
python3 ros2_bridge/safety_monitor.py

# Run mock robot dynamics test (no ROS needed)
python3 mock_robot/mock_g1_node.py
```
