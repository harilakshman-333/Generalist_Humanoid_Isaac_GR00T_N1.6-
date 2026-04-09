# Task: UnifoLM-WBT-Dataset Integration

Integrate the newly released Unitree whole-body teleoperation dataset to bridge the sim-to-real gap and enhance VLA performance.

## Phase 1: Data Acquisition & Preprocessing
- [ ] **Download Dataset**: Pull `UnifoLM-WBT-Dataset` from Hugging Face.
- [ ] **Schema Mapping**: Verify compatibility between dataset action/observation formats and the `gr00t_model` inference pipeline.
- [ ] **Trajectory Extraction**: Extract high-quality manipulation trajectories (folding, cleaning) for imitation learning.

## Phase 2: VLA Enhancement
- [ ] **Fine-tuning Pipeline**: Set up a training script in `gr00t_model/` to fine-tune `UnifoLM-VLA-Base` on real G1 demonstrations.
- [ ] **Evaluation**: Compare "mock" VLA outputs vs. dataset-aligned VLA outputs in the `mock_robot` simulation.

## Phase 3: RL & Dynamics Refinement
- [ ] **Statistical Analysis**: Calculate real-world joint torque and velocity distributions from WBT data.
- [ ] **Domain Rand Update**: Update `isaac_lab/domain_rand.py` with tighter, data-informed bounds.
- [ ] **Imitation Learning**: Implement a GAIL or DAgger-style training loop in Isaac Lab using WBT expert demonstrations.

## Phase 4: Validation
- [ ] **Sim-to-Real Benchmark**: Test if the WBT-aligned policy exhibits fewer self-collisions or joint limit violations compared to pure RL.
