# Humanoid 28-DOF Backflip — Isaac Lab RL Training

A complete reinforcement learning pipeline to train a 28-DOF humanoid robot 
to perform backflips in simulation using NVIDIA Isaac Lab, with planned 
Sim-to-Sim transfer to MuJoCo and Sim-to-Real deployment.

---

## Project Overview

This project trains a humanoid robot to perform backflips using Proximal Policy 
Optimization (PPO) in NVIDIA Isaac Sim 4.5 / Isaac Lab v2.3.2. The robot learns 
purely from trial and error — no human demonstrations or motion capture data.

**Robot:** Custom 28-DOF humanoid (humanoid_pkg)  
**Simulator:** NVIDIA Isaac Sim 4.5.0  
**Framework:** Isaac Lab v2.3.2  
**Algorithm:** PPO (Proximal Policy Optimization) via RSL-RL  
**Hardware:** Ubuntu 24.04, RTX 4090, Intel Ultra 7 265K, 64GB RAM  

---

## Project Structure
```
backflip_project/
├── assets/
│   └── robots/
│       ├── humanoid_28dof.urdf          # Original URDF
│       ├── humanoid_28dof_converted.usd # Isaac Sim USD (converted)
│       └── meshes/                      # STL mesh files (39 files)
├── envs/
│   ├── __init__.py
│   └── backflip_env_cfg.py              # Main environment config + agent config
├── mdp/
│   ├── __init__.py
│   └── rewards.py                       # Custom reward functions
├── scripts/
│   ├── train.py                         # Training script
│   ├── play.py                          # Evaluation/visualization script
│   ├── play_mujoco.py                   # MuJoCo Sim-to-Sim transfer
│   └── convert_urdf.py                  # URDF to USD conversion utility
├── config/
│   ├── __init__.py
│   ├── agent_cfg.py                     # PPO agent configuration
│   └── register_task.py                 # Gymnasium task registration
├── logs/                                # Training logs and checkpoints
├── .vscode/
│   ├── launch.json                      # VS Code debug configurations
│   └── settings.json                    # Python interpreter settings
├── requirements.txt
└── README.md
```

---

## Robot Specification

| Property | Value |
|----------|-------|
| Total DOF | 28 revolute joints |
| Leg joints | 12 (hip pitch/roll/yaw, knee, foot x2 per side) |
| Torso joints | 4 (torso, waist roll/pitch/rod) |
| Arm joints | 10 (shoulder pitch/roll, elbow yaw/forearm, wrist per side) |
| Head joints | 2 (neck yaw, head) |
| Spawn height | 1.2m |
| Physics timestep | 0.01s (100Hz) |
| Policy timestep | 0.02s (50Hz, decimation=2) |

---

## Environment Design

### Observation Space (93 dimensions)
| Term | Dimensions | Description |
|------|-----------|-------------|
| base_lin_vel | 3 | Root linear velocity in body frame |
| base_ang_vel | 3 | Root angular velocity in body frame |
| projected_gravity | 3 | Gravity vector in body frame |
| joint_pos | 28 | Joint positions relative to default |
| joint_vel | 28 | Joint velocities |
| last_action | 28 | Previous action output |

### Action Space (28 dimensions)
Joint position targets for all 28 DOF, scaled by 0.5.

### Reward Function

The reward function is designed around three phases of a backflip:

**Phase 1 — Jump**
```python
jump_height = gaussian_reward(height, target=1.3m)  # weight: 3.0
alive = step_reward()                                 # weight: 1.0
```

**Phase 2 — Rotate**
```python
backflip_rotation = tanh(ang_vel_y / 6.0)           # weight: 8.0
# target: -6.0 rad/s backward rotation
```

**Phase 3 — Land**
```python
upright_after_land = clamp(uprightness - 0.85)       # weight: 6.0
landing_smoothness = -vertical_velocity^2            # weight: -0.001
```

**Regularisation**
```python
dof_torques_l2   # weight: -1e-6  (smooth torques)
action_rate_l2   # weight: -0.01  (smooth actions)
knee_deviation   # weight: -0.1   (protect knees)
waist_deviation  # weight: -0.05  (protect waist)
```

### Termination Conditions
- Pelvis height below 0.25m (robot fell)
- Torso/pelvis contact with ground
- Episode timeout (5 seconds)

### Domain Randomization
- Physics material: friction ±30%, restitution 0–0.1
- Base mass: ±1.5 kg
- External pushes: ±0.5 m/s every 8–10 seconds

---

## PPO Agent Configuration

| Parameter | Value |
|-----------|-------|
| Hidden layers | [512, 256, 128] |
| Activation | ELU |
| Learning rate | 1e-4 (adaptive) |
| Steps per env | 24 |
| Mini batches | 4 |
| Epochs per update | 5 |
| Clip parameter | 0.2 |
| Entropy coefficient | 0.005 |
| Discount (gamma) | 0.99 |
| GAE lambda | 0.95 |
| Max iterations | 10,000 |

---

## Installation

### Prerequisites
- Ubuntu 22.04 or 24.04
- NVIDIA GPU (RTX 3090 or better recommended)
- NVIDIA Driver 535+
- Conda/Miniconda

### Step 1 — Create conda environment
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
```

### Step 2 — Install Isaac Sim 4.5
```bash
pip install isaacsim==4.5.0.0 \
    isaacsim-rl isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com
```

### Step 3 — Clone and install Isaac Lab
```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.3.2
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_rl
```

### Step 4 — Clone robot repository
```bash
git clone https://github.com/Sentient-X/humanoid_pkg.git ~/humanoid_pkg
cd ~/humanoid_pkg
git lfs install
git lfs pull
```

### Step 5 — Clone this project
```bash
git clone <this_repo> ~/backflip_project
```

### Step 6 — Apply Isaac Lab compatibility patches
Several patches are required for Isaac Sim 4.5 + Isaac Lab v2.3.2:
```bash
# Patch 1: Fix flatdict import in simulation_context.py
# Patch 2: Fix XR device imports
# Patch 3: Fix URDF importer extension loading
# Patch 4: Fix merge_fixed_joints API
cd ~/IsaacLab
python ~/backflip_project/scripts/apply_patches.py
```

### Step 7 — Convert URDF to USD
```bash
cd ~/IsaacLab
python scripts/tools/convert_urdf.py \
    ~/humanoid_pkg/urdf/humanoid_28dof.urdf \
    ~/backflip_project/assets/robots/humanoid_28dof_converted.usd \
    --merge-joints \
    --headless
```

### Step 8 — Register task in Isaac Lab
```bash
cp -r ~/backflip_project/envs/ \
    ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/backflip_28dof/
```

---

## Training

### Start training
```bash
cd ~/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task 28dof_env \
    --num_envs 64 \
    --headless
```

### Monitor with TensorBoard
```bash
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/28dof_backflip --port 6006
# Open browser: http://localhost:6006
```

### Resume training from checkpoint
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task 28dof_env \
    --num_envs 64 \
    --headless \
    --resume
```

---

## Evaluation

### Record video of trained policy
```bash
cd ~/IsaacLab
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task 28dof_env-Play \
    --num_envs 1 \
    --load_run 2026-03-16_15-22-58 \
    --headless \
    --video \
    --video_length 500
```

### Play in MuJoCo (Sim-to-Sim)
```bash
cd ~/backflip_project
python scripts/play_mujoco.py
```

---

## Training Results

Training completed in **65 minutes** on RTX 4090 (10,000 iterations).

| Metric | Start | Final | Improvement |
|--------|-------|-------|-------------|
| Mean reward | 3.4 | 27.69 | +714% |
| Backflip rotation reward | 0.55 | 5.30 | +864% |
| Mean episode length | 16 steps | 100 steps | +525% |
| Episode completion rate | 0% | 49% | — |
| Base height termination | 100% | 27% | -73% |
| Torso contact termination | 100% | 23% | -77% |
| Training speed | — | ~4000 steps/s | — |

### Learning Curve Summary
- **Iterations 0–200:** Robot learns basic stability
- **Iterations 200–1000:** Discovers backward rotation motion
- **Iterations 1000–3000:** Jump height and rotation speed improve
- **Iterations 3000–7000:** Episode completion rate climbs, landing improves
- **Iterations 7000–10000:** Policy matures, 49% full episode completion

---

## Known Issues and Patches

### Isaac Sim 4.5 + Isaac Lab v2.3.2 Compatibility

| Issue | File | Fix |
|-------|------|-----|
| `flatdict` import error | `simulation_context.py` | Replace with pure Python dict flatten |
| `XRPoseValidityFlags` missing | `manus_vive_utils.py` | Stub with `object` |
| `XrCfg` missing | `*_env_cfg.py` | Stub with `object` |
| `omni.replicator` TypeError | `manager_based_env.py` | Catch `Exception` not `ModuleNotFoundError` |
| `isaacsim.asset` not found | `urdf_converter.py` | Load via extension manager |
| `set_merge_fixed_ignore_inertia` missing | `urdf_converter.py` | Comment out deprecated call |
| `ImplicitActuatorCfg` in wrong module | `backflip_env_cfg.py` | Import from `isaaclab.actuators` |
| Joint name with space | URDF | URDF importer auto-corrects to double underscore |

---

## Roadmap

- [x] Isaac Lab environment setup
- [x] URDF to USD conversion
- [x] Custom reward function design
- [x] PPO training — 10,000 iterations
- [x] Policy visualization (video)
- [ ] Sim-to-Sim transfer (MuJoCo)
- [ ] Policy distillation / fine-tuning
- [ ] Sim-to-Real deployment

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [humanoid_pkg](https://github.com/Sentient-X/humanoid_pkg)
- [MuJoCo](https://mujoco.org/)

---

## Citation

If you use this work, please cite:
```bibtex
@misc{humanoid_backflip_2026,
  title={Humanoid 28-DOF Backflip Training with Isaac Lab},
  author={Rishabh},
  year={2026},
  url={<repo_url>}
}
```
