"""
Sim2Sim script for asimov_toe robot.
Matches mjlab training configuration exactly.

Supports both PyTorch (.pt) and ONNX (.onnx) model formats.

Usage:
  # Using PyTorch checkpoint (default)
  python sim2sim.py

  # Using ONNX model
  python sim2sim.py --onnx path/to/model.onnx

  # Using specific PyTorch checkpoint
  python sim2sim.py --pt path/to/model.pt
"""
import argparse
import os
os.environ["MUJOCO_GL"] = "egl"  # For headless rendering

import mujoco
import numpy as np
import time
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
import csv
import imageio
from datetime import datetime


# =============================================================================
# Utility Classes
# =============================================================================

class CSVLogger:
    """Simple CSV logger with dynamic column registration"""
    def __init__(self, log_dir="logs"):
        self.columns = []
        self.log_dir = log_dir
        self.header_written = False
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"simulation_log_{timestamp}.csv")
        print(f"Logging to: {self.log_file}")

    def log(self, **kwargs):
        expanded_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, list)):
                if isinstance(value, np.ndarray):
                    value = value.flatten().tolist()
                for i, v in enumerate(value):
                    expanded_kwargs[f"{key}_{i}"] = v
            else:
                expanded_kwargs[key] = value

        for key in expanded_kwargs.keys():
            if key not in self.columns:
                self.columns.append(key)

        if not self.header_written:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
            self.header_written = True

        row = [expanded_kwargs.get(col, '') for col in self.columns]
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class ActorNetwork(nn.Module):
    """Actor network matching RSL-RL checkpoint format"""
    def __init__(self, num_obs=45, num_actions=12):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        )
        self.register_buffer('obs_mean', torch.zeros(1, num_obs))
        self.register_buffer('obs_std', torch.ones(1, num_obs))

    def forward(self, obs):
        obs_normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return self.actor(obs_normalized)


def load_policy_from_checkpoint(checkpoint_path, num_obs=45, num_actions=12):
    """Load policy from RSL-RL style checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    policy = ActorNetwork(num_obs, num_actions)

    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('actor.') and not key.startswith('actor_obs'):
            actor_state_dict[key.replace('actor.', '')] = value

    policy.actor.load_state_dict(actor_state_dict)

    if 'actor_obs_normalizer._mean' in state_dict:
        policy.obs_mean = state_dict['actor_obs_normalizer._mean']
        policy.obs_std = state_dict['actor_obs_normalizer._std']

    policy.eval()
    return policy


class OnnxPolicy:
    """ONNX model inference wrapper for sim2sim."""

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with:\n"
                "  pip install onnxruntime  # CPU version\n"
                "  pip install onnxruntime-gpu  # GPU version"
            )

        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Loaded ONNX model from {onnx_path}")
        print(f"  Input: {self.input_name}, Output: {self.output_name}")

    def __call__(self, obs: torch.Tensor) -> np.ndarray:
        """Run inference on observations."""
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        actions_np = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        return actions_np.squeeze()


def load_policy(model_path: str, num_obs: int = 45, num_actions: int = 12):
    """Load policy from either PyTorch checkpoint or ONNX file.

    Args:
        model_path: Path to .pt or .onnx file
        num_obs: Number of observations (only used for .pt)
        num_actions: Number of actions (only used for .pt)

    Returns:
        Policy object (either ActorNetwork or OnnxPolicy)
    """
    if model_path.endswith('.onnx'):
        return OnnxPolicy(model_path)
    else:
        return load_policy_from_checkpoint(model_path, num_obs, num_actions)


# =============================================================================
# Helper Functions
# =============================================================================

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_gravity_orientation(quaternion):
    """Compute projected gravity in body frame from quaternion [w,x,y,z]"""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sim2Sim for asimov_toe robot. Supports PyTorch and ONNX models."
    )
    parser.add_argument(
        "--pt", type=str, default=None,
        help="Path to PyTorch checkpoint (.pt file)"
    )
    parser.add_argument(
        "--onnx", type=str, default=None,
        help="Path to ONNX model (.onnx file)"
    )
    parser.add_argument(
        "--xml", type=str,
        default="/home/ishneet/Desktop/asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov_toe.xml",
        help="Path to robot XML file"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
        help="Velocity command [vx, vy, wz] (default: 0.5 0.0 0.0)"
    )
    return parser.parse_args()


# =============================================================================
# Configuration (matching mjlab asimov_toe exactly)
# =============================================================================

# Default paths (can be overridden by command-line args)
DEFAULT_PT_PATH = "/home/ishneet/Desktop/asimov-mjlab/logs/rsl_rl/asimov_toe_velocity/2025-12-01_15-59-05/model_1500.pt"

# Simulation settings (matching mjlab: timestep=0.005, decimation=4 -> 50Hz policy)
simulation_duration = 30.0
simulation_dt = 0.005  # 200 Hz physics (same as mjlab)
control_decimation = 4  # Policy at 50 Hz (same as mjlab)

num_obs = 45
num_actions = 12

# Joint order: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] x2 (L/R)

# Identified gains from CAN data (symmetric L/R, Kd capped at 5.0)
# Order: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
kp_per_joint = np.array([22.5, 118.0, 130.0, 84.0, 14.0, 17.0], dtype=np.float32)
kps = np.concatenate([kp_per_joint, kp_per_joint])

# Damping from CAN data identification
kd_per_joint = np.array([1.4, 5.0, 5.0, 4.2, 1.7, 1.1], dtype=np.float32)
kds = np.concatenate([kd_per_joint, kd_per_joint])

# Torque limits from ENCOS specs (ankles doubled for parallel linkage)
torque_per_joint = np.array([55.0, 90.0, 60.0, 50.0, 72.0, 72.0], dtype=np.float32)
torque_limits = np.concatenate([torque_per_joint, torque_per_joint])

# Default angles (KNEES_BENT_KEYFRAME)
default_angles = np.array([
    0.2, 0.0, 0.0, 0.4, -0.25, 0.0,   # Left leg
    -0.2, 0.0, 0.0, -0.4, 0.25, 0.0,  # Right leg
], dtype=np.float32)

# Observation scales (matching mjlab velocity_env_cfg.py)
ang_vel_scale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05

# Action scales: 0.25 * effort_limit / stiffness (from asimov_toe_constants.py)
action_scales = np.concatenate([
    0.25 * torque_per_joint / kp_per_joint,
    0.25 * torque_per_joint / kp_per_joint
])

# =============================================================================
# Main Simulation
# =============================================================================

def main():
    args = parse_args()

    # Determine model path
    if args.onnx is not None:
        model_path = args.onnx
        model_type = "ONNX"
    elif args.pt is not None:
        model_path = args.pt
        model_type = "PyTorch"
    else:
        model_path = DEFAULT_PT_PATH
        model_type = "PyTorch"

    xml_path = args.xml
    simulation_duration = args.duration
    cmd = np.array(args.cmd, dtype=np.float32)

    print("=" * 60)
    print("Sim2Sim for asimov_toe (matching mjlab config)")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")
    print(f"Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")

    # Initialize
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    logger = CSVLogger(log_dir="logs")

    # Load robot model
    print(f"Loading XML: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Set armature values (CRITICAL: mjlab sets these via spec, we must set them manually)
    # Armature = reflected inertia from gearbox: J_reflected = J_rotor * gear_ratio^2
    # Values from ENCOS motor specs (stats.md), ankles doubled for parallel linkage
    armature_map = {
        "hip_pitch": 0.0652,   # EC-A6416-P2-25: 104.395 kg·mm² * 25²
        "hip_roll": 0.100,     # EC-A5013-H17-100: 10 kg·mm² * 100²
        "hip_yaw": 0.0343,     # EC-A3814-H14-107: 3 kg·mm² * 107²
        "knee": 0.0330,        # EC-A4315-P2-36: 25.5 kg·mm² * 36²
        "ankle_pitch": 0.0472, # EC-A4310-P2-36: 18.2 kg·mm² * 36² * 2 (parallel linkage)
        "ankle_roll": 0.0472,  # EC-A4310-P2-36: 18.2 kg·mm² * 36² * 2 (parallel linkage)
    }

    print("\nSetting armature values:")
    for i in range(m.njnt):
        jnt = m.joint(i)
        if jnt.type[0] == 0:  # Skip free joint
            continue
        dof_id = m.jnt_dofadr[i]
        for key, armature in armature_map.items():
            if key in jnt.name:
                m.dof_armature[dof_id] = armature
                print(f"  {jnt.name}: armature = {armature:.6f}")
                break

    # Get actuated joint indices
    actuated_joints = m.actuator_trnid[:, 0]
    qpos_ids = m.jnt_qposadr[actuated_joints]
    qvel_ids = m.jnt_dofadr[actuated_joints]

    # Print joint order
    actuated_joint_names = [m.joint(jid).name for jid in actuated_joints]
    print("\nActuated Joint Order:")
    for i, name in enumerate(actuated_joint_names):
        print(f"  {i:02d}: {name}")

    assert len(qpos_ids) == num_actions, f"Expected {num_actions} actuators, got {len(qpos_ids)}"

    # Set initial pose (includes toe joints which aren't actuated)
    nq = m.nq - 7
    initial_q_all = np.array([
        0.2, 0.0, 0.0, 0.4, -0.25, 0.0, 0.0,   # Left leg + toe
        -0.2, 0.0, 0.0, -0.4, 0.25, 0.0, 0.0,  # Right leg + toe
    ], dtype=np.float32)

    d.qpos[7:7+nq] = initial_q_all
    d.qpos[2] = 0.73  # Initial height
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Load policy
    print(f"\nLoading policy: {model_path}")
    policy = load_policy(model_path, num_obs=num_obs, num_actions=num_actions)
    is_onnx = isinstance(policy, OnnxPolicy)
    if not is_onnx:
        print("Policy loaded with observation normalization")

    # Setup video recording
    renderer = mujoco.Renderer(m, height=480, width=640)
    frames = []
    video_fps = 30
    frame_skip = int(1.0 / (simulation_dt * video_fps))
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/asimov_toe_sim2sim.mp4"

    print(f"\nRunning simulation for {simulation_duration}s...")
    print(f"Video will be saved to: {video_path}")

    # CSV for target positions
    csv_file = open("target_dof_pos_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step"] + actuated_joint_names)

    start = time.time()
    total_steps = int(simulation_duration / simulation_dt)

    for step_idx in range(total_steps):
        # Get current joint state
        q = d.qpos[qpos_ids]
        dq = d.qvel[qvel_ids]

        # PD control
        tau = pd_control(target_dof_pos, q, kps, np.zeros_like(kds), dq, kds)
        tau = np.clip(tau, -torque_limits, torque_limits)

        # Apply torques and step
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1

        # Policy runs at 50Hz
        if counter % control_decimation == 0:
            # Get observations (matching mjlab exactly)
            qj = d.qpos[qpos_ids]
            dqj = d.qvel[qvel_ids]

            # Angular velocity from frameangvel sensor (same as mjlab uses)
            ang_vel = d.sensor("imu_ang_vel").data.copy()

            # Projected gravity from ground truth quaternion (same as mjlab uses)
            gravity_orientation = get_gravity_orientation(d.qpos[3:7])

            # Build observation vector (order matches env_cfgs.py)
            obs[0:3] = ang_vel * ang_vel_scale
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd  # No scaling (mjlab passes raw commands)
            obs[9:21] = (qj - default_angles) * dof_pos_scale
            obs[21:33] = dqj * dof_vel_scale
            obs[33:45] = action

            # Run policy (handles both PyTorch and ONNX)
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                action_out = policy(obs_tensor)
                # Handle both tensor and numpy outputs
                if isinstance(action_out, torch.Tensor):
                    action = action_out.numpy().squeeze()
                else:
                    action = action_out

            # Compute target position with per-joint action scales
            target_dof_pos = default_angles + action * action_scales

            csv_writer.writerow([counter] + list(target_dof_pos))

        # Record frame for video
        if step_idx % frame_skip == 0:
            renderer.update_scene(d, camera="front_camera")  # Actually gives side view when robot walks +X
            frame = renderer.render()
            frames.append(frame)

        # Progress update
        if step_idx % int(5.0 / simulation_dt) == 0:
            sim_time = step_idx * simulation_dt
            print(f"  Sim time: {sim_time:.1f}s / {simulation_duration}s")

    elapsed = time.time() - start
    print(f"\nSimulation completed in {elapsed:.2f}s (real time)")

    # Save video
    print(f"Saving video with {len(frames)} frames...")
    imageio.mimsave(video_path, frames, fps=video_fps)
    print(f"Video saved to: {video_path}")

    # Cleanup
    csv_file.close()
    renderer.close()
    print("Done!")


if __name__ == "__main__":
    main()
