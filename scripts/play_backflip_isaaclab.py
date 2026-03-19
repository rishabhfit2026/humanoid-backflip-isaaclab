"""
Run Isaac Lab backflip policy using humanoid-mjlab's proper environment and viewer.
Usage:
    cd ~/humanoid-mjlab
    unset PYTHONPATH LD_LIBRARY_PATH
    uv run python scripts/play_backflip_isaaclab.py
Then open: http://localhost:8080
"""
import torch
import numpy as np
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer

POLICY_PATH = "/home/rishabh/IsaacLab/logs/rsl_rl/28dof_backflip/2026-03-18_15-50-56/exported/policy.pt"

# Isaac Lab joint order (28 DOF)
ISAAC_ORDER = [
    "left_hip_pitch_joint",  "right_hip_pitch_joint",
    "left_hip_roll_joint",   "right_hip_roll_joint",
    "left_hip_yaw_joint",    "right_hip_yaw_joint",
    "left_knee_joint",       "right_knee_joint",
    "left_foot1_joint",      "right_foot1_joint",
    "left_foot2_joint",      "right_foot2_joint",
    "torso_joint",           "waist_roll_joint",    "waist_pitch_joint",
    "waist_rod_joint",
    "left_shoulder_pitch_joint",  "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
    "left_elbow_yaw_joint",       "right_elbow_yaw_joint",
    "left_elbow_forearm_yaw_joint","right_elbow_forearm_yaw_joint",
    "left_wrist_yaw_joint",       "right_wrist_yaw_joint",
    "neck_yaw_joint",             "head_joint",
]

# mjlab ctrl order (25 DOF, from actuator names)
MJLAB_ORDER = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",       "left_foot1_joint",       "left_foot2_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",      "right_foot1_joint",      "right_foot2_joint",
    "torso_joint",           "waist_roll_joint",        "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_elbow_yaw_joint",  "left_elbow_forearm_yaw_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_elbow_yaw_joint", "right_elbow_forearm_yaw_joint","right_wrist_yaw_joint",
]

# Build mapping: isaac index → mjlab index
isaac_idx  = {n: i for i, n in enumerate(ISAAC_ORDER)}
mjlab_idx  = {n: i for i, n in enumerate(MJLAB_ORDER)}
isaac2mjlab = [mjlab_idx.get(n, -1) for n in ISAAC_ORDER]

# Load Isaac Lab policy
print("Loading Isaac Lab backflip policy...")
isaac_policy = torch.jit.load(POLICY_PATH, map_location="cpu")
isaac_policy.eval()
print("✅ Policy loaded (93 obs → 28 actions)")

# Load humanoid-mjlab environment
print("Loading humanoid-mjlab environment...")
env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Humanoid", play=True)
env_cfg.num_envs = 1
env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(env)
print("✅ Environment loaded")

# Policy wrapper that converts mjlab obs → Isaac Lab format → mjlab actions
class BackflipPolicyWrapper:
    """
    Converts mjlab TensorDict obs to Isaac Lab format,
    runs Isaac Lab policy, converts 28-dim actions to 25-dim mjlab actions.
    """
    def __call__(self, obs) -> torch.Tensor:
        # obs is a TensorDict - extract the policy observations
        if hasattr(obs, "get"):
            # TensorDict - get policy key
            policy_obs = obs.get("policy")  # shape: (batch, 86)
        else:
            policy_obs = obs

        batch = policy_obs.shape[0]
        device = policy_obs.device

        # mjlab obs: [ang_vel(3), proj_gravity(3), cmd(3), jpos(25), jvel(25), actions(25), gait(2)] = 86
        # isaac obs: [lin_vel(3), ang_vel(3), proj_gravity(3), jpos(28), jvel(28), actions(28)] = 93

        ang_vel      = policy_obs[:, 0:3]
        proj_gravity = policy_obs[:, 3:6]
        jpos_mjlab   = policy_obs[:, 9:34]   # 25 joints
        jvel_mjlab   = policy_obs[:, 34:59]  # 25 joints
        acts_mjlab   = policy_obs[:, 59:84]  # 25 joints

        # Expand mjlab 25-DOF → Isaac Lab 28-DOF
        jpos_isaac = torch.zeros(batch, 28, device=device)
        jvel_isaac = torch.zeros(batch, 28, device=device)
        acts_isaac = torch.zeros(batch, 28, device=device)

        for ii, mi in enumerate(isaac2mjlab):
            if mi >= 0:
                jpos_isaac[:, ii] = jpos_mjlab[:, mi]
                jvel_isaac[:, ii] = jvel_mjlab[:, mi]
                acts_isaac[:, ii] = acts_mjlab[:, mi]

        # lin_vel not in mjlab obs — use zeros
        lin_vel = torch.zeros(batch, 3, device=device)

        isaac_obs = torch.cat([
            lin_vel, ang_vel, proj_gravity,
            jpos_isaac, jvel_isaac, acts_isaac
        ], dim=1)  # → (batch, 93)

        with torch.no_grad():
            action_isaac = isaac_policy(isaac_obs.cpu()).to(device)  # (batch, 28)

        # Convert Isaac Lab 28-DOF actions → mjlab 25-DOF
        action_mjlab = torch.zeros(batch, 25, device=device)
        for ii, mi in enumerate(isaac2mjlab):
            if mi >= 0:
                action_mjlab[:, mi] = action_isaac[:, ii]

        return action_mjlab


policy = BackflipPolicyWrapper()

print("\n🚀 Running backflip policy in humanoid-mjlab environment!")
print("   Open browser: http://localhost:8080\n")

# Use humanoid-mjlab's ViserPlayViewer — gives the blue checkered environment!
viewer = ViserPlayViewer(env=env, policy=policy)
viewer.run()
