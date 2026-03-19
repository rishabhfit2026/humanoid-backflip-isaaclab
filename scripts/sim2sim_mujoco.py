"""
Sim-to-Sim Transfer: Isaac Lab → MuJoCo
Fixed with correct joint mapping from debug analysis.
"""
import numpy as np
import torch
import mujoco
import mujoco.viewer
import time

POLICY_PATH = "/home/rishabh/IsaacLab/logs/rsl_rl/28dof_backflip/2026-03-18_15-50-56/exported/policy.pt"
MJCF_PATH   = "/home/rishabh/humanoid_pkg/mjcf/humanoid_28dof.xml"

MAX_EPISODE_STEPS = 250
TARGET_DT         = 0.02
PHYSICS_STEPS     = 2
MIN_HEIGHT        = 0.05   # very low — let backflip complete
SPAWN_HEIGHT      = 0.42
ACTION_SCALE      = 0.2    # matches Isaac Lab JointPositionActionCfg scale=0.5

# ── Isaac Lab action order (index 0-27) ──────────────────────────────────────
ISAAC_ORDER = [
    "left_hip_pitch_joint",          # 0
    "right_hip_pitch_joint",         # 1
    "left_hip_roll_joint",           # 2
    "right_hip_roll_joint",          # 3
    "left_hip_yaw_joint",            # 4
    "right_hip_yaw_joint",           # 5
    "left_knee_joint",               # 6
    "right_knee_joint",              # 7
    "left_foot1_joint",              # 8
    "right_foot1_joint",             # 9
    "left_foot2_joint",              # 10
    "right_foot2_joint",             # 11
    "torso_joint",                   # 12
    "waist_roll_joint",              # 13
    "waist_pitch_joint",             # 14
    "waist_rod_joint",               # 15
    "left_shoulder_pitch_joint",     # 16
    "right_shoulder_pitch_joint",    # 17
    "left_shoulder_roll_joint",      # 18
    "right_shoulder_roll_joint",     # 19
    "left_elbow_yaw_joint",          # 20
    "right_elbow_yaw_joint",         # 21
    "left_elbow_forearm_yaw_joint",  # 22
    "right_elbow_forearm_yaw_joint", # 23
    "left_wrist_yaw_joint",          # 24
    "right_wrist_yaw_joint",         # 25
    "neck_yaw_joint",                # 26
    "head_joint",                    # 27
]

# ── MuJoCo ctrl order (from debug output) ────────────────────────────────────
# Exact actuator order from MuJoCo model (verified from debug)
# Exact actuator order from MuJoCo model (verified from debug)
# Exact actuator order from MuJoCo model (verified from debug)
# Exact actuator order from MuJoCo model (verified from debug)
MUJOCO_CTRL_ORDER = [
    "torso_joint",                   # ctrl[0]
    "waist_roll_joint",              # ctrl[1]
    "waist_pitch_joint",             # ctrl[2]
    "waist_rod_joint",               # ctrl[3]
    "left_hip_pitch_joint",          # ctrl[4]
    "left_hip_roll_joint",           # ctrl[5]
    "left_hip_yaw_joint",            # ctrl[6]
    "left_knee_joint",               # ctrl[7]
    "left_foot1_joint",              # ctrl[8]
    "left_foot2_joint",              # ctrl[9]
    "right_hip_pitch_joint",         # ctrl[10]
    "right_hip_roll_joint",          # ctrl[11]
    "right_hip_yaw_joint",           # ctrl[12]
    "right_knee_joint",              # ctrl[13]
    "right_foot1_joint",             # ctrl[14]
    "right_foot2_joint",             # ctrl[15]
    "left_shoulder_pitch_joint",     # ctrl[16]
    "left_shoulder_roll_joint",      # ctrl[17]
    "left_elbow_yaw_joint",          # ctrl[18]
    "left_elbow_forearm_yaw_joint",  # ctrl[19]
    "left_wrist_yaw_joint",          # ctrl[20]
    "right_shoulder_pitch_joint",    # ctrl[21]
    "right_shoulder_roll_joint",     # ctrl[22]
    "right_elbow_yaw_joint",         # ctrl[23]
    "right_elbow_forearm_yaw_joint", # ctrl[24]
    "right_wrist_yaw_joint",         # ctrl[25]
    "neck_yaw_joint",                # ctrl[26]
    "head_joint",                    # ctrl[27]
]

# ── MuJoCo qpos order (qpos[7:35], from debug output) ────────────────────────
MUJOCO_QPOS_ORDER = [
    "torso_joint",                   # qpos[7]
    "waist_roll_joint",              # qpos[8]
    "waist_pitch_joint",             # qpos[9]
    "left_shoulder_pitch_joint",     # qpos[10]
    "left_shoulder_roll_joint",      # qpos[11]
    "left_elbow_yaw_joint",          # qpos[12]
    "left_elbow_forearm_yaw_joint",  # qpos[13]
    "left_wrist_yaw_joint",          # qpos[14]
    "right_shoulder_pitch_joint",    # qpos[15]
    "right_shoulder_roll_joint",     # qpos[16]
    "right_elbow_yaw_joint",         # qpos[17]
    "right_elbow_forearm_yaw_joint", # qpos[18]
    "right_wrist_yaw_joint",         # qpos[19]
    "neck_yaw_joint",                # qpos[20]
    "head_joint",                    # qpos[21]
    "waist_rod_joint",               # qpos[22]
    "right_hip_pitch_joint",         # qpos[23]
    "right_hip_roll_joint",          # qpos[24]
    "right_hip_yaw_joint",           # qpos[25]
    "right_knee_joint",              # qpos[26]
    "right_foot1_joint",             # qpos[27]
    "right_foot2_joint",             # qpos[28]
    "left_hip_pitch_joint",          # qpos[29]
    "left_hip_roll_joint",           # qpos[30]
    "left_hip_yaw_joint",            # qpos[31]
    "left_knee_joint",               # qpos[32]
    "left_foot1_joint",              # qpos[33]
    "left_foot2_joint",              # qpos[34]
]

# ── Default positions in Isaac order ─────────────────────────────────────────
ISAAC_DEFAULT = {
    "left_hip_pitch_joint":          -0.20,
    "right_hip_pitch_joint":         -0.20,
    "left_hip_roll_joint":            0.0,
    "right_hip_roll_joint":           0.0,
    "left_hip_yaw_joint":             0.0,
    "right_hip_yaw_joint":            0.0,
    "left_knee_joint":                0.45,
    "right_knee_joint":               0.45,
    "left_foot1_joint":              -0.20,
    "right_foot1_joint":             -0.20,
    "left_foot2_joint":               0.0,
    "right_foot2_joint":              0.0,
    "torso_joint":                    0.0,
    "waist_roll_joint":               0.0,
    "waist_pitch_joint":              0.0,
    "waist_rod_joint":                0.0,
    "left_shoulder_pitch_joint":      0.0,
    "right_shoulder_pitch_joint":     0.0,
    "left_shoulder_roll_joint":       0.3,
    "right_shoulder_roll_joint":     -0.3,
    "left_elbow_yaw_joint":           0.0,
    "right_elbow_yaw_joint":          0.0,
    "left_elbow_forearm_yaw_joint":   0.5,
    "right_elbow_forearm_yaw_joint":  0.5,
    "left_wrist_yaw_joint":           0.0,
    "right_wrist_yaw_joint":          0.0,
    "neck_yaw_joint":                 0.0,
    "head_joint":                     0.0,
}

# Build lookup dicts
isaac_to_idx = {n: i for i, n in enumerate(ISAAC_ORDER)}

# Default arrays
ISAAC_DEFAULT_ARR = np.array([ISAAC_DEFAULT[n] for n in ISAAC_ORDER], dtype=np.float32)

# Correct ctrl defaults from actuator→joint mapping analysis
# ctrl order: torso, waist_roll, waist_pitch, waist_rod,
#             left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_foot1, left_foot2,
#             right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_foot1, right_foot2,
#             left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_forearm, left_wrist,
#             right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_forearm, right_wrist,
#             neck_yaw, head
CTRL_DEFAULT = np.array([
    0.0,   # ctrl[0]  torso_joint
    0.0,   # ctrl[1]  waist_roll_joint
    0.0,   # ctrl[2]  waist_pitch_joint
    0.0,   # ctrl[3]  waist_rod_joint
   -0.20,  # ctrl[4]  left_hip_pitch_joint
    0.0,   # ctrl[5]  left_hip_roll_joint
    0.0,   # ctrl[6]  left_hip_yaw_joint
    0.45,  # ctrl[7]  left_knee_joint
   -0.20,  # ctrl[8]  left_foot1_joint
    0.0,   # ctrl[9]  left_foot2_joint
   -0.20,  # ctrl[10] right_hip_pitch_joint
    0.0,   # ctrl[11] right_hip_roll_joint
    0.0,   # ctrl[12] right_hip_yaw_joint
    0.45,  # ctrl[13] right_knee_joint
   -0.20,  # ctrl[14] right_foot1_joint
    0.0,   # ctrl[15] right_foot2_joint
    0.0,   # ctrl[16] left_shoulder_pitch_joint
    0.3,   # ctrl[17] left_shoulder_roll_joint
    0.0,   # ctrl[18] left_elbow_yaw_joint
    0.5,   # ctrl[19] left_elbow_forearm_yaw_joint
    0.0,   # ctrl[20] left_wrist_yaw_joint
    0.0,   # ctrl[21] right_shoulder_pitch_joint
   -0.3,   # ctrl[22] right_shoulder_roll_joint
    0.0,   # ctrl[23] right_elbow_yaw_joint
    0.5,   # ctrl[24] right_elbow_forearm_yaw_joint
    0.0,   # ctrl[25] right_wrist_yaw_joint
    0.0,   # ctrl[26] neck_yaw_joint
    0.0,   # ctrl[27] head_joint
], dtype=np.float32)

# Correct ctrl defaults from actuator→joint mapping analysis
# ctrl order: torso, waist_roll, waist_pitch, waist_rod,
#             left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_foot1, left_foot2,
#             right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_foot1, right_foot2,
#             left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_forearm, left_wrist,
#             right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_forearm, right_wrist,
#             neck_yaw, head
CTRL_DEFAULT = np.array([
    0.0,   # ctrl[0]  torso_joint
    0.0,   # ctrl[1]  waist_roll_joint
    0.0,   # ctrl[2]  waist_pitch_joint
    0.0,   # ctrl[3]  waist_rod_joint
   -0.20,  # ctrl[4]  left_hip_pitch_joint
    0.0,   # ctrl[5]  left_hip_roll_joint
    0.0,   # ctrl[6]  left_hip_yaw_joint
    0.45,  # ctrl[7]  left_knee_joint
   -0.20,  # ctrl[8]  left_foot1_joint
    0.0,   # ctrl[9]  left_foot2_joint
   -0.20,  # ctrl[10] right_hip_pitch_joint
    0.0,   # ctrl[11] right_hip_roll_joint
    0.0,   # ctrl[12] right_hip_yaw_joint
    0.45,  # ctrl[13] right_knee_joint
   -0.20,  # ctrl[14] right_foot1_joint
    0.0,   # ctrl[15] right_foot2_joint
    0.0,   # ctrl[16] left_shoulder_pitch_joint
    0.3,   # ctrl[17] left_shoulder_roll_joint
    0.0,   # ctrl[18] left_elbow_yaw_joint
    0.5,   # ctrl[19] left_elbow_forearm_yaw_joint
    0.0,   # ctrl[20] left_wrist_yaw_joint
    0.0,   # ctrl[21] right_shoulder_pitch_joint
   -0.3,   # ctrl[22] right_shoulder_roll_joint
    0.0,   # ctrl[23] right_elbow_yaw_joint
    0.5,   # ctrl[24] right_elbow_forearm_yaw_joint
    0.0,   # ctrl[25] right_wrist_yaw_joint
    0.0,   # ctrl[26] neck_yaw_joint
    0.0,   # ctrl[27] head_joint
], dtype=np.float32)

# Correct ctrl defaults from actuator→joint mapping analysis
# ctrl order: torso, waist_roll, waist_pitch, waist_rod,
#             left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_foot1, left_foot2,
#             right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_foot1, right_foot2,
#             left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_forearm, left_wrist,
#             right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_forearm, right_wrist,
#             neck_yaw, head
CTRL_DEFAULT = np.array([
    0.0,   # ctrl[0]  torso_joint
    0.0,   # ctrl[1]  waist_roll_joint
    0.0,   # ctrl[2]  waist_pitch_joint
    0.0,   # ctrl[3]  waist_rod_joint
   -0.20,  # ctrl[4]  left_hip_pitch_joint
    0.0,   # ctrl[5]  left_hip_roll_joint
    0.0,   # ctrl[6]  left_hip_yaw_joint
    0.45,  # ctrl[7]  left_knee_joint
   -0.20,  # ctrl[8]  left_foot1_joint
    0.0,   # ctrl[9]  left_foot2_joint
   -0.20,  # ctrl[10] right_hip_pitch_joint
    0.0,   # ctrl[11] right_hip_roll_joint
    0.0,   # ctrl[12] right_hip_yaw_joint
    0.45,  # ctrl[13] right_knee_joint
   -0.20,  # ctrl[14] right_foot1_joint
    0.0,   # ctrl[15] right_foot2_joint
    0.0,   # ctrl[16] left_shoulder_pitch_joint
    0.3,   # ctrl[17] left_shoulder_roll_joint
    0.0,   # ctrl[18] left_elbow_yaw_joint
    0.5,   # ctrl[19] left_elbow_forearm_yaw_joint
    0.0,   # ctrl[20] left_wrist_yaw_joint
    0.0,   # ctrl[21] right_shoulder_pitch_joint
   -0.3,   # ctrl[22] right_shoulder_roll_joint
    0.0,   # ctrl[23] right_elbow_yaw_joint
    0.5,   # ctrl[24] right_elbow_forearm_yaw_joint
    0.0,   # ctrl[25] right_wrist_yaw_joint
    0.0,   # ctrl[26] neck_yaw_joint
    0.0,   # ctrl[27] head_joint
], dtype=np.float32)

# Correct ctrl defaults from actuator→joint mapping analysis
# ctrl order: torso, waist_roll, waist_pitch, waist_rod,
#             left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_foot1, left_foot2,
#             right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_foot1, right_foot2,
#             left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_forearm, left_wrist,
#             right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_forearm, right_wrist,
#             neck_yaw, head
CTRL_DEFAULT = np.array([
    0.0,   # ctrl[0]  torso_joint
    0.0,   # ctrl[1]  waist_roll_joint
    0.0,   # ctrl[2]  waist_pitch_joint
    0.0,   # ctrl[3]  waist_rod_joint
   -0.20,  # ctrl[4]  left_hip_pitch_joint
    0.0,   # ctrl[5]  left_hip_roll_joint
    0.0,   # ctrl[6]  left_hip_yaw_joint
    0.45,  # ctrl[7]  left_knee_joint
   -0.20,  # ctrl[8]  left_foot1_joint
    0.0,   # ctrl[9]  left_foot2_joint
   -0.20,  # ctrl[10] right_hip_pitch_joint
    0.0,   # ctrl[11] right_hip_roll_joint
    0.0,   # ctrl[12] right_hip_yaw_joint
    0.45,  # ctrl[13] right_knee_joint
   -0.20,  # ctrl[14] right_foot1_joint
    0.0,   # ctrl[15] right_foot2_joint
    0.0,   # ctrl[16] left_shoulder_pitch_joint
    0.3,   # ctrl[17] left_shoulder_roll_joint
    0.0,   # ctrl[18] left_elbow_yaw_joint
    0.5,   # ctrl[19] left_elbow_forearm_yaw_joint
    0.0,   # ctrl[20] left_wrist_yaw_joint
    0.0,   # ctrl[21] right_shoulder_pitch_joint
   -0.3,   # ctrl[22] right_shoulder_roll_joint
    0.0,   # ctrl[23] right_elbow_yaw_joint
    0.5,   # ctrl[24] right_elbow_forearm_yaw_joint
    0.0,   # ctrl[25] right_wrist_yaw_joint
    0.0,   # ctrl[26] neck_yaw_joint
    0.0,   # ctrl[27] head_joint
], dtype=np.float32)

# qpos[7+i] default
QPOS_DEFAULT = np.array([ISAAC_DEFAULT.get(n, 0.0) for n in MUJOCO_QPOS_ORDER], dtype=np.float32)

# isaac_action[i] → which ctrl index?
# action is OFFSET from default, scaled by ACTION_SCALE
# target = default + action * ACTION_SCALE
ISAAC_TO_CTRL = []
for name in ISAAC_ORDER:
    if name in MUJOCO_CTRL_ORDER:
        ISAAC_TO_CTRL.append(MUJOCO_CTRL_ORDER.index(name))
    else:
        ISAAC_TO_CTRL.append(-1)

# qpos[7+i] → which isaac index? (for observations)
QPOS_TO_ISAAC = []
for name in MUJOCO_QPOS_ORDER:
    if name in isaac_to_idx:
        QPOS_TO_ISAAC.append(isaac_to_idx[name])
    else:
        QPOS_TO_ISAAC.append(-1)

print("Loading policy...")
policy = torch.jit.load(POLICY_PATH, map_location="cpu")
policy.eval()
print("✅ Policy loaded")

print("Loading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data  = mujoco.MjData(model)
print(f"✅ nq={model.nq} nv={model.nv} nu={model.nu}")


def reset(model, data):
    mujoco.mj_resetData(model, data)
    # Try to use keyframe if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"Using keyframe 'stand'")
    else:
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = SPAWN_HEIGHT
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qpos[7:35] = QPOS_DEFAULT
        data.ctrl[:28]  = CTRL_DEFAULT  # verified correct standing pose  # verified correct standing pose  # verified correct standing pose  # verified correct standing pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_obs(model, data, last_action_isaac):
    base_quat = data.qpos[3:7].copy()
    rot_mat   = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, base_quat)
    rot_mat   = rot_mat.reshape(3, 3)

    base_lin_vel      = rot_mat.T @ data.qvel[0:3]
    base_ang_vel      = data.qvel[3:6].copy()
    projected_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

    # Read joint states in MuJoCo qpos order → convert to Isaac order
    mj_pos = data.qpos[7:35].copy()   # 28 joints in MUJOCO_QPOS_ORDER
    mj_vel = data.qvel[6:34].copy()   # 28 joints

    isaac_pos_rel = np.zeros(28, dtype=np.float32)
    isaac_vel     = np.zeros(28, dtype=np.float32)

    for mj_i, isaac_i in enumerate(QPOS_TO_ISAAC):
        if isaac_i >= 0:
            default = ISAAC_DEFAULT_ARR[isaac_i]
            isaac_pos_rel[isaac_i] = mj_pos[mj_i] - default
            isaac_vel[isaac_i]     = mj_vel[mj_i]

    obs = np.concatenate([
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        isaac_pos_rel,
        isaac_vel,
        last_action_isaac,
    ]).astype(np.float32)

    return obs


def apply_action(data, action_isaac):
    """
    action_isaac is OFFSET from default (scaled).
    target = default + action * ACTION_SCALE
    Uses CTRL_DEFAULT as base (verified correct joint positions).
    """
    ctrl = CTRL_DEFAULT.copy()
    for isaac_i, ctrl_i in enumerate(ISAAC_TO_CTRL):
        if ctrl_i >= 0:
            ctrl[ctrl_i] = CTRL_DEFAULT[ctrl_i] + action_isaac[isaac_i] * ACTION_SCALE
    data.ctrl[:28] = ctrl


print("\n🚀 Launching MuJoCo viewer...")
print("   Left drag=rotate  Right drag=zoom  Space=pause  ESC=quit\n")

reset(model, data)
last_action_isaac = np.zeros(28, dtype=np.float32)
episode_step  = 0
episode_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -20
    viewer.cam.distance  = 4.0
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

    while viewer.is_running():
        t0 = time.time()

        obs = get_obs(model, data, last_action_isaac)

        with torch.no_grad():
            action_isaac = policy(
                torch.from_numpy(obs).unsqueeze(0)
            ).squeeze().numpy()

        apply_action(data, action_isaac)
        last_action_isaac = action_isaac.copy()

        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(model, data)
        episode_step += 1

        pelvis_h = data.qpos[2]
        fell     = pelvis_h < MIN_HEIGHT
        timeout  = episode_step >= MAX_EPISODE_STEPS

        if fell or timeout:
            episode_count += 1
            status = "timeout ✅" if timeout else "fell    ❌"
            print(f"Episode {episode_count:4d}: {status}  "
                  f"height={pelvis_h:.2f}m  steps={episode_step:3d}")
            reset(model, data)
            last_action_isaac = np.zeros(28, dtype=np.float32)
            episode_step = 0

        viewer.sync()
        dt = time.time() - t0
        if dt < TARGET_DT:
            time.sleep(TARGET_DT - dt)

print(f"\nFinished. Total episodes: {episode_count}")
