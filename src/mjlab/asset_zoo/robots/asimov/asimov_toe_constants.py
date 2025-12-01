"""Asimov bipedal robot with toe joints constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator import BuiltinPositionActuatorCfg, LearnedMlpActuatorCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ASIMOV_TOE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "asimov" / "xmls" / "asimov_toe.xml"
)
assert ASIMOV_TOE_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ASIMOV_TOE_XML.parent.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ASIMOV_TOE_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Reflected inertia values from ENCOS motor specs (stats.md)
# J_reflected = J_rotor * gear_ratio^2
# Units: kg·m²
REFLECTED_INERTIA = {
    "hip_pitch": 0.0652,  # EC-A6416-P2-25: 104.395 kg·mm² * 25^2
    "hip_roll": 0.100,    # EC-A5013-H17-100: 10 kg·mm² * 100^2
    "hip_yaw": 0.0343,    # EC-A3814-H14-107: 3 kg·mm² * 107^2
    "knee": 0.0330,       # EC-A4315-P2-36: 25.5 kg·mm² * 36^2
    "ankle": 0.0472,      # EC-A4310-P2-36: 18.2 kg·mm² * 36^2 * 2 (parallel linkage)
}

# Identified gains from CAN data (symmetric L/R, Kd capped at 5.0)
# These are the actual gains measured from hardware system identification.
IDENTIFIED_GAINS = {
    "hip_pitch": {"kp": 22.5, "kd": 1.4},
    "hip_roll": {"kp": 118.0, "kd": 5.0},
    "hip_yaw": {"kp": 130.0, "kd": 5.0},
    "knee": {"kp": 84.0, "kd": 4.2},
    "ankle_pitch": {"kp": 14.0, "kd": 1.7},
    "ankle_roll": {"kp": 17.0, "kd": 1.1},
}

# Asimov joint actuator configuration
# Updated with torque limits from encos based on current limits:
# [55, 90, 60, 50, 36, 36, 55, 90, 60, 50, 36, 36]
# Order: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (per leg)

# Hip pitch joints - 55 Nm
ASIMOV_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_hip_pitch_joint",),
  effort_limit=55.0,
  armature=REFLECTED_INERTIA["hip_pitch"],
  stiffness=IDENTIFIED_GAINS["hip_pitch"]["kp"],
  damping=IDENTIFIED_GAINS["hip_pitch"]["kd"],
)

# Hip roll joints - 90 Nm
ASIMOV_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_hip_roll_joint",),
  effort_limit=90.0,
  armature=REFLECTED_INERTIA["hip_roll"],
  stiffness=IDENTIFIED_GAINS["hip_roll"]["kp"],
  damping=IDENTIFIED_GAINS["hip_roll"]["kd"],
)

# Hip yaw joints - 60 Nm
ASIMOV_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_hip_yaw_joint",),
  effort_limit=60.0,
  armature=REFLECTED_INERTIA["hip_yaw"],
  stiffness=IDENTIFIED_GAINS["hip_yaw"]["kp"],
  damping=IDENTIFIED_GAINS["hip_yaw"]["kd"],
)

# Knee joints - 50 Nm
ASIMOV_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_knee_joint",),
  effort_limit=50.0,
  armature=REFLECTED_INERTIA["knee"],
  stiffness=IDENTIFIED_GAINS["knee"]["kp"],
  damping=IDENTIFIED_GAINS["knee"]["kd"],
)

# Ankle pitch joints - 72 Nm (36 Nm * 2 for parallel linkage)
ASIMOV_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_ankle_pitch_joint",),
  effort_limit=72.0,
  armature=REFLECTED_INERTIA["ankle"],
  stiffness=IDENTIFIED_GAINS["ankle_pitch"]["kp"],
  damping=IDENTIFIED_GAINS["ankle_pitch"]["kd"],
)

# Ankle roll joints - 72 Nm (36 Nm * 2 for parallel linkage)
ASIMOV_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  joint_names_expr=(".*_ankle_roll_joint",),
  effort_limit=72.0,
  armature=REFLECTED_INERTIA["ankle"],
  stiffness=IDENTIFIED_GAINS["ankle_roll"]["kp"],
  damping=IDENTIFIED_GAINS["ankle_roll"]["kd"],
)

# Toe joints - passive spring with low control authority
ASIMOV_TOE_ACTUATOR = BuiltinPositionActuatorCfg(
  joint_names_expr=("left_toe_joint", "right_toe_joint"),
  effort_limit=5.0,  # From URDF
  armature=0.0001,  # Minimal inertia
  stiffness=50.0,   # From URDF spring_stiffness
  damping=0.8,      # From URDF damping
)

##
# Learned actuator config.
##

# Path to trained actuator network (TorchScript model).
# Train using: python scripts/train_asimov_actuator.py --csv data.csv --output <path>
ASIMOV_LEARNED_ACTUATOR_PATH = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "asimov"
  / "assets"
  / "asimov_actuator.pt"
)

# Learned MLP actuator for hip, knee, and ankle joints (excluding toe).
# Network architecture: MLP with 32 hidden units, 2 layers, softsign activation.
# Input: [pos_error[t-2:t], vel[t-2:t]] where pos_error = target - current
# This matches mjlab's LearnedMlpActuator convention.
ASIMOV_LEARNED_ACTUATOR_CFG = LearnedMlpActuatorCfg(
  joint_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  network_file=str(ASIMOV_LEARNED_ACTUATOR_PATH),
  pos_scale=1.0,    # Network trained with (target - current), matches mjlab
  vel_scale=1.0,
  torque_scale=1.0,
  input_order="pos_vel",
  history_length=3,
  # Max effort from hip_roll (90 Nm).
  saturation_effort=90.0,
  velocity_limit=32.0,
  effort_limit=90.0,
  # Average armature across joint types.
  armature=REFLECTED_INERTIA["hip_pitch"],
)

##
# Keyframe config.
##

# Default standing pose for Asimov (legs straight)
STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.71),  # Adjusted for toe geometry at ground level
  joint_pos={
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,   # Left: 0 = straight
    "right_knee_joint": 0.0,  # Right: 0 = straight (both at neutral)
    ".*_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    # Toe joints - try reference position (0.0) first
    "left_toe_joint": 0.0,   # At reference position
    "right_toe_joint": 0.0,   # At reference position
  },
  joint_vel={".*": 0.0},
)

# Knees bent pose for stability
# NOTE: Coordinate system corrected to match real hardware (Alex's update)
# Knees:
# - Left knee: axis=(0, 1, 0), range=[0, 1.5] → positive = extend back
# - Right knee: axis=(0, -1, 0), range=[-1.5, 0] → negative = extend back
# Ankle pitch:
# - Left ankle: axis=(0, 1, 0) → positive = pitch up
# - Right ankle: axis=(0, -1, 0) → negative = pitch up
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.73),  # Adjusted for toe geometry at ground level
  joint_pos={
    # Based on proven booster_gym Asimov_unitree.yaml config
    "left_hip_pitch_joint": 0.2,      # More upright than G1
    "right_hip_pitch_joint": -0.2,    # Opposite due to canted axis
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,           # Left: positive = extend backwards (corrected)
    "right_knee_joint": -0.4,         # Right: negative = extend backwards (corrected)
    "left_ankle_pitch_joint": -0.25,  # From v1.yaml config
    "right_ankle_pitch_joint": 0.25,  # Opposite axis
    ".*_ankle_roll_joint": 0.0,
    # Toe joints - try reference position (0.0) first
    "left_toe_joint": 0.0,            # At reference position
    "right_toe_joint": 0.0,            # At reference position
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Enable all collisions including self collisions
# Feet get condim=3, other parts get condim=1
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_(foot|toe)\d+_collision$": 3,
    ".*_collision": 1
  },
  priority={r"^(left|right)_(foot|toe)\d+_collision$": 1},
  friction={r"^(left|right)_(foot|toe)\d+_collision$": (0.8,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_(foot|toe)\d+_collision$": 3,
    ".*_collision": 1
  },
  priority={r"^(left|right)_(foot|toe)\d+_collision$": 1},
  friction={r"^(left|right)_(foot|toe)\d+_collision$": (0.8,)},
)

# Only feet collision enabled (recommended for initial training)
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_(foot|toe)\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

ASIMOV_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ASIMOV_ACTUATOR_HIP_PITCH,
    ASIMOV_ACTUATOR_HIP_ROLL,
    ASIMOV_ACTUATOR_HIP_YAW,
    ASIMOV_ACTUATOR_KNEE,
    ASIMOV_ACTUATOR_ANKLE_PITCH,
    ASIMOV_ACTUATOR_ANKLE_ROLL,
    ASIMOV_TOE_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Learned actuator articulation (replaces builtin PD with learned network).
ASIMOV_ARTICULATION_LEARNED = EntityArticulationInfoCfg(
  actuators=(
    ASIMOV_LEARNED_ACTUATOR_CFG,
    ASIMOV_TOE_ACTUATOR,  # Keep toe as builtin (passive spring)
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_asimov_robot_cfg() -> EntityCfg:
  """Get a fresh Asimov robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ASIMOV_ARTICULATION,
  )


def get_asimov_robot_cfg_learned() -> EntityCfg:
  """Get Asimov robot with learned actuator network.

  Uses a trained MLP to predict torques from joint state history.
  Train the network using: python scripts/train_asimov_actuator.py

  Returns:
    EntityCfg configured with learned actuator model.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ASIMOV_ARTICULATION_LEARNED,
  )


# Compute action scales for each joint
# Using 0.25 multiplier (same as G1) for stable control on lighter robot
ASIMOV_ACTION_SCALE: dict[str, float] = {}
for a in ASIMOV_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  assert e is not None
  for n in names:
    # 0.25 multiplier (same as G1) for stable control on lighter robot
    ASIMOV_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_asimov_robot_cfg())

  viewer.launch(robot.spec.compile())
