"""Asimov bipedal robot with toe joints velocity tracking environment configurations."""

from mjlab.asset_zoo.robots.asimov.asimov_toe_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def asimov_toe_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Asimov rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  # Increase nconmax for capsule-based foot collisions
  cfg.sim.nconmax = 50

  cfg.scene.entities = {"robot": get_asimov_robot_cfg()}

  # Asimov feet sites at ankle roll joints
  site_names = ("left_ankle_roll_joint_site", "right_ankle_roll_joint_site")
  # Foot and toe capsule collision geometries
  geom_names = (
    r"left_foot\d+_collision",
    r"left_toe\d+_collision",
    r"right_foot\d+_collision",
    r"right_toe\d+_collision",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Action scales for all joints except toes (12 DOF: hip/knee/ankle)
  action_scale_no_toe = {
    k: v for k, v in ASIMOV_ACTION_SCALE.items() if "toe" not in k
  }

  # Direct joint position control for all actuators except toes
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(r"^(?!.*toe).*$",),  # exclude only toes
      scale=action_scale_no_toe,
      use_default_offset=True,
      preserve_order=True,
    ),
  }

  cfg.viewer.body_name = "pelvis_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8  # Asimov is shorter than G1

  # Conservative velocity commands for initial training:
  # 1. Forward-only (no backward/lateral) to simplify learning
  # 2. Wider turning range to encourage dynamic motion
  # 3. Can increase complexity after stable forward walking is learned
  twist_cmd.ranges.lin_vel_x = (0.0, 0.8)   # Forward only (no backward)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)   # No lateral movement initially
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)  # Wider turning range

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Larger variance for canted hip pitch (allows coupled roll/pitch motion)
    r".*hip_pitch.*": 0.5,
    # Hip roll: reduced for ±45° range (was wider before Alex's corrections)
    r".*hip_roll.*": 0.12,
    # Hip yaw: reduced for ±45° range (was wider before Alex's corrections)
    r".*hip_yaw.*": 0.1,
    # Large knee variance (coordinate system corrected to match hardware)
    r".*knee.*": 0.5,
    # Lower ankle variance due to limited ROM (~±20° roll, asymmetric pitch)
    r".*ankle_pitch.*": 0.2,
    r".*ankle_roll.*": 0.12,
    # Toe joints - passive, low variance
    r".*toe.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Even larger variance for dynamic motion with canted hips
    r".*hip_pitch.*": 0.8,
    # Hip roll: reduced for ±45° range (was wider before Alex's corrections)
    r".*hip_roll.*": 0.18,
    # Hip yaw: reduced for ±45° range (was wider before Alex's corrections)
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.8,
    # Keep ankles constrained even when running
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.15,
    # Toe joints - allow more motion during running
    r".*toe.*": 0.4,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Increase body angular velocity penalty (narrow stance = less stable)
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  # Enable air time reward for lighter robot (better for jumping)
  # Balanced at 1.0 to work with foot clearance penalties
  cfg.rewards["air_time"].weight = 1.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Customize observations:
  # - Remove any linear velocity term.
  # - Exclude toe joints from joint_pos / joint_vel (keep 12 main leg joints).
  # - Ensure policy observation ordering matches expected layout.
  assert cfg.observations is not None
  assert "policy" in cfg.observations
  assert "critic" in cfg.observations
  policy_obs = cfg.observations["policy"]
  critic_obs = cfg.observations["critic"]

  # Remove linear velocity observation if present.
  policy_obs.terms.pop("base_lin_vel", None)
  critic_obs.terms.pop("base_lin_vel", None)

  # Restrict joint observations to 12 non-toe joints.
  joint_asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=(
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
    ),
  )
  for terms in (policy_obs.terms, critic_obs.terms):
    for name in ("joint_pos", "joint_vel"):
      if name in terms:
        terms[name].params["asset_cfg"] = joint_asset_cfg

  # Rename the command observation to velocity_commands for clarity.
  if "command" in policy_obs.terms:
    policy_obs.terms["velocity_commands"] = policy_obs.terms.pop("command")
  if "command" in critic_obs.terms:
    critic_obs.terms["velocity_commands"] = critic_obs.terms.pop("command")

  # Reorder policy terms to match the desired layout:
  # base_ang_vel (3), projected_gravity (3), velocity_commands (3),
  # joint_pos (12), joint_vel (12), actions (12).
  ordered_policy_terms = {}
  for name in (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
  ):
    if name in policy_obs.terms:
      ordered_policy_terms[name] = policy_obs.terms[name]
  # Append any remaining terms (if present) to preserve them.
  for name, term in policy_obs.terms.items():
    if name not in ordered_policy_terms:
      ordered_policy_terms[name] = term
  policy_obs.terms = ordered_policy_terms

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def asimov_toe_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Asimov flat terrain velocity tracking configuration."""
  cfg = asimov_toe_rough_env_cfg(play=play)

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  return cfg


# Backwards compatibility aliases
ASIMOV_ROUGH_ENV_CFG = asimov_toe_rough_env_cfg()
ASIMOV_FLAT_ENV_CFG = asimov_toe_flat_env_cfg()
