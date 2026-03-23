"""humanoid bipedal robot velocity tracking environment configurations."""

from mjlab.asset_zoo.robots.humanoid.humanoid_constants import (
  HUMANOID_ACTION_SCALE,
  HUMANOID_WALKING_REFERENCE,
  get_humanoid_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import RANDOM_GRID_TERRAINS_CFG


def humanoid_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_humanoid_robot_cfg()}

  # humanoid feet sites
  site_names = ("left_foot", "right_foot")
  geom_names = (
    "left_foot2_link_collision",
    "right_foot2_link_collision",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot2_link|right_foot2_link)$",
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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = HUMANOID_ACTION_SCALE

  cfg.viewer.body_name = "pelvis_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8  # humanoid is shorter than G1

  # More conservative velocity commands due to:
  # 1. Narrow stance (11.3 cm hip width)
  # 2. Canted hip pitch (less stable)
  # 3. Limited ankle ROM
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  # Remove base_lin_vel - not available on real robot IMU
  del cfg.observations["policy"].terms["base_lin_vel"]
  del cfg.observations["critic"].terms["base_lin_vel"]

  # Add gait clock observation for phase-aware locomotion
  gait_clock_cfg = ObservationTermCfg(
    func=mdp.gait_clock,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "gait_frequency": 1.25,  # Match imitation data frequency
    },
  )
  cfg.observations["policy"].terms["gait_clock"] = gait_clock_cfg
  cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Larger variance for canted hip pitch (allows coupled roll/pitch motion)
    r".*hip_pitch.*": 0.5,
    # Moderate hip roll (asymmetric ranges)
    r".*hip_roll.*": 0.25,
    # Standard hip yaw
    r".*hip_yaw.*": 0.2,
    # Large knee variance (extends backwards, different from G1)
    r".*knee.*": 0.5,
    # Lower ankle variance due to limited ROM (±20° pitch, ±15° roll)
    r".*foot1.*": 0.2,
    r".*foot2.*": 0.12,
    r".*torso.*": 0.3,
    r".*waist_roll.*": 0.3,
    r".*waist_pitch.*": 0.1,
    r".*shoulder.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist_yaw.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Even larger variance for dynamic motion with canted hips
    r".*hip_pitch.*": 0.8,
    r".*hip_roll.*": 0.35,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.8,
    # Keep feet constrained even when running
    r".*foot1.*": 0.25,
    r".*foot2.*": 0.15,
    r".*torso.*": 0.3,
    r".*waist_roll.*": 0.15,
    r".*waist_pitch.*": 0.15,
    r".*shoulder.*": 0.15,
    r".*elbow.*": 0.15,
    r".*wrist_yaw.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Increase body angular velocity penalty (narrow stance = less stable)
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  # Enable air time reward for lighter robot (better for jumping)
  cfg.rewards["air_time"].weight = 0.5

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Imitation reward for walking reference motion
  # Only lower body joints (12) match the walking reference CSV
  imitation_asset_cfg = SceneEntityCfg(name="robot")
  imitation_asset_cfg.joint_names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_foot1_joint",
    "left_foot2_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_foot1_joint",
    "right_foot2_joint",
  )
  cfg.rewards["imitation"] = RewardTermCfg(
    func=mdp.imitation_joint_pos,
    weight=0.0,
    params={
      "data_path": str(HUMANOID_WALKING_REFERENCE),
      "gait_frequency": 1.25,
      "std": 0.5,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": imitation_asset_cfg,
    },
  )

  # Alternating feet contact reward for proper bipedal gait
  cfg.rewards["alternating_feet"] = RewardTermCfg(
    func=mdp.alternating_feet_contact,
    weight=0.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
      "gait_frequency": 1.25,  # Match imitation data frequency
    },
  )

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




def humanoid_random_grid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid random grid terrain velocity tracking configuration."""
  cfg = humanoid_rough_env_cfg(play=play)

  # Switch to random grid terrain
  from dataclasses import replace
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(RANDOM_GRID_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 0

  # Increase contact limits for random grid terrain
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 2000

  # Smaller terrain = easier curriculum threshold
  from dataclasses import replace as dc_replace
  cfg.scene.terrain.terrain_generator = dc_replace(
      cfg.scene.terrain.terrain_generator,
      size=(4.0, 4.0)
  )
  return cfg

def humanoid_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid flat terrain velocity tracking configuration."""
  cfg = humanoid_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def humanoid_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid balance-only config with reactive stepping.

  The robot learns to balance in place and take small corrective steps
  when disturbed. No velocity tracking - just stay upright.
  """
  cfg = humanoid_flat_env_cfg(play=play)

  # Zero out velocity commands (not trying to go anywhere)
  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  # Disable velocity tracking rewards (not trying to go anywhere)
  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0

  # Keep foot rewards for reactive stepping
  # Keep upright, pose, body_ang_vel for balance

  return cfg
