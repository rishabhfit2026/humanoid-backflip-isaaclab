"""
envs/backflip_env_cfg.py  —  humanoid_28dof.urdf
=================================================
Updated for Isaac Lab v2.3.2 with correct imports
28 revolute DOF humanoid robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from mdp.rewards import base_height_reward, base_ang_vel_reward, upright_posture_reward


@configclass
class BackflipSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rishabh/backflip_project/assets/robots/humanoid_28dof_converted.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.2),
            joint_pos={
                "left_hip_pitch_joint":             -0.20,
                "right_hip_pitch_joint":            -0.20,
                "left_hip_roll_joint":               0.0,
                "right_hip_roll__joint":             0.0,
                "left_hip_yaw_joint":                0.0,
                "right_hip_yaw_joint":               0.0,
                "left_knee_joint":                   0.45,
                "right_knee_joint":                  0.45,
                "left_foot1_joint":                 -0.20,
                "right_foot1_joint":                -0.20,
                "left_foot2_joint":                  0.0,
                "right_foot2_joint":                 0.0,
                "torso_joint":                       0.0,
                "waist_roll_joint":                  0.0,
                "waist_pitch_joint":                 0.0,
                "waist_rod_joint":                   0.0,
                "left_shoulder_pitch_joint":         0.0,
                "right_shoulder_pitch_joint":        0.0,
                "left_shoulder_roll_joint":          0.3,
                "right_shoulder_roll_joint":        -0.3,
                "left_elbow_yaw_joint":              0.0,
                "right_elbow_yaw_joint":             0.0,
                "left_elbow_forearm_yaw_joint":      0.5,
                "right_elbow_forearm_yaw_joint":     0.5,
                "left_wrist_yaw_joint":              0.0,
                "right_wrist_yaw_joint":             0.0,
                "neck_yaw_joint":                    0.0,
                "head_joint":                        0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hips": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_hip_pitch_joint",  "right_hip_pitch_joint",
                    "left_hip_roll_joint",   "right_hip_roll__joint",
                    "left_hip_yaw_joint",    "right_hip_yaw_joint",
                ],
                effort_limit=200,
                velocity_limit=10.0,
                stiffness={
                    ".*hip_pitch.*": 150.0,
                    ".*hip_roll.*":  100.0,
                    ".*hip_yaw.*":    80.0,
                },
                damping={
                    ".*hip_pitch.*": 5.0,
                    ".*hip_roll.*":  4.0,
                    ".*hip_yaw.*":   3.0,
                },
            ),
            "knees_feet": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_knee_joint",   "right_knee_joint",
                    "left_foot1_joint",  "right_foot1_joint",
                    "left_foot2_joint",  "right_foot2_joint",
                ],
                effort_limit=200,
                velocity_limit=10.0,
                stiffness={
                    ".*knee.*":  200.0,
                    ".*foot1.*":  40.0,
                    ".*foot2.*":  30.0,
                },
                damping={
                    ".*knee.*":  5.0,
                    ".*foot1.*": 2.0,
                    ".*foot2.*": 1.5,
                },
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=[
                    "torso_joint", "waist_roll_joint",
                    "waist_pitch_joint", "waist_rod_joint",
                ],
                effort_limit=80,
                velocity_limit=5.0,
                stiffness=100.0,
                damping=4.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_shoulder_pitch_joint",    "right_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",     "right_shoulder_roll_joint",
                    "left_elbow_yaw_joint",         "right_elbow_yaw_joint",
                    "left_elbow_forearm_yaw_joint", "right_elbow_forearm_yaw_joint",
                    "left_wrist_yaw_joint",         "right_wrist_yaw_joint",
                ],
                effort_limit=30,
                velocity_limit=10.0,
                stiffness={
                    ".*shoulder.*": 40.0,
                    ".*elbow.*":    25.0,
                    ".*wrist.*":    10.0,
                },
                damping={
                    ".*shoulder.*": 2.0,
                    ".*elbow.*":    1.5,
                    ".*wrist.*":    0.5,
                },
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["neck_yaw_joint", "head_joint"],
                effort_limit=5,
                velocity_limit=3.0,
                stiffness=20.0,
                damping=1.0,
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,
                                    noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,
                                    noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel,
                                    noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel,
                                    noise=Unoise(n_min=-1.5,  n_max=1.5))
        actions           = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Joint position targets for all 28 DOF."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    backflip_rotation = RewTerm(
        func=base_ang_vel_reward, weight=8.0,
        params={"axis": "y", "target_ang_vel": -6.0},
    )
    upright_after_land = RewTerm(
        func=upright_posture_reward, weight=6.0,
        params={"upright_threshold": 0.85},
    )
    jump_height = RewTerm(
        func=base_height_reward, weight=3.0,
        params={"target_height": 1.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    landing_smoothness = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.001)
    dof_torques_l2     = RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)
    action_rate_l2     = RewTerm(func=mdp.action_rate_l2,   weight=-0.01)
    knee_deviation = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
            joint_names=["left_knee_joint", "right_knee_joint"])},
    )
    waist_deviation = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
            joint_names=["waist_roll_joint", "waist_pitch_joint"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["pelvis"]),
            "threshold": 1.0,
        },
    )
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25},
    )


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(8.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.08, 0.08), "velocity_range": (-0.1, 0.1)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class HumanoidBackflipEnvCfg(ManagerBasedRLEnvCfg):
    scene        : BackflipSceneCfg = BackflipSceneCfg(num_envs=4096, env_spacing=2.5)
    observations : ObservationsCfg  = ObservationsCfg()
    actions      : ActionsCfg       = ActionsCfg()
    rewards      : RewardsCfg       = RewardsCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    events       : EventCfg         = EventCfg()

    def __post_init__(self):
        self.decimation          = 4
        self.episode_length_s    = 5.0
        self.sim.dt              = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity               = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity      = 16 * 1024
        self.sim.physx.friction_correlation_distance           = 0.00625


@configclass
class HumanoidBackflipEnvCfg_PLAY(HumanoidBackflipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


##############################################################################
# RSL-RL PPO Agent Config
##############################################################################

agent_cfg = RslRlOnPolicyRunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=24,
    max_iterations=10_000,
    save_interval=100,
    experiment_name="28dof_backflip",
    empirical_normalization=False,
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)