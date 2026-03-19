"""
Backflip environment config for humanoid-mjlab — 28 DOF.
Uses humanoid_pkg/mjcf/humanoid_28dof.xml (same as Isaac Lab).
MuJoCo Warp GPU physics — zero Sim-to-Sim gap!
"""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.backflip import mdp
from mjlab.asset_zoo.robots.humanoid.humanoid_constants import get_humanoid_robot_cfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create 28 DOF humanoid backflip environment."""

    # ── Contact sensor ────────────────────────────────────────────────────────
    feet_contact_cfg = ContactSensorCfg(
        name="contact_forces",
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

    # ── Observations ──────────────────────────────────────────────────────────
    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    rewards = {
        "phase_backflip": RewardTermCfg(
            func=mdp.phase_aware_backflip_reward,
            weight=0.5,
            params={"sensor_name": "contact_forces"},
        ),
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=0.5,
        ),
        "jump_height": RewardTermCfg(
            func=mdp.base_height_reward,
            weight=30.0,
            params={"target_height": 1.5},
        ),

        "landing_smoothness": RewardTermCfg(
            func=mdp.lin_vel_z_l2,
            weight=-0.0001,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),

    }

    # ── Terminations ──────────────────────────────────────────────────────────
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.25},
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    events = {
        "reset_scene": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    # ── Full config ───────────────────────────────────────────────────────────
    return ManagerBasedRlEnvCfg(
        decimation=4,
        episode_length_s=5.0,
        sim=SimulationCfg(
            nconmax=35,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        scene=SceneCfg(
            num_envs=1 if play else 4096,
            env_spacing=2.5,
            entities={"robot": get_humanoid_robot_cfg()},
            sensors=(feet_contact_cfg,),
            terrain=TerrainImporterCfg(terrain_type="plane"),
        ),
        observations={
            "policy": ObservationGroupCfg(terms=policy_terms),
            "critic": ObservationGroupCfg(terms=policy_terms),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
            actuator_names=(".*",),
                scale=0.5,
                use_default_offset=True,
            )
        },
        rewards=rewards,
        terminations=terminations,
        events=events,
        viewer=ViewerConfig(
            lookat=(0.0, 0.0, 1.0),
            distance=4.0,
            elevation=-20.0,
            azimuth=90.0,
        ),
    )


def backflip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config — same as Isaac Lab backflip training."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
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
        experiment_name="humanoid_backflip_28dof_mjlab",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=8000,
    )
