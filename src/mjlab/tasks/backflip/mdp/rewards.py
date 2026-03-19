"""Backflip reward functions for humanoid-mjlab."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward peaked at target_height."""
    asset: Entity = env.scene[asset_cfg.name]
    # Use root body position — first body position in world frame
    height = asset.data.body_link_pos_w[:, 0, 2]  # pelvis height
    return torch.exp(-((height - target_height) ** 2) / (2 * 0.2 ** 2))


def upright_posture_reward(
    env: ManagerBasedRlEnv,
    upright_threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for being upright after landing."""
    asset: Entity = env.scene[asset_cfg.name]
    uprightness = -asset.data.projected_gravity_b[:, 2]
    return torch.clamp(uprightness - upright_threshold, min=0.0)


def phase_aware_backflip_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "contact_forces",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Phase-aware backflip reward."""
    asset:  Entity        = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    air_time    = sensor.data.current_air_time[:, 0]
    height      = asset.data.body_link_pos_w[:, 0, 2]       # pelvis height
    ang_vel_y   = asset.data.root_link_ang_vel_b[:, 1]      # pitch angular velocity
    lin_vel_z   = asset.data.root_link_lin_vel_b[:, 2]      # vertical velocity
    uprightness = -asset.data.projected_gravity_b[:, 2]     # 1=upright

    is_grounded = air_time < 0.05
    is_airborne = ~is_grounded
    is_rotating = is_airborne & (ang_vel_y < -1.0)
    is_landing  = is_airborne & (height < 1.0) & (ang_vel_y > -1.0)

    zeros = torch.zeros_like(height)

    r_jump   = torch.where(is_grounded & (lin_vel_z > 0.0),
                           lin_vel_z.clamp(0.0, 5.0), zeros)
    r_rotate = torch.where(is_rotating,
                           (-ang_vel_y).clamp(0.0, 6.0), zeros)
    r_land   = torch.where(is_landing,
                           uprightness.clamp(min=0.0), zeros)

    total = 2.0 * r_jump + 4.0 * r_rotate + 3.0 * r_land
    return torch.clamp(total, 0.0, 10.0)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize large changes in actions."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def joint_torques_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize large joint torques."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return (~env.termination_manager.terminated).float()
