"""
mdp/rewards.py
==============
Custom reward functions — updated for new isaaclab imports
"""

from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward peaked at target_height — encourages jumping."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.exp(-((height - target_height) ** 2) / (2 * 0.2 ** 2))


def base_ang_vel_reward(
    env: ManagerBasedRLEnv,
    axis: str,
    target_ang_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward angular velocity on a given axis.
    Backflip: axis='y', target_ang_vel=-6.0 (backward spin).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    axis_map = {"x": 0, "y": 1, "z": 2}
    vel = asset.data.root_ang_vel_b[:, axis_map[axis]]
    if target_ang_vel < 0:
        return torch.tanh(-vel / abs(target_ang_vel))
    return torch.tanh(vel / abs(target_ang_vel))


def upright_posture_reward(
    env: ManagerBasedRLEnv,
    upright_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for being upright after landing."""
    asset: Articulation = env.scene[asset_cfg.name]
    uprightness = -asset.data.projected_gravity_b[:, 2]
    return torch.clamp(uprightness - upright_threshold, min=0.0)


def phase_aware_backflip_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phase-aware reward: jump → rotate backward → land cleanly."""
    asset:  Articulation  = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    air_time  = sensor.data.current_air_time[:, 0]
    height    = asset.data.root_pos_w[:, 2]
    ang_vel_y = asset.data.root_ang_vel_b[:, 1]

    is_airborne = air_time > 0.05
    is_rotating = is_airborne & (ang_vel_y < -1.0)
    is_landing  = is_airborne & (height < 0.9) & (ang_vel_y > -1.0)

    zeros    = torch.zeros_like(height)
    r_jump   = torch.where(~is_airborne, asset.data.root_lin_vel_b[:, 2].clamp(min=0), zeros)
    r_rotate = torch.where(is_rotating,  (-ang_vel_y).clamp(max=10.0), zeros)
    r_land   = torch.where(is_landing,   (-ang_vel_y).clamp(min=0),    zeros)

    return 2.0 * r_jump + 3.0 * r_rotate + 2.0 * r_land