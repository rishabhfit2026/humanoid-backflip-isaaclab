"""
mdp/rewards.py
==============
Custom reward functions — updated for new isaaclab imports
Phase-aware backflip reward with proper jump/land balance
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
    """
    Phase-aware reward for backflip with three sequential phases:

    Phase 1 — Takeoff:
        Reward upward push-off velocity while feet are on ground.
        Encourages explosive jump before rotation.

    Phase 2 — Rotate:
        Reward backward angular velocity (ang_vel_y < 0) while airborne.
        Clamped to max 6 rad/s to prevent reward hacking.

    Phase 3 — Land:
        Reward uprightness (gravity pointing down in body frame)
        when robot is coming back down (height < 1.0m, decelerating spin).
        Encourages clean upright landing.

    Total is hard-clamped to [0, 10] to prevent value function explosion.
    """
    asset:  Articulation  = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # ── State variables ───────────────────────────────────────────────────────
    air_time  = sensor.data.current_air_time[:, 0]   # seconds since last contact
    height    = asset.data.root_pos_w[:, 2]           # pelvis height (m)
    ang_vel_y = asset.data.root_ang_vel_b[:, 1]       # pitch angular velocity (rad/s)
    lin_vel_z = asset.data.root_lin_vel_b[:, 2]       # vertical velocity (m/s)
    uprightness = -asset.data.projected_gravity_b[:, 2]  # 1=upright, -1=upside down

    # ── Phase detection ───────────────────────────────────────────────────────
    is_grounded = air_time < 0.05                     # feet touching ground
    is_airborne = ~is_grounded                        # feet off ground
    is_rotating = is_airborne & (ang_vel_y < -1.0)    # spinning backward
    is_landing  = (                                   # coming down to land
        is_airborne
        & (height < 1.0)
        & (ang_vel_y > -1.0)
    )

    zeros = torch.zeros_like(height)

    # ── Phase 1: Takeoff ──────────────────────────────────────────────────────
    # Reward upward velocity ONLY when grounded (push-off phase)
    # Clamp to 5 m/s max — prevents reward for infinite upward velocity
    r_jump = torch.where(
        is_grounded & (lin_vel_z > 0.0),
        lin_vel_z.clamp(min=0.0, max=5.0),
        zeros,
    )

    # ── Phase 2: Rotate ───────────────────────────────────────────────────────
    # Reward backward spin when airborne
    # Clamp to 6 rad/s max — prevents reward hacking via uncontrolled spin
    r_rotate = torch.where(
        is_rotating,
        (-ang_vel_y).clamp(min=0.0, max=6.0),
        zeros,
    )

    # ── Phase 3: Land ─────────────────────────────────────────────────────────
    # Reward uprightness when coming back down
    # Robot must be upright (gravity pointing down) to get this reward
    r_land = torch.where(
        is_landing,
        uprightness.clamp(min=0.0),
        zeros,
    )

    # ── Combine and clamp ─────────────────────────────────────────────────────
    # Jump gets lower weight (2) — it's easier to learn
    # Rotate gets highest weight (4) — hardest and most important
    # Land gets medium weight (3) — critical for full backflip
    total = 2.0 * r_jump + 4.0 * r_rotate + 3.0 * r_land

    # Hard cap prevents value function explosion (seen at iter 77 previously)
    return torch.clamp(total, min=0.0, max=10.0)