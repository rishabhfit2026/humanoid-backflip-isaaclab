"""Backflip MDP — velocity mdp + backflip rewards."""
from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403
from .rewards import (  # noqa: F401
    base_height_reward,
    upright_posture_reward,
    phase_aware_backflip_reward,
    lin_vel_z_l2,
    action_rate_l2,
    joint_torques_l2,
    is_alive,
)
