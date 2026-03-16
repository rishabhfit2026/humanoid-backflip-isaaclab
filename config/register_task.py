"""
config/register_task.py
=======================
Registers 28dof_env with Isaac Lab gym registry.
Updated for new isaaclab imports.
"""

import gymnasium as gym

gym.register(
    id="28dof_env",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.backflip_env_cfg:HumanoidBackflipEnvCfg",
        "rsl_rl_cfg_entry_point": "config.agent_cfg:agent_cfg",
    },
)

gym.register(
    id="28dof_env-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.backflip_env_cfg:HumanoidBackflipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "config.agent_cfg:agent_cfg",
    },
)