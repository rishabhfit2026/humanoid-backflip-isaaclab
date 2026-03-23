"""humanoid velocity task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import humanoid_balance_env_cfg, humanoid_flat_env_cfg, humanoid_rough_env_cfg, humanoid_random_grid_env_cfg
from .rl_cfg import humanoid_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Humanoid",
    env_cfg=humanoid_rough_env_cfg(),
    play_env_cfg=humanoid_rough_env_cfg(play=True),
    rl_cfg=humanoid_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Humanoid",
    env_cfg=humanoid_flat_env_cfg(),
    play_env_cfg=humanoid_flat_env_cfg(play=True),
    rl_cfg=humanoid_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Balance-Humanoid",
    env_cfg=humanoid_balance_env_cfg(),
    play_env_cfg=humanoid_balance_env_cfg(play=True),
    rl_cfg=humanoid_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-RandomGrid-Humanoid",
    env_cfg=humanoid_random_grid_env_cfg(),
    play_env_cfg=humanoid_random_grid_env_cfg(play=True),
    rl_cfg=humanoid_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
