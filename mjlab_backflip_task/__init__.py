"""28-DOF Humanoid Backflip Task."""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .backflip_env_cfg import backflip_env_cfg, backflip_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Backflip-Humanoid-28DOF",
    env_cfg=backflip_env_cfg(),
    play_env_cfg=backflip_env_cfg(play=True),
    rl_cfg=backflip_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
