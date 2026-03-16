"""
scripts/train.py
================
Usage:
    python scripts/train.py --task 28dof_env --num_envs 64 --headless
"""

import argparse
import os
import sys

# Add Isaac Lab to path
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_tasks"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task",           type=str, default="28dof_env")
parser.add_argument("--num_envs",       type=int, default=4096)
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--resume",         action="store_true")
parser.add_argument("--checkpoint",     type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import config.register_task  # noqa
from config.agent_cfg import agent_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=agent_cfg.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"\n✅ Env ready | obs={env.num_obs} | actions={env.num_actions} | envs={args_cli.num_envs}\n")

    log_dir = os.path.join("logs", "28dof_env")
    runner  = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if args_cli.resume:
        ckpt = args_cli.checkpoint or get_checkpoint_path(log_dir, ".*")
        print(f"▶ Resuming: {ckpt}")
        runner.load(ckpt)

    print("🚀 Training started. Watch: tensorboard --logdir logs/\n")
    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(
        runner.alg.actor_critic,
        runner.obs_normalizer,
        path=export_dir,
        filename="policy.pt",
    )
    export_policy_as_onnx(
        runner.alg.actor_critic,
        normalizer=runner.obs_normalizer,
        path=os.path.join(export_dir, "policy.onnx"),
    )

    print(f"\n✅ Policy exported to {export_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()