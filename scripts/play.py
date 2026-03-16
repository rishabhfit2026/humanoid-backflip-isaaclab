"""
scripts/play.py
===============
Usage:
    python scripts/play.py --checkpoint logs/28dof_env/model_10000.pt
"""

import argparse
import os
import sys

sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_tasks"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task",       type=str, default="28dof_env-Play")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import config.register_task  # noqa
from config.agent_cfg import agent_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )
    env    = gym.make(args_cli.task, cfg=env_cfg)
    env    = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.get_observations()
    step   = 0
    ep_rew = torch.zeros(args_cli.num_envs, device="cuda:0")

    print(f"\n▶ Playing: {args_cli.checkpoint}\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        ep_rew += rewards
        step   += 1

        if step % 200 == 0:
            print(f"  step {step:5d} | mean reward: {ep_rew.mean().item():.2f}")
            ep_rew.zero_()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()