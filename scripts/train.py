"""
scripts/train.py
================
Training script for 28-DOF humanoid backflip policy.

Usage:
    python scripts/train.py --task 28dof_env --num_envs 4096 --headless
    python scripts/train.py --task 28dof_env --num_envs 4096 --headless --resume
    python scripts/train.py --task 28dof_env --num_envs 4096 --headless --max_iterations 10000
"""

import argparse
import os
import sys

# ── Isaac Lab path setup ──────────────────────────────────────────────────────
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_tasks"))
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab_rl"))

from isaaclab.app import AppLauncher

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train 28-DOF humanoid backflip policy")
parser.add_argument("--task",           type=str,   default="28dof_env",
                    help="Registered task name")
parser.add_argument("--num_envs",       type=int,   default=4096,
                    help="Number of parallel environments")
parser.add_argument("--seed",           type=int,   default=42,
                    help="Random seed")
parser.add_argument("--max_iterations", type=int,   default=10_000,
                    help="Total PPO iterations — 8k-10k recommended for phase-aware reward")
parser.add_argument("--resume",         action="store_true",
                    help="Resume from latest checkpoint")
parser.add_argument("--checkpoint",     type=str,   default=None,
                    help="Path to specific checkpoint to resume from")
parser.add_argument("--log_dir",        type=str,   default=None,
                    help="Override default log directory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── Launch Isaac Sim ──────────────────────────────────────────────────────────
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ──────────────────────────────────────────────────
import torch
import gymnasium as gym
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import config.register_task  # noqa: F401
from config.agent_cfg import agent_cfg


def main():
    # ── Override seed if provided ─────────────────────────────────────────────
    agent_cfg.seed = args_cli.seed

    # ── Override max iterations if provided ───────────────────────────────────
    agent_cfg.max_iterations = args_cli.max_iterations

    # ── Build environment ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Task:        {args_cli.task}")
    print(f"  Num envs:    {args_cli.num_envs}")
    print(f"  Seed:        {args_cli.seed}")
    print(f"  Max iter:    {args_cli.max_iterations}")
    print(f"  Device:      {agent_cfg.device}")
    print(f"{'='*60}\n")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=agent_cfg.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"\n✅ Environment ready")
    print(f"   Observations : {env.num_obs}")
    print(f"   Actions      : {env.num_actions}")
    print(f"   Environments : {args_cli.num_envs}\n")

    # ── Log directory ─────────────────────────────────────────────────────────
    if args_cli.log_dir:
        log_dir = args_cli.log_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir   = os.path.join(
            os.path.expanduser("~/IsaacLab/logs/rsl_rl"),
            agent_cfg.experiment_name,
            timestamp,
        )

    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Logging to: {log_dir}")
    print(f"   Monitor:  tensorboard --logdir {os.path.dirname(log_dir)}\n")

    # ── Build runner ──────────────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # ── Resume from checkpoint if requested ───────────────────────────────────
    if args_cli.resume:
        if args_cli.checkpoint:
            ckpt_path = args_cli.checkpoint
        else:
            ckpt_path = get_checkpoint_path(
                os.path.dirname(log_dir),
                run_dir=".*",
                checkpoint="model_.*.pt",
            )
        print(f"▶  Resuming from: {ckpt_path}\n")
        runner.load(ckpt_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("🚀 Training started!\n")
    print(f"   Stop early when you see:")
    print(f"   jump_height        ≥ 0.4")
    print(f"   upright_after_land ≥ 0.4")
    print(f"   phase_backflip     5–15")
    print(f"   episode_length     ≥ 80\n")

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    # ── Export policy ─────────────────────────────────────────────────────────
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    print(f"\n📦 Exporting policy to: {export_dir}")

    export_policy_as_jit(
        runner.alg.actor_critic,
        runner.obs_normalizer,
        path=export_dir,
        filename="policy.pt",
    )
    print("   ✅ policy.pt  (TorchScript — use for sim2sim and real robot)")

    export_policy_as_onnx(
        runner.alg.actor_critic,
        normalizer=runner.obs_normalizer,
        path=os.path.join(export_dir, "policy.onnx"),
    )
    print("   ✅ policy.onnx (ONNX — use for embedded hardware)")

    print(f"\n✅ Training complete!")
    print(f"   Log dir    : {log_dir}")
    print(f"   Policy     : {export_dir}/policy.pt")
    print(f"   TensorBoard: tensorboard --logdir {os.path.dirname(log_dir)}")
    print(f"\nNext step — Sim-to-Sim transfer:")
    print(f"   python scripts/sim2sim_mujoco.py")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()