"""
Play velocity policy with WASD keyboard control in native MuJoCo viewer.
Controls:
    W = forward      S = backward
    A = turn left    D = turn right
    Q = strafe left  E = strafe right
    Space = stop     R = reset
    ESC = quit
"""
import torch
import tyro
from dataclasses import dataclass
from pathlib import Path
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
    KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E,
    KEY_SPACE, KEY_R
)

TASK_ID = "Mjlab-Velocity-Flat-Humanoid"
SPEED   = 0.5   # m/s
TURN    = 0.4   # rad/s

# Shared velocity command (modified by keyboard)
cmd = [0.0, 0.0, 0.0]  # [vx, vy, wz]

def key_callback(key: int) -> None:
    global cmd
    if key == KEY_W:
        cmd[0] = SPEED      # forward
    elif key == KEY_S:
        cmd[0] = -SPEED     # backward
    elif key == KEY_Q:
        cmd[1] = SPEED      # strafe left
    elif key == KEY_E:
        cmd[1] = -SPEED     # strafe right
    elif key == KEY_A:
        cmd[2] = TURN       # turn left
    elif key == KEY_D:
        cmd[2] = -TURN      # turn right
    elif key == KEY_SPACE:
        cmd[0] = 0.0        # stop all
        cmd[1] = 0.0
        cmd[2] = 0.0
    print(f"CMD: vx={cmd[0]:+.1f}  vy={cmd[1]:+.1f}  wz={cmd[2]:+.1f}")


@dataclass
class Config:
    checkpoint_file: str


def main():
    cfg = tyro.cli(Config)

    print("Loading environment...")
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.num_envs = 1
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env)
    print("✅ Environment loaded")

    print("Loading policy...")
    agent_cfg = load_rl_cfg(TASK_ID)
    runner = MjlabOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device="cuda:0"
    )
    runner.load(cfg.checkpoint_file)
    policy = runner.get_inference_policy(device="cuda:0")
    print("✅ Policy loaded")

    # Override command with keyboard values
    class KeyboardPolicy:
        def __call__(self, obs):
            # Set velocity command in environment
            if hasattr(env.unwrapped, 'command_manager'):
                cm = env.unwrapped.command_manager
                if hasattr(cm, '_terms') and 'twist' in cm._terms:
                    term = cm._terms['twist']
                    term.command[:, 0] = cmd[0]  # vx
                    term.command[:, 1] = cmd[1]  # vy
                    term.command[:, 2] = cmd[2]  # wz
            return policy(obs)

    keyboard_policy = KeyboardPolicy()

    print("\n🚀 Launching MuJoCo viewer with keyboard control!")
    print("Controls:")
    print("  W/S = forward/backward")
    print("  A/D = turn left/right")
    print("  Q/E = strafe left/right")
    print("  SPACE = stop")
    print("  ENTER = reset")
    print("  ESC = quit\n")

    viewer = NativeMujocoViewer(
        env=env,
        policy=keyboard_policy,
        key_callback=key_callback,
    )
    viewer.run()


if __name__ == "__main__":
    main()
