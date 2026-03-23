"""Keyboard controlled velocity policy — random grid terrain — native viewer."""
from dataclasses import asdict
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE
import mjlab.tasks

TASK_ID = "Mjlab-Velocity-RandomGrid-Humanoid"
DEVICE  = "cuda:0"
SPEED   = 1.0
TURN    = 0.6

vx, vy, wz = 0.0, 0.0, 0.0

def key_callback(key):
    global vx, vy, wz
    if key == KEY_W:
        vx, vy, wz = SPEED, 0.0, 0.0
        print(f"\r↑ Forward  vx={vx:+.1f}   ", end='', flush=True)
    elif key == KEY_S:
        vx, vy, wz = -SPEED, 0.0, 0.0
        print(f"\r↓ Backward vx={vx:+.1f}   ", end='', flush=True)
    elif key == KEY_A:
        vx, vy, wz = 0.0, 0.0, TURN
        print(f"\r← Turn Left  wz={wz:+.1f} ", end='', flush=True)
    elif key == KEY_D:
        vx, vy, wz = 0.0, 0.0, -TURN
        print(f"\r→ Turn Right wz={wz:+.1f} ", end='', flush=True)
    elif key == KEY_SPACE:
        vx, vy, wz = 0.0, 0.0, 0.0
        print(f"\r⬛ STOP                   ", end='', flush=True)

print("Loading environment...")
env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
env = RslRlVecEnvWrapper(env)
print("✅ Environment loaded")

print("Loading policy...")
agent_cfg = load_rl_cfg(TASK_ID)
runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner

import os
CKPT = sorted([f for f in os.listdir("logs/rsl_rl/humanoid_velocity/2026-03-21_18-37-49/") if f.endswith(".pt")])[-1]
CKPT = f"logs/rsl_rl/humanoid_velocity/2026-03-21_18-37-49/{CKPT}"
print(f"Using: {CKPT}")

runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(CKPT, map_location=DEVICE)
base_policy = runner.get_inference_policy(device=DEVICE)
print("✅ Policy loaded")

class KeyboardPolicy:
    def __call__(self, obs):
        try:
            twist = env.unwrapped.command_manager._terms["twist"]
            twist.command[:, 0] = vx
            twist.command[:, 1] = vy
            twist.command[:, 2] = wz
        except Exception:
            pass
        return base_policy(obs)

print("\n========================================")
print("🎮 KEYBOARD CONTROL — RANDOM GRID")
print("  W = Forward    S = Backward")
print("  A = Turn Left  D = Turn Right")
print("  SPACE = Stop   ESC = Quit")
print("========================================\n")

NativeMujocoViewer(
    env=env,
    policy=KeyboardPolicy(),
    key_callback=key_callback
).run()
env.close()
