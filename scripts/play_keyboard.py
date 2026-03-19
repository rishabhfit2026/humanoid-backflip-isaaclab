"""Keyboard controlled velocity policy — native MuJoCo viewer."""
import threading
from dataclasses import asdict
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE
import mjlab.tasks  # noqa

TASK_ID = "Mjlab-Velocity-Flat-Humanoid"
CKPT    = "logs/rsl_rl/humanoid_velocity/2026-03-19_16-22-00/model_1000.pt"
DEVICE  = "cuda:0"
SPEED   = 0.8
TURN    = 0.5

cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

def key_callback(key: int) -> None:
    if key == KEY_W:
        cmd["vx"] = SPEED;  cmd["vy"] = 0.0; cmd["wz"] = 0.0
        print(f"\r↑ Forward  vx={cmd['vx']:+.1f}   ", end='', flush=True)
    elif key == KEY_S:
        cmd["vx"] = -SPEED; cmd["vy"] = 0.0; cmd["wz"] = 0.0
        print(f"\r↓ Backward vx={cmd['vx']:+.1f}   ", end='', flush=True)
    elif key == KEY_A:
        cmd["wz"] = TURN;   cmd["vx"] = 0.0; cmd["vy"] = 0.0
        print(f"\r← Turn Left  wz={cmd['wz']:+.1f} ", end='', flush=True)
    elif key == KEY_D:
        cmd["wz"] = -TURN;  cmd["vx"] = 0.0; cmd["vy"] = 0.0
        print(f"\r→ Turn Right wz={cmd['wz']:+.1f} ", end='', flush=True)
    elif key == KEY_SPACE:
        cmd["vx"] = 0.0; cmd["vy"] = 0.0; cmd["wz"] = 0.0
        print(f"\r⬛ STOP                        ", end='', flush=True)

print("Loading environment...")
env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
env = RslRlVecEnvWrapper(env)
print("✅ Environment loaded")

print("Loading policy...")
agent_cfg = load_rl_cfg(TASK_ID)
runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(CKPT, map_location=DEVICE)
base_policy = runner.get_inference_policy(device=DEVICE)
print("✅ Policy loaded")

class KeyboardPolicy:
    def __call__(self, obs):
        try:
            twist = env.unwrapped.command_manager._terms["twist"]
            twist.command[:, 0] = cmd["vx"]
            twist.command[:, 1] = cmd["vy"]
            twist.command[:, 2] = cmd["wz"]
        except Exception:
            pass
        return base_policy(obs)

print("\n========================================")
print("🎮 KEYBOARD CONTROL — NATIVE VIEWER")
print("  W = Forward    S = Backward")
print("  A = Turn Left  D = Turn Right")
print("  SPACE = Stop   ESC = Quit")
print("========================================\n")

NativeMujocoViewer(
    env=env,
    policy=KeyboardPolicy(),
    key_callback=key_callback,
).run()
env.close()
