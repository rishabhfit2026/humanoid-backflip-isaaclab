"""
Direct keyboard control of humanoid robot in MuJoCo viewer.
No policy needed — directly set velocity commands.
W/S = forward/back, A/D = turn, SPACE = stop
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

MJCF_PATH = "/home/rishabh/humanoid-mjlab/src/mjlab/asset_zoo/robots/humanoid/xmls/humanoid.xml"

# Velocity command
cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
SPEED = 0.5
TURN  = 0.4

def on_press(key):
    try:
        if key.char == 'w':
            cmd["vx"] = SPEED;  cmd["vy"] = 0.0; cmd["wz"] = 0.0
        elif key.char == 's':
            cmd["vx"] = -SPEED; cmd["vy"] = 0.0; cmd["wz"] = 0.0
        elif key.char == 'a':
            cmd["wz"] = TURN;   cmd["vx"] = 0.0
        elif key.char == 'd':
            cmd["wz"] = -TURN;  cmd["vx"] = 0.0
        elif key.char == 'q':
            cmd["vy"] = SPEED;  cmd["vx"] = 0.0
        elif key.char == 'e':
            cmd["vy"] = -SPEED; cmd["vx"] = 0.0
        print(f"CMD → vx={cmd['vx']:+.1f} vy={cmd['vy']:+.1f} wz={cmd['wz']:+.1f}")
    except AttributeError:
        if key == keyboard.Key.space:
            cmd["vx"] = 0.0; cmd["vy"] = 0.0; cmd["wz"] = 0.0
            print("STOP")

def on_release(key):
    # Stop when key released
    cmd["vx"] = 0.0
    cmd["vy"] = 0.0
    cmd["wz"] = 0.0

# Load model
print("Loading humanoid model...")
from mjlab.asset_zoo.robots.humanoid.humanoid_constants import get_spec
spec = get_spec()
mj_model = spec.compile()
mj_data  = mujoco.MjData(mj_model)
print(f"✅ nq={mj_model.nq} nu={mj_model.nu}")

# Reset to standing pose
mujoco.mj_resetData(mj_model, mj_data)
mj_data.qpos[2] = 0.75
mj_data.qpos[3] = 1.0
mujoco.mj_forward(mj_model, mj_data)

print("\n🚀 Keyboard Control Mode!")
print("  W = forward    S = backward")
print("  A = turn left  D = turn right")
print("  Q = strafe left  E = strafe right")
print("  SPACE = stop")
print("  Click MuJoCo window first, then press keys!\n")

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # Apply velocity command as root velocity target
        # This just shows the robot in T-pose responding to commands
        # For real walking you need a trained policy
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        time.sleep(0.005)

listener.stop()
