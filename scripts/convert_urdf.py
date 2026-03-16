"""
scripts/convert_urdf.py
=======================
Convert humanoid_28dof.urdf → USD. Run once before training.

Usage:
    python scripts/convert_urdf.py \
        --urdf assets/robots/humanoid_28dof.urdf \
        --output assets/robots/humanoid_28dof.usd
"""

import argparse
import os
import sys

sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf",   type=str, required=True)
parser.add_argument("--output", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    urdf_path   = os.path.abspath(args_cli.urdf)
    output_path = os.path.abspath(args_cli.output)

    print(f"\n🔧 Converting: {urdf_path}")
    print(f"   Output    : {output_path}\n")

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from omni.isaac.urdf import _urdf
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints    = True
    cfg.convex_decomp         = False
    cfg.import_inertia_tensor = True
    cfg.fix_base              = False
    cfg.make_default_prim     = True
    cfg.create_physics_scene  = True
    cfg.distance_scale        = 1.0

    result, _ = _urdf.import_robot(urdf_path, cfg)

    if result:
        import omni.usd
        omni.usd.get_context().save_as_stage(output_path)
        print(f"✅ Saved: {output_path}")
        print("\n⚠️  Check Isaac Sim Stage panel:")
        print("   - right_hip_roll_ joint name after conversion")
        print("   - Robot spawns upright at ~1.05m")
    else:
        print("❌ Conversion failed — check URDF mesh paths")


if __name__ == "__main__":
    main()
    simulation_app.close()