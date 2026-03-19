"""
28 DOF humanoid robot config for backflip task.
Uses humanoid_pkg/mjcf/humanoid_28dof.xml — same as Isaac Lab training.
"""
import mujoco
from pathlib import Path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

MJCF_28DOF = Path("/home/rishabh/humanoid_pkg/mjcf/humanoid_28dof.xml")
MESH_DIR   = Path("/home/rishabh/humanoid_pkg/meshes")

def get_spec_28dof() -> mujoco.MjSpec:
    """Load 28 DOF humanoid MjSpec — no actuators, mjlab adds its own."""
    return mujoco.MjSpec.from_file(str(MJCF_28DOF))


def get_28dof_robot_cfg() -> EntityCfg:
    """Create EntityCfg for 28 DOF humanoid — backflip task."""
    return EntityCfg(
        spec_fn=get_spec_28dof,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # ── Legs ──────────────────────────────────────────────────
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_hip_pitch_joint",),
                    stiffness=150.0,
                    damping=5.0,
                    effort_limit=200.0,
                    armature=0.0652,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_hip_roll_joint",),
                    stiffness=100.0,
                    damping=4.0,
                    effort_limit=200.0,
                    armature=0.100,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_hip_yaw_joint",),
                    stiffness=80.0,
                    damping=3.0,
                    effort_limit=200.0,
                    armature=0.0343,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_knee_joint",),
                    stiffness=200.0,
                    damping=5.0,
                    effort_limit=200.0,
                    armature=0.0330,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_foot1_joint", ".*_foot2_joint"),
                    stiffness=40.0,
                    damping=2.0,
                    effort_limit=200.0,
                    armature=0.0236,
                ),
                # ── Torso ─────────────────────────────────────────────────
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("torso_joint",),
                    stiffness=100.0,
                    damping=4.0,
                    effort_limit=80.0,
                    armature=0.0330,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("waist_roll_joint", "waist_pitch_joint"),
                    stiffness=100.0,
                    damping=4.0,
                    effort_limit=80.0,
                    armature=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("waist_rod_joint",),
                    stiffness=100.0,
                    damping=4.0,
                    effort_limit=80.0,
                    armature=0.01,
                ),
                # ── Arms ──────────────────────────────────────────────────
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_shoulder_.*_joint",),
                    stiffness=40.0,
                    damping=2.0,
                    effort_limit=30.0,
                    armature=0.02,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_elbow_.*_joint",),
                    stiffness=25.0,
                    damping=1.5,
                    effort_limit=30.0,
                    armature=0.015,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*_wrist_yaw_joint",),
                    stiffness=10.0,
                    damping=0.5,
                    effort_limit=30.0,
                    armature=0.01,
                ),
                # ── Head ──────────────────────────────────────────────────
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("neck_yaw_joint", "head_joint"),
                    stiffness=20.0,
                    damping=1.0,
                    effort_limit=5.0,
                    armature=0.01,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos={
                "left_hip_pitch_joint":         -0.20,
                "right_hip_pitch_joint":        -0.20,
                "left_knee_joint":               0.45,
                "right_knee_joint":              0.45,
                "left_foot1_joint":             -0.20,
                "right_foot1_joint":            -0.20,
                "left_shoulder_roll_joint":      0.3,
                "right_shoulder_roll_joint":    -0.3,
                "left_elbow_forearm_yaw_joint":  0.5,
                "right_elbow_forearm_yaw_joint": 0.5,
            },
        ),
    )
