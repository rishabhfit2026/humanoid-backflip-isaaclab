"""28-DOF humanoid robot config using humanoid_pkg MJCF."""
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg

HUMANOID_28DOF_XML = "/home/rishabh/humanoid_pkg/mjcf/humanoid_28dof.xml"

def get_28dof_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(HUMANOID_28DOF_XML)

def get_28dof_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_28dof_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*hip_pitch.*",),
                    stiffness=150.0, damping=5.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*hip_roll.*",),
                    stiffness=100.0, damping=4.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*hip_yaw.*",),
                    stiffness=80.0, damping=3.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*knee.*",),
                    stiffness=200.0, damping=5.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*foot.*",),
                    stiffness=40.0, damping=2.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("torso_joint", "waist_roll_joint",
                                      "waist_pitch_joint", "waist_rod_joint"),
                    stiffness=100.0, damping=4.0, effort_limit=80.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*shoulder.*",),
                    stiffness=40.0, damping=2.0, effort_limit=30.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*elbow.*",),
                    stiffness=25.0, damping=1.5, effort_limit=30.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=(".*wrist.*",),
                    stiffness=10.0, damping=0.5, effort_limit=30.0,
                ),
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("neck_yaw_joint", "head_joint"),
                    stiffness=20.0, damping=1.0, effort_limit=5.0,
                ),
            ),
        ),
    )
