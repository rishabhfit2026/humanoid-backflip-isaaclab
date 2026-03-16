import mujoco
import os

BASE_DIR = "/home/ahmed/E_Disk/worksapce/humanoid_ws/src/humanoid_pkg/meshes"


os.chdir(BASE_DIR)

urdf_path = os.path.join(BASE_DIR, "humanoid_pkg.urdf")

model = mujoco.MjModel.from_xml_path(urdf_path)
mujoco.mj_saveLastXML("humanoid_pkg.xml", model)

print(" MJCF generated successfully")
