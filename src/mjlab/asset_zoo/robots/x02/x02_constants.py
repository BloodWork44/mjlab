"""BitBots x02 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# URDF and assets.
##

X02_URDF: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "x02" / "xmls" / "x02.urdf"
assert X02_URDF.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X02_URDF.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X02_URDF))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# PD gains and effort limits taken from HoST legged_gym x02 config.
# Armature set to 0.01 (HoST default for x02).
##

ARMATURE = 0.01

# Stiffness (Kp) and damping (Kd) per joint group, from HoST.
X02_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=160.0,
  damping=4.0,
  effort_limit=50.0,
  armature=ARMATURE,
)
X02_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=50.0,
  armature=ARMATURE,
)
X02_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=72.0,
  armature=ARMATURE,
)
X02_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_pitch_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=60.0,
  armature=ARMATURE,
)
X02_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=30.0,
  armature=ARMATURE,
)
X02_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_joint",),
  stiffness=100.0,
  damping=4.0,
  effort_limit=50.0,
  armature=ARMATURE,
)
X02_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=100.0,
  damping=4.0,
  effort_limit=24.0,
  armature=ARMATURE,
)
X02_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint",),
  stiffness=100.0,
  damping=4.0,
  effort_limit=24.0,
  armature=ARMATURE,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.87),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_pitch_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 0.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim=3,
  friction=(0.6,),
)

##
# Final config.
##

X02_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X02_ACTUATOR_HIP_YAW,
    X02_ACTUATOR_HIP_ROLL,
    X02_ACTUATOR_HIP_PITCH,
    X02_ACTUATOR_KNEE,
    X02_ACTUATOR_ANKLE,
    X02_ACTUATOR_TORSO,
    X02_ACTUATOR_SHOULDER,
    X02_ACTUATOR_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_x02_robot_cfg() -> EntityCfg:
  """Get a fresh x02 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X02_ARTICULATION,
  )


X02_ACTION_SCALE: dict[str, float] = {}
for a in X02_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    X02_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x02_robot_cfg())

  viewer.launch(robot.spec.compile())
