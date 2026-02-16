"""Convert openhe/g1-retargeted-motions PKL files to CSV for csv_to_npz.py.

The PKL dataset has 23 DOF (no wrist joints). This script maps them to the
29-joint ordering expected by csv_to_npz.py, inserting zeros for the 6 missing
wrist joints.

Usage:
    uv run python -m mjlab.scripts.pkl_to_csv input.pkl output.csv
    uv run python -m mjlab.scripts.pkl_to_csv input.pkl output.csv --list-keys
"""

from pathlib import Path

import joblib
import numpy as np
import tyro

# 23-DOF ordering in the openhe PKL files (no wrist joints).
DOF_23_NAMES = [
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
]

# 29-joint ordering expected by csv_to_npz.py.
DOF_29_NAMES = [
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "left_wrist_roll",  # missing in PKL → 0
  "left_wrist_pitch",  # missing in PKL → 0
  "left_wrist_yaw",  # missing in PKL → 0
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
  "right_wrist_roll",  # missing in PKL → 0
  "right_wrist_pitch",  # missing in PKL → 0
  "right_wrist_yaw",  # missing in PKL → 0
]

# Build mapping: for each of the 29 joints, index into 23-DOF array or -1.
_DOF23_INDEX = {name: i for i, name in enumerate(DOF_23_NAMES)}
DOF_29_TO_23_MAP = [_DOF23_INDEX.get(name, -1) for name in DOF_29_NAMES]


def map_23_to_29(dof_23: np.ndarray) -> np.ndarray:
  """Map (T, 23) joint array to (T, 29), filling missing joints with 0."""
  T = dof_23.shape[0]
  dof_29 = np.zeros((T, 29), dtype=dof_23.dtype)
  for i_29, i_23 in enumerate(DOF_29_TO_23_MAP):
    if i_23 >= 0:
      dof_29[:, i_29] = dof_23[:, i_23]
  return dof_29


def main(
  input_file: str,
  output_file: str,
  motion_key: str | None = None,
  list_keys: bool = False,
):
  """Convert a PKL motion file to CSV format for csv_to_npz.py.

  Args:
      input_file: Path to the PKL file from openhe/g1-retargeted-motions.
      output_file: Path for the output CSV file.
      motion_key: Key inside the PKL dict. If None, uses the first key.
      list_keys: If True, just print available keys and exit.
  """
  data = joblib.load(input_file)

  if list_keys:
    print("Available keys in PKL file:")
    for key in data.keys():
      motion = data[key]
      n_frames = motion["dof"].shape[0]
      fps = motion.get("fps", "?")
      print(f"  '{key}' — {n_frames} frames, {fps} fps")
    return

  if motion_key is None:
    motion_key = next(iter(data.keys()))
    print(f"Using motion key: '{motion_key}'")

  motion = data[motion_key]
  root_pos = motion["root_trans_offset"]  # (T, 3)
  root_rot = motion["root_rot"]  # (T, 4) — assumed wxyz
  dof = motion["dof"]  # (T, 23)
  fps = motion.get("fps", 30)

  print(f"Motion: {root_pos.shape[0]} frames at {fps} FPS")
  print(f"  root_pos: {root_pos.shape}, root_rot: {root_rot.shape}, dof: {dof.shape}")

  # csv_to_npz.py expects quaternion in xyzw order and converts to wxyz
  # internally. The PKL already stores xyzw, so pass through as-is.
  root_rot_xyzw = root_rot

  # Map 23 DOF → 29 joints.
  dof_29 = map_23_to_29(dof)

  # Assemble CSV: [base_pos(3), base_rot_xyzw(4), joints(29)] = 36 columns.
  csv_data = np.concatenate([root_pos, root_rot_xyzw, dof_29], axis=1)

  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_path, csv_data, delimiter=",", fmt="%.8f")

  print(
    f"Saved CSV: {output_path} ({csv_data.shape[0]} frames, {csv_data.shape[1]} columns)"
  )
  print(f"\nNext step — convert to NPZ:")
  print(
    f"  uv run python -m mjlab.scripts.csv_to_npz"
    f" --input-file {output_path}"
    f" --input-fps {fps}"
    f" --output-name my_motion"
  )


if __name__ == "__main__":
  import mjlab

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
