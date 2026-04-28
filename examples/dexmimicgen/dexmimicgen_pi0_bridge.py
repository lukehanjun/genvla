"""Shared NumPy helpers for pi0 ↔ DexMimicGen (no robosuite).

Used by ``rollout_sim.py`` (simulator env) and ``visualize_actions.py`` (openpi venv only).
"""

from __future__ import annotations

import numpy as np

ACTION_COMPONENTS = {
    "right": {
        "position": slice(0, 3),
        "rotation": slice(3, 9),
        "gripper": slice(9, 10),
    },
    "left": {
        "position": slice(10, 13),
        "rotation": slice(13, 19),
        "gripper": slice(19, 20),
    },
}


def action_20d_from_action_dict(action_group, index: int) -> np.ndarray:
    """Build the canonical 20D pi0 action vector from a DexMimicGen ``action_dict`` group.

    Layout:
    [r_pos(3), r_rot6d(6), r_grip(1), l_pos(3), l_rot6d(6), l_grip(1)]
    """
    return np.concatenate(
        [
            np.asarray(action_group["right_rel_pos"][index], dtype=np.float32),
            np.asarray(action_group["right_rel_rot_6d"][index], dtype=np.float32),
            np.asarray(action_group["right_gripper"][index], dtype=np.float32),
            np.asarray(action_group["left_rel_pos"][index], dtype=np.float32),
            np.asarray(action_group["left_rel_rot_6d"][index], dtype=np.float32),
            np.asarray(action_group["left_gripper"][index], dtype=np.float32),
        ],
        dtype=np.float32,
    )


def quat_xyzw_to_rotmat(q):
    """Quaternion (x, y, z, w) to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    q = q / n
    if q[3] < 0:
        q = -q
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_rot6d(rot_mat):
    """3x3 rotation matrix -> 6D continuous rotation.

    Uses DexMimicGen's row-major convention (first two rows of R = first two cols of R.T)
    so the state rot6d matches the action rot6d stored in the dataset.
    """
    return np.asarray(rot_mat[:2, :], dtype=np.float32).reshape(-1)


def quat_xyzw_to_rot6d(q):
    return rotmat_to_rot6d(quat_xyzw_to_rotmat(q))


def rot6d_to_rotmat(r6):
    """6D continuous rotation -> 3x3 rotation matrix (Gram-Schmidt)."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[:3], r6[3:6]
    n1 = np.linalg.norm(a1)
    a1 = np.array([1, 0, 0], dtype=np.float64) if n1 < 1e-8 else a1 / n1
    a2 = a2 - np.dot(a1, a2) * a1
    n2 = np.linalg.norm(a2)
    a2 = np.array([0, 1, 0], dtype=np.float64) if n2 < 1e-8 else a2 / n2
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=1)


def rotmat_to_axis_angle(rot_mat):
    """3x3 rotation matrix -> axis-angle (3D vector, angle = ||v||)."""
    rot_mat = np.asarray(rot_mat, dtype=np.float64)
    angle = np.arccos(np.clip((np.trace(rot_mat) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def rot6d_to_axis_angle(r6):
    """6D rotation -> axis-angle (DexMimicGen rot6d uses transposed convention)."""
    return rotmat_to_axis_angle(rot6d_to_rotmat(r6).T)


def convert_action_20d_to_14d(action_20d):
    """20D pi0 action -> 14D robosuite OSC_POSE action.

    Input:  [r_pos(3), r_rot6d(6), r_grip(1), l_pos(3), l_rot6d(6), l_grip(1)]
    Output: [r_pos(3), r_aa(3), r_grip(1), l_pos(3), l_aa(3), l_grip(1)]
    """
    r_pos = action_20d[0:3]
    r_rot6d = action_20d[3:9]
    r_grip = action_20d[9:10]
    l_pos = action_20d[10:13]
    l_rot6d = action_20d[13:19]
    l_grip = action_20d[19:20]
    r_aa = rot6d_to_axis_angle(r_rot6d)
    l_aa = rot6d_to_axis_angle(l_rot6d)
    return np.concatenate([r_pos, r_aa, r_grip, l_pos, l_aa, l_grip]).astype(np.float64)


def build_observation(obs, prompt="thread the needle with both arms"):
    """Build the observation dict expected by the pi0 DexMimicGen policy server.

    State layout matches ``convert_dexmimicgen_data_to_lerobot.py``:
    [r_eef_pos(3), r_rot6d(6), r_grip(1), l_eef_pos(3), l_rot6d(6), l_grip(1)]
    """
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_xyzw_to_rot6d(obs["robot0_eef_quat"]),
            np.asarray([obs["robot0_gripper_qpos"][0]], dtype=np.float32),
            np.asarray(obs["robot1_eef_pos"], dtype=np.float32),
            quat_xyzw_to_rot6d(obs["robot1_eef_quat"]),
            np.asarray([obs["robot1_gripper_qpos"][0]], dtype=np.float32),
        ]
    )
    # robosuite/mujoco camera observations are returned vertically flipped;
    # flip to upright so the policy sees the same orientation as the training data.
    images = {
        "top": np.ascontiguousarray(obs["agentview_image"][::-1], dtype=np.uint8),
        "right_wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1], dtype=np.uint8),
        "left_wrist": np.ascontiguousarray(obs["robot1_eye_in_hand_image"][::-1], dtype=np.uint8),
    }
    return {
        "state": state,
        "images": images,
        "prompt": prompt,
    }
