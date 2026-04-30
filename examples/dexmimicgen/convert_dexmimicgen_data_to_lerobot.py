#!/usr/bin/env python3

"""
Convert a DexMimicGen HDF5 dataset to the LeRobot format expected by openpi training.

Example usage:
uv run examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py \
    --dataset-path /home/exx/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5
"""

import dataclasses
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from openpi.policies import dexmimicgen_policy

DEFAULT_REPO_ID = "local/dexmimicgen_two_arm_threading"


def _gripper_names(prefix: str, dim: int, suffix: str) -> list[str]:
    if dim == 1:
        return [f"{prefix}_gripper_{suffix}"]
    return [f"{prefix}_gripper_{suffix}_{i}" for i in range(dim)]


def _state_names(right_gripper_dim: int, left_gripper_dim: int) -> list[str]:
    return [
        "right_eef_pos_x",
        "right_eef_pos_y",
        "right_eef_pos_z",
        "right_eef_rot6d_0",
        "right_eef_rot6d_1",
        "right_eef_rot6d_2",
        "right_eef_rot6d_3",
        "right_eef_rot6d_4",
        "right_eef_rot6d_5",
        *_gripper_names("right", right_gripper_dim, "position"),
        "left_eef_pos_x",
        "left_eef_pos_y",
        "left_eef_pos_z",
        "left_eef_rot6d_0",
        "left_eef_rot6d_1",
        "left_eef_rot6d_2",
        "left_eef_rot6d_3",
        "left_eef_rot6d_4",
        "left_eef_rot6d_5",
        *_gripper_names("left", left_gripper_dim, "position"),
    ]


def _action_names(right_gripper_dim: int, left_gripper_dim: int) -> list[str]:
    return [
        "right_rel_pos_x",
        "right_rel_pos_y",
        "right_rel_pos_z",
        "right_rel_rot6d_0",
        "right_rel_rot6d_1",
        "right_rel_rot6d_2",
        "right_rel_rot6d_3",
        "right_rel_rot6d_4",
        "right_rel_rot6d_5",
        *_gripper_names("right", right_gripper_dim, "action"),
        "left_rel_pos_x",
        "left_rel_pos_y",
        "left_rel_pos_z",
        "left_rel_rot6d_0",
        "left_rel_rot6d_1",
        "left_rel_rot6d_2",
        "left_rel_rot6d_3",
        "left_rel_rot6d_4",
        "left_rel_rot6d_5",
        *_gripper_names("left", left_gripper_dim, "action"),
    ]


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_path: Path
    repo_id: str = DEFAULT_REPO_ID
    default_prompt: str = dexmimicgen_policy.DEFAULT_PROMPT
    push_to_hub: bool = False
    max_episodes: int | None = None
    right_gripper_state_dim: int | None = None
    left_gripper_state_dim: int | None = None
    right_gripper_action_dim: int | None = None
    left_gripper_action_dim: int | None = None


def _episode_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _standardize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError(f"Expected non-zero quaternion, got {quaternion}")
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def _quat_xyzw_to_rotmat(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _standardize_quaternion_xyzw(quaternion)
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _rotmat_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    # Match DexMimicGen's action rot6d convention (first two rows of R, = first two cols of R.T)
    # so state and action parameterize rotation identically.
    return np.asarray(rotation_matrix[:2, :], dtype=np.float32).reshape(-1)


def _quat_xyzw_to_rot6d(quaternion: np.ndarray) -> np.ndarray:
    return _rotmat_to_rot6d(_quat_xyzw_to_rotmat(quaternion))


def _create_dataset(
    repo_id: str,
    *,
    state_names: list[str],
    action_names: list[str],
    fps: int = 20,
    image_writer_threads: int = 8,
    image_writer_processes: int = 4,
) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda_bimanual",
        fps=fps,
        features={
            "observation.images.top": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": [state_names],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": [action_names],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def _dataset_width(group: h5py.Group, key: str) -> int:
    shape = group[key].shape
    if len(shape) < 2:
        return 1
    return int(shape[-1])


def _resolve_dims(args: Args, obs_group: h5py.Group, action_group: h5py.Group) -> tuple[int, int, int, int]:
    right_action_dim = args.right_gripper_action_dim or _dataset_width(action_group, "right_gripper")
    left_action_dim = args.left_gripper_action_dim or _dataset_width(action_group, "left_gripper")
    right_state_dim = args.right_gripper_state_dim or min(_dataset_width(obs_group, "robot0_gripper_qpos"), right_action_dim)
    left_state_dim = args.left_gripper_state_dim or min(_dataset_width(obs_group, "robot1_gripper_qpos"), left_action_dim)
    return right_state_dim, left_state_dim, right_action_dim, left_action_dim


def _gripper_qpos(obs_group: h5py.Group, key: str, index: int, dim: int) -> np.ndarray:
    qpos = np.asarray(obs_group[key][index], dtype=np.float32).reshape(-1)
    if qpos.shape[0] < dim:
        raise ValueError(f"Expected {key} to have at least {dim} dims, got {qpos.shape[0]}")
    return qpos[:dim]


def _state_vector(obs_group: h5py.Group, index: int, right_gripper_dim: int, left_gripper_dim: int) -> np.ndarray:
    # Keep the state layout aligned with the action ordering: right arm first, then left arm.
    return np.concatenate(
        [
            np.asarray(obs_group["robot0_eef_pos"][index], dtype=np.float32),
            _quat_xyzw_to_rot6d(obs_group["robot0_eef_quat"][index]),
            _gripper_qpos(obs_group, "robot0_gripper_qpos", index, right_gripper_dim),
            np.asarray(obs_group["robot1_eef_pos"][index], dtype=np.float32),
            _quat_xyzw_to_rot6d(obs_group["robot1_eef_quat"][index]),
            _gripper_qpos(obs_group, "robot1_gripper_qpos", index, left_gripper_dim),
        ],
        dtype=np.float32,
    )


def _action_vector(
    action_group: h5py.Group,
    index: int,
    right_gripper_dim: int | None = None,
    left_gripper_dim: int | None = None,
) -> np.ndarray:
    right_gripper = np.asarray(action_group["right_gripper"][index], dtype=np.float32).reshape(-1)
    left_gripper = np.asarray(action_group["left_gripper"][index], dtype=np.float32).reshape(-1)
    if right_gripper_dim is not None:
        right_gripper = right_gripper[:right_gripper_dim]
    if left_gripper_dim is not None:
        left_gripper = left_gripper[:left_gripper_dim]
    return np.concatenate(
        [
            np.asarray(action_group["right_rel_pos"][index], dtype=np.float32),
            np.asarray(action_group["right_rel_rot_6d"][index], dtype=np.float32),
            right_gripper,
            np.asarray(action_group["left_rel_pos"][index], dtype=np.float32),
            np.asarray(action_group["left_rel_rot_6d"][index], dtype=np.float32),
            left_gripper,
        ],
        dtype=np.float32,
    )


def main(args: Args) -> None:
    with h5py.File(args.dataset_path, "r") as f:
        data_group = f["data"]
        episode_keys = sorted(data_group.keys(), key=_episode_sort_key)
        if args.max_episodes is not None:
            episode_keys = episode_keys[: args.max_episodes]
        if not episode_keys:
            raise ValueError(f"No episodes found in {args.dataset_path}")

        first_episode = data_group[episode_keys[0]]
        right_state_dim, left_state_dim, right_action_dim, left_action_dim = _resolve_dims(
            args, first_episode["obs"], first_episode["action_dict"]
        )
        state_names = _state_names(right_state_dim, left_state_dim)
        action_names = _action_names(right_action_dim, left_action_dim)
        dataset = _create_dataset(args.repo_id, state_names=state_names, action_names=action_names)

        print(f"Found {len(episode_keys)} episodes for conversion")
        print(
            "Using gripper dims: "
            f"state right={right_state_dim}, state left={left_state_dim}, "
            f"action right={right_action_dim}, action left={left_action_dim}"
        )

        for episode_key in tqdm.tqdm(episode_keys, desc="Converting DexMimicGen episodes"):
            episode = data_group[episode_key]
            obs = episode["obs"]
            actions = episode["action_dict"]
            num_frames = int(episode.attrs["num_samples"])

            for i in range(num_frames):
                action = _action_vector(actions, i, right_action_dim, left_action_dim)
                dataset.add_frame(
                    {
                        # robosuite/mujoco camera observations are returned vertically flipped;
                        # flip to upright so SigLIP sees natural-orientation images.
                        "observation.images.top": np.ascontiguousarray(obs["agentview_image"][i][::-1], dtype=np.uint8),
                        "observation.images.left_wrist": np.ascontiguousarray(
                            obs["robot1_eye_in_hand_image"][i][::-1], dtype=np.uint8
                        ),
                        "observation.images.right_wrist": np.ascontiguousarray(
                            obs["robot0_eye_in_hand_image"][i][::-1], dtype=np.uint8
                        ),
                        "observation.state": _state_vector(obs, i, right_state_dim, left_state_dim),
                        "action": action,
                        "task": args.default_prompt,
                    }
                )
            dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["dexmimicgen", "bimanual", "panda", "lerobot"],
            private=False,
            push_videos=False,
            license="cc-by-4.0",
        )

    print(f"Wrote dataset to {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
