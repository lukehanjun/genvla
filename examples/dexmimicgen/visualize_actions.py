#!/usr/bin/env python3
"""
Visualize DexMimicGen / pi0 actions as static figures and optional video.

Two modes:

1) **dataset** — Plot actions from a local LeRobot dataset (per-timestep 20-D vectors).

2) **policy** — Load a trained checkpoint, build one observation from an HDF5 demo frame,
   run :meth:`Policy.infer`, and plot the returned action chunk (50 × 20).

Examples::

    uv run examples/dexmimicgen/visualize_actions.py dataset \\
        --repo-id local/dexmimicgen_two_arm_threading --episode 0 \\
        --output-dir ./viz_out

    uv run examples/dexmimicgen/visualize_actions.py policy \\
        --train-config pi0_dexmimicgen_two_arm_threading \\
        --checkpoint checkpoints/pi0_dexmimicgen_two_arm_threading/exp/10000 \\
        --hdf5 /path/to/two_arm_threading.hdf5 --demo demo_0 --frame 0 \\
        --output-dir ./viz_out

Requires matplotlib (``uv pip install matplotlib`` or the openpi dev dependency group).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))
from dexmimicgen_pi0_bridge import build_observation


def _load_episode_actions(repo_id: str, episode_index: int) -> np.ndarray:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id)
    if episode_index < 0 or episode_index >= len(ds.episode_data_index["from"]):
        raise ValueError(
            f"episode_index {episode_index} out of range (num episodes: "
            f"{len(ds.episode_data_index['from'])})"
        )
    s = int(ds.episode_data_index["from"][episode_index])
    e = int(ds.episode_data_index["to"][episode_index])
    sub = ds.hf_dataset.select(range(s, e))
    return np.stack([np.asarray(x, dtype=np.float32) for x in sub["action"]], axis=0)


def _obs_dict_from_hdf5(hdf5_path: str, demo_key: str, frame: int) -> dict:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        og = f[f"data/{demo_key}/obs"]
        obs = {k: np.asarray(og[k][frame]) for k in og.keys()}
    return obs


def _plot_dataset_episode(actions: np.ndarray, out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(actions.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(actions.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xlabel("time step")
    ax.set_ylabel("action dim (20)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_actions_heatmap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    names = (
        [f"r_pos{i}" for i in range(3)]
        + [f"r_r6_{i}" for i in range(6)]
        + ["r_grip"]
        + [f"l_pos{i}" for i in range(3)]
        + [f"l_r6_{i}" for i in range(6)]
        + ["l_grip"]
    )
    for i in range(20):
        axes[i].plot(t, actions[:, i])
        axes[i].set_ylabel(names[i], rotation=0, labelpad=30, fontsize=8)
    axes[-1].set_xlabel("time step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_actions_per_dim.png", dpi=150)
    plt.close(fig)

    # 3D end-effector relative deltas (first 3 = right, next 10 skip to left at 10:13)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    r = actions[:, 0:3]
    l = actions[:, 10:13]
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label="right Δpos", color="C0")
    ax.plot(l[:, 0], l[:, 1], l[:, 2], label="left Δpos", color="C1")
    ax.set_title("Relative position deltas (dataset actions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_delta_pos_3d.png", dpi=150)
    plt.close(fig)


def _plot_policy_chunk(chunk: np.ndarray, out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if chunk.ndim != 2 or chunk.shape[1] != 20:
        raise ValueError(f"Expected actions shape (H, 20), got {chunk.shape}")

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(chunk.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xlabel("chunk step")
    ax.set_ylabel("action dim")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "policy_chunk_heatmap.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(chunk[:, 0], chunk[:, 1], chunk[:, 2], label="right Δpos", color="C0")
    ax.plot(chunk[:, 10], chunk[:, 11], chunk[:, 12], label="left Δpos", color="C1")
    ax.set_title("Policy chunk: relative position (first 3 / left arm 10:12)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "policy_chunk_delta_pos_3d.png", dpi=150)
    plt.close(fig)


def _write_mp4_dataset(actions: np.ndarray, path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    t_full = np.arange(actions.shape[0])
    window = 50
    for start in range(0, max(1, actions.shape[0] - window + 1), max(1, window // 5)):
        end = min(start + window, actions.shape[0])
        sl = slice(start, end)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(actions[sl].T, aspect="auto", interpolation="nearest", cmap="coolwarm")
        ax.set_title(f"actions (window {start}:{end})")
        ax.set_xlabel("time in window")
        ax.set_ylabel("dim")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        plt.close(fig)
    writer.close()


def cmd_dataset(args: argparse.Namespace) -> None:
    actions = _load_episode_actions(args.repo_id, args.episode)
    out = Path(args.output_dir)
    title = f"{args.repo_id} episode {args.episode} (T={actions.shape[0]})"
    _plot_dataset_episode(actions, out, title)
    if args.video:
        _write_mp4_dataset(actions, out / "episode_actions_sliding.mp4", fps=args.fps)
    print(f"Wrote figures to {out.resolve()}")


def cmd_policy(args: argparse.Namespace) -> None:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    obs_dict = _obs_dict_from_hdf5(args.hdf5, args.demo, args.frame)
    observation = build_observation(obs_dict, prompt=args.prompt)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.train_config),
        args.checkpoint,
        default_prompt=args.prompt,
    )
    result = policy.infer(observation)
    chunk = np.asarray(result["actions"], dtype=np.float32)
    out_dir = Path(args.output_dir)
    title = f"{args.train_config} @ {args.checkpoint} ({args.demo} frame {args.frame})"
    _plot_policy_chunk(chunk, out_dir, title)
    print(f"Wrote policy chunk visualizations to {out_dir.resolve()}")


def main() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required. Install with: uv pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Visualize DexMimicGen / pi0 actions")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_ds = sub.add_parser("dataset", help="Plot actions from a LeRobot dataset episode")
    p_ds.add_argument("--repo-id", type=str, default="local/dexmimicgen_two_arm_threading")
    p_ds.add_argument("--episode", type=int, default=0)
    p_ds.add_argument("--output-dir", type=str, default="./dexmimicgen_action_viz")
    p_ds.add_argument("--video", action="store_true", help="Write a sliding-window MP4")
    p_ds.add_argument("--fps", type=int, default=20)
    p_ds.set_defaults(func=cmd_dataset)

    p_po = sub.add_parser("policy", help="Plot an action chunk from a trained policy")
    p_po.add_argument("--train-config", type=str, required=True, help="TrainConfig name, e.g. pi0_dexmimicgen_two_arm_threading")
    p_po.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory with model.safetensors or params")
    p_po.add_argument("--hdf5", type=str, required=True, help="DexMimicGen HDF5 (for one demo frame)")
    p_po.add_argument("--demo", type=str, default="demo_0")
    p_po.add_argument("--frame", type=int, default=0)
    p_po.add_argument("--prompt", type=str, default="thread the needle with both arms")
    p_po.add_argument("--output-dir", type=str, default="./dexmimicgen_action_viz")
    p_po.set_defaults(func=cmd_policy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
