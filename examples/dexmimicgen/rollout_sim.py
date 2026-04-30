#!/usr/bin/env python3
"""
Run a fine-tuned pi0 policy in the DexMimicGen TwoArmThreading robosuite simulator
and record a video of the rollout.
This script acts as a lightweight client that communicates with the openpi policy
server over WebSockets. It runs inside the dexmimicgen conda environment (which has
robosuite/mujoco) while the policy server runs in the openpi venv (which has JAX/PyTorch).
Usage:
    # Terminal 1 -- start the policy server (openpi venv):
    cd /home/horowitz3/pi0/openpi
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi0_dexmimicgen_two_arm_threading \\
        --policy.dir=checkpoints/pi0_dexmimicgen_two_arm_threading/dexmimicgen_two_arm_threading_rot6d/10000 \\
        --port 8000
    # Terminal 2 -- run this rollout script (dexmimicgen conda env):
    conda run -n dexmimicgen python examples/dexmimicgen/rollout_sim.py \\
        --host localhost --port 8000 \\
        --dataset /path/to/two_arm_threading.hdf5 \\
        --num-episodes 1 --max-steps 400 --replan-steps 10 \\
        --video-path rollout_output.mp4
"""

import argparse
import collections
import json
import logging
import os
from pathlib import Path
import sys
import time

import dexmimicgen  # noqa: F401 -- registers custom environments
import h5py
import imageio
import numpy as np
import robosuite
import websockets.sync.client

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _REPO_ROOT / "packages" / "openpi-client" / "src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))
from openpi_client import msgpack_numpy  # noqa: E402

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))
from dexmimicgen_pi0_bridge import action_20d_from_action_dict  # noqa: E402
from dexmimicgen_pi0_bridge import action_components_for_dim  # noqa: E402
from dexmimicgen_pi0_bridge import build_observation  # noqa: E402
from dexmimicgen_pi0_bridge import convert_policy_action_to_robosuite  # noqa: E402


# ---------------------------------------------------------------------------
# Environment helpers (adapted from dexmimicgen/scripts/playback_datasets.py)
# ---------------------------------------------------------------------------
def get_env_meta_from_dataset(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def reset_to(env, state_dict):
    """Reset the environment to a specific simulator state."""
    if "model" in state_dict:
        ep_meta = {}
        if state_dict.get("ep_meta") is not None:
            ep_meta = json.loads(state_dict["ep_meta"])
        if hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        elif hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state_dict["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state_dict:
        env.sim.set_state_from_flattened(state_dict["states"])
        env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    elif hasattr(env, "update_sites"):
        env.update_sites()


def create_env(env_meta):
    """Create a robosuite env from HDF5 dataset metadata."""
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")
    return robosuite.make(**env_kwargs)


# ---------------------------------------------------------------------------
# Action comparison / visualization helpers
# ---------------------------------------------------------------------------


def _normalize_policy_actions(actions, action_dim: int = 20) -> np.ndarray:
    """Return policy actions as ``(H, action_dim)`` float32, accepting common server shapes."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"Expected policy actions with shape (H, D) or (1, H, D), got {actions.shape}")
    if actions.shape[1] < action_dim:
        raise ValueError(f"Expected policy actions with at least {action_dim} dims, got {actions.shape}")
    return actions[:, :action_dim]


def _load_ground_truth_actions_20d(episode_group: h5py.Group) -> np.ndarray:
    """Load canonical 20D ground-truth actions from a DexMimicGen episode."""
    if "action_dict" not in episode_group:
        raise KeyError(
            f"Episode {episode_group.name} has no action_dict; cannot compare policy 20D rot6d actions to ground truth"
        )
    action_group = episode_group["action_dict"]
    if "num_samples" in episode_group.attrs:
        num_samples = int(episode_group.attrs["num_samples"])
    else:
        first_key = next(iter(action_group.keys()))
        num_samples = len(action_group[first_key])
    return np.stack([action_20d_from_action_dict(action_group, i) for i in range(num_samples)], axis=0)


def _action_error_summary(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    position_threshold: float,
    rotation_threshold: float,
    gripper_threshold: float,
) -> dict:
    n = min(len(predicted), len(ground_truth))
    if n <= 0:
        return {"num_compared_steps": 0, "components": {}, "significant_error": False}

    pred = np.asarray(predicted[:n], dtype=np.float32)
    gt = np.asarray(ground_truth[:n], dtype=np.float32)
    thresholds = {
        "position": position_threshold,
        "rotation": rotation_threshold,
        "gripper": gripper_threshold,
    }
    summary = {
        "num_compared_steps": n,
        "thresholds": thresholds,
        "components": {},
        "significant_error": False,
    }
    action_components = action_components_for_dim(pred.shape[1])
    for hand, components in action_components.items():
        summary["components"][hand] = {}
        for component, sl in components.items():
            diff = pred[:, sl] - gt[:, sl]
            max_abs = float(np.max(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(np.square(diff))))
            threshold = thresholds[component]
            is_significant = max_abs > threshold
            summary["components"][hand][component] = {
                "max_abs_error": max_abs,
                "rmse": rmse,
                "threshold": threshold,
                "significant_error": is_significant,
            }
            summary["significant_error"] = bool(summary["significant_error"] or is_significant)
    return summary


def _plot_action_comparison(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    *,
    title: str,
    summary: dict,
) -> None:
    import matplotlib.pyplot as plt

    n = min(len(predicted), len(ground_truth))
    if n <= 0:
        raise ValueError("Cannot plot action comparison with zero overlapping steps")
    predicted = np.asarray(predicted[:n], dtype=np.float32)
    ground_truth = np.asarray(ground_truth[:n], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n)

    component_labels = {
        "position": ("position Δ", ("x", "y", "z")),
        "rotation": ("rotation 6D", tuple(f"r6_{i}" for i in range(6))),
    }
    colors = [f"C{i}" for i in range(6)]
    action_components = action_components_for_dim(predicted.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for row, hand in enumerate(("right", "left")):
        for col, component in enumerate(("position", "rotation", "gripper")):
            ax = axes[row, col]
            sl = action_components[hand][component]
            if component == "gripper":
                width = sl.stop - sl.start
                component_title, dim_labels = "gripper", tuple("grip" if width == 1 else f"grip_{i}" for i in range(width))
            else:
                component_title, dim_labels = component_labels[component]
            for dim_idx, label in enumerate(dim_labels):
                color = colors[dim_idx]
                ax.plot(
                    t, ground_truth[:, sl][:, dim_idx], linestyle="--", color=color, alpha=0.85, label=f"gt {label}"
                )
                ax.plot(t, predicted[:, sl][:, dim_idx], linestyle="-", color=color, alpha=0.85, label=f"pred {label}")
            component_summary = summary["components"][hand][component]
            status = "SIGNIFICANT" if component_summary["significant_error"] else "ok"
            ax.set_title(
                f"{hand} {component_title}\n"
                f"max|err|={component_summary['max_abs_error']:.4g}, "
                f"rmse={component_summary['rmse']:.4g}, {status}"
            )
            ax.grid(visible=True, alpha=0.25)
            if row == 1:
                ax.set_xlabel("rollout step")
            ax.set_ylabel("action value")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8, ncols=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_action_comparison(
    predicted: list[np.ndarray],
    ground_truth: np.ndarray,
    output_dir: Path,
    *,
    episode_index: int,
    demo_key: str,
    position_threshold: float,
    rotation_threshold: float,
    gripper_threshold: float,
) -> dict | None:
    if not predicted:
        logging.warning("No policy actions were recorded for %s; skipping action comparison plot", demo_key)
        return None

    pred = np.stack(predicted, axis=0).astype(np.float32)
    gt = ground_truth[: len(pred)].astype(np.float32)
    summary = _action_error_summary(
        pred,
        gt,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        gripper_threshold=gripper_threshold,
    )
    summary.update(
        {
            "episode_index": episode_index,
            "demo_key": demo_key,
            "num_predicted_steps": len(pred),
            "num_ground_truth_steps": len(ground_truth),
        }
    )

    episode_dir = output_dir / f"episode_{episode_index:03d}_{demo_key}"
    png_path = episode_dir / "pred_vs_ground_truth_actions.png"
    json_path = episode_dir / "pred_vs_ground_truth_actions_metrics.json"
    title = (
        f"Policy prediction vs DexMimicGen ground truth ({demo_key}, compared steps={summary['num_compared_steps']})"
    )
    _plot_action_comparison(pred, gt, png_path, title=title, summary=summary)
    episode_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if summary["significant_error"]:
        print(
            "  CODEBASE ERROR SUSPECTED: policy/ground-truth action difference exceeds "
            f"thresholds; see {png_path} and {json_path}"
        )
    else:
        print(f"  Action comparison within thresholds; saved {png_path}")
    return summary


def _write_video(video_frames: list[np.ndarray], video_path: str, *, fps: int) -> None:
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for frame in video_frames:
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"  Saved video to {video_path} ({len(video_frames)} frames at {fps} fps)")


# ---------------------------------------------------------------------------
# Policy client with extended timeout for first-inference compilation
# ---------------------------------------------------------------------------
class PolicyClient:
    """Lightweight WebSocket policy client with configurable ping timeout.
    The first model inference can take minutes (PyTorch compilation).
    The default websockets keepalive timeout of 20s is too short for that,
    so we disable pings entirely and use a generous recv timeout instead.
    """

    def __init__(self, host, port, recv_timeout=600):
        uri = f"ws://{host}:{port}"
        logging.info("Waiting for server at %s ...", uri)
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=30,
                )
                metadata = msgpack_numpy.unpackb(self._ws.recv())
                logging.info("Server metadata: %s", list(metadata.keys()) if metadata else "(empty)")
                break
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)
        self._packer = msgpack_numpy.Packer()
        self._recv_timeout = recv_timeout

    def infer(self, obs):
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv(timeout=self._recv_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Error from server:\n{response}")
        return msgpack_numpy.unpackb(response)


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------


def run_rollout(args):
    print(f"Connecting to policy server at {args.host}:{args.port} ...")
    client = PolicyClient(args.host, args.port)
    print("Connected.")
    env_meta = get_env_meta_from_dataset(args.dataset)
    env = create_env(env_meta)
    print(f"Created environment: {env_meta['env_name']}")
    print(f"Action dim: {env.action_spec[0].shape}")
    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    camera_names = args.camera_names
    total_successes = 0
    if args.action_viz_dir:
        action_viz_dir = Path(args.action_viz_dir)
    else:
        video_stem_path = Path(args.video_path).with_suffix("")
        action_viz_dir = video_stem_path.with_name(f"{video_stem_path.name}_action_viz")
    action_summaries = []
    for ep_idx in range(args.num_episodes):
        demo_key = demo_keys[ep_idx % len(demo_keys)]
        print(f"\n{'=' * 60}")
        print(f"Episode {ep_idx + 1}/{args.num_episodes} (using {demo_key} for initial state)")
        print(f"{'=' * 60}")
        with h5py.File(args.dataset, "r") as f:
            ep_grp = f[f"data/{demo_key}"]
            initial_state = {
                "states": ep_grp["states"][0],
                "model": ep_grp.attrs["model_file"],
            }
            if "ep_meta" in ep_grp.attrs:
                initial_state["ep_meta"] = ep_grp.attrs["ep_meta"]
            ground_truth_actions_20d = None if args.no_action_viz else _load_ground_truth_actions_20d(ep_grp)
        reset_to(env, initial_state)
        obs = env._get_observations()  # noqa: SLF001
        action_plan = collections.deque()
        predicted_actions_20d = []
        video_frames = []
        success = False
        for step in range(args.max_steps):
            t0 = time.time()
            if not action_plan:
                observation = build_observation(obs, prompt=args.prompt, gripper_state_dim=args.gripper_state_dim)
                result = client.infer(observation)
                action_chunk_20d = _normalize_policy_actions(result["actions"], action_dim=args.policy_action_dim)
                n_use = min(args.replan_steps, len(action_chunk_20d))
                for i in range(n_use):
                    action_20d = action_chunk_20d[i]
                    action_14d = convert_policy_action_to_robosuite(action_20d)
                    action_plan.append((action_14d, action_20d.copy()))
            action, action_20d = action_plan.popleft()
            predicted_actions_20d.append(action_20d)
            obs, reward, done, info = env.step(action)
            if env._check_success():  # noqa: SLF001
                success = True

            if step % args.video_skip == 0:
                frame_parts = []
                for cam in camera_names:
                    im = env.sim.render(
                        height=args.render_height,
                        width=args.render_width,
                        camera_name=cam,
                    )[::-1]
                    frame_parts.append(im)
                video_frames.append(np.concatenate(frame_parts, axis=1))
            elapsed = time.time() - t0
            if step % 50 == 0:
                print(
                    f"  step {step:4d}/{args.max_steps}  "
                    f"inference={elapsed * 1000:.0f}ms  "
                    f"queue={len(action_plan)}  "
                    f"success={success}"
                )
            if success and args.stop_on_success:
                print(f"  SUCCESS at step {step}!")
                break
        if success:
            total_successes += 1
        print(f"  Episode result: {'SUCCESS' if success else 'FAILURE'} ({len(video_frames)} video frames captured)")
        video_path = args.video_path
        if args.num_episodes > 1:
            base, ext = os.path.splitext(args.video_path)
            video_path = f"{base}_ep{ep_idx}{ext}"
        if not args.no_action_viz:
            action_summary = _write_action_comparison(
                predicted_actions_20d,
                ground_truth_actions_20d,
                action_viz_dir,
                episode_index=ep_idx,
                demo_key=demo_key,
                position_threshold=args.position_error_threshold,
                rotation_threshold=args.rotation_error_threshold,
                gripper_threshold=args.gripper_error_threshold,
            )
            if action_summary is not None:
                action_summaries.append(action_summary)
        if video_frames:
            fps = max(1, int((1.0 / env.control_timestep) / args.video_skip))
            try:
                _write_video(video_frames, video_path, fps=fps)
            except Exception:
                logging.exception(
                    "Failed to write video to %s. Action diagnostics were already written; "
                    "install imageio[ffmpeg] or imageio[pyav] in this environment to enable MP4 output.",
                    video_path,
                )
    print(f"\nDone. {total_successes}/{args.num_episodes} episodes succeeded.")
    if action_summaries:
        significant = sum(1 for summary in action_summaries if summary["significant_error"])
        print(f"Action comparison: {significant}/{len(action_summaries)} episodes exceeded thresholds.")
    env.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run pi0 policy rollout in DexMimicGen simulation",
    )

    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the DexMimicGen HDF5 dataset (for env metadata & initial states)",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of rollout episodes")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument(
        "--replan-steps", type=int, default=10, help="Execute this many steps from each action chunk before replanning"
    )
    parser.add_argument("--video-path", type=str, default="rollout_output.mp4", help="Output video file path")
    parser.add_argument(
        "--action-viz-dir",
        type=str,
        default=None,
        help="Directory for per-episode predicted-vs-ground-truth action plots. Defaults to '<video stem>_action_viz'.",
    )
    parser.add_argument(
        "--no-action-viz", action="store_true", help="Disable per-episode predicted-vs-ground-truth action plots"
    )
    parser.add_argument(
        "--position-error-threshold",
        type=float,
        default=0.05,
        help="Max absolute position-action error that triggers a codebase-error warning",
    )
    parser.add_argument(
        "--rotation-error-threshold",
        type=float,
        default=0.5,
        help="Max absolute rot6d-action error that triggers a codebase-error warning",
    )
    parser.add_argument(
        "--gripper-error-threshold",
        type=float,
        default=0.5,
        help="Max absolute gripper-action error that triggers a codebase-error warning",
    )
    parser.add_argument("--video-skip", type=int, default=1, help="Record every Nth frame (1 = every frame)")
    parser.add_argument(
        "--camera-names",
        type=str,
        nargs="+",
        default=["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
        help="Camera views to render (concatenated horizontally)",
    )
    parser.add_argument("--render-height", type=int, default=512, help="Render height per camera")
    parser.add_argument("--render-width", type=int, default=512, help="Render width per camera")

    parser.add_argument(
        "--prompt", type=str, default="thread the needle with both arms", help="Language prompt for the policy"
    )
    parser.add_argument(
        "--policy-action-dim",
        type=int,
        default=20,
        help="Number of leading policy action dims to execute/compare: 20 for threading, 30 for drawer cleanup.",
    )
    parser.add_argument(
        "--gripper-state-dim",
        type=int,
        default=1,
        help="Number of gripper qpos dims per hand to include in the policy observation state.",
    )
    parser.add_argument("--stop-on-success", action="store_true", help="Stop episode early upon task success")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_rollout(args)


if __name__ == "__main__":
    main()
