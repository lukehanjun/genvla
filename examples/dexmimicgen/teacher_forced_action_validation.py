#!/usr/bin/env python3
"""Teacher-forced DexMimicGen action validation for a fine-tuned pi0 policy.

This script evaluates behavior cloning quality without rollout drift:

1. Load true observations from a DexMimicGen HDF5 demo.
2. Run the policy on each selected true demo observation.
3. Compare the predicted action chunk against ground-truth dataset actions.

By default it reports the first action in each predicted chunk, i.e. prediction
from observation t compared to ground-truth action t. Optionally, set
``--compare-chunk-steps`` > 1 to also compare chunk step k against ground-truth
action t+k for the same teacher-forced observation t.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import tqdm

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))
from dexmimicgen_pi0_bridge import ACTION_COMPONENTS  # noqa: E402
from dexmimicgen_pi0_bridge import action_20d_from_action_dict  # noqa: E402
from dexmimicgen_pi0_bridge import build_observation  # noqa: E402

COMPONENT_ORDER = (
    ("right", "position"),
    ("right", "rotation"),
    ("right", "gripper"),
    ("left", "position"),
    ("left", "rotation"),
    ("left", "gripper"),
)


def _episode_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _select_demo_keys(hdf5_path: str, demos: list[str] | None, num_demos: int | None) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        available = sorted(f["data"].keys(), key=_episode_sort_key)

    if demos:
        missing = sorted(set(demos) - set(available))
        if missing:
            raise ValueError(f"Requested demos not found in {hdf5_path}: {missing}")
        selected = demos
    else:
        selected = available

    if num_demos is not None:
        selected = selected[:num_demos]
    return selected


def _frame_indices(num_frames: int, *, stride: int, max_frames: int | None, compare_chunk_steps: int) -> np.ndarray:
    # Need room for comparing predicted chunk step k to ground-truth action t+k.
    stop = max(0, num_frames - compare_chunk_steps + 1)
    indices = np.arange(0, stop, stride, dtype=np.int64)
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _load_observation(obs_group: h5py.Group, frame: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(obs_group[key][frame]) for key in obs_group}


def _load_actions_20d(episode_group: h5py.Group) -> np.ndarray:
    if "action_dict" not in episode_group:
        raise KeyError(f"Episode {episode_group.name} has no action_dict")
    action_group = episode_group["action_dict"]
    if "num_samples" in episode_group.attrs:
        num_samples = int(episode_group.attrs["num_samples"])
    else:
        first_key = next(iter(action_group.keys()))
        num_samples = len(action_group[first_key])
    return np.stack([action_20d_from_action_dict(action_group, index) for index in range(num_samples)], axis=0)


def _normalize_policy_actions(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"Expected policy actions with shape (H, D) or (1, H, D), got {actions.shape}")
    if actions.shape[1] < 20:
        raise ValueError(f"Expected policy actions with at least 20 dims, got {actions.shape}")
    return actions[:, :20]


def _component_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for hand, component in COMPONENT_ORDER:
        metrics.setdefault(hand, {})
        sl = ACTION_COMPONENTS[hand][component]
        diff = predicted[:, sl] - ground_truth[:, sl]
        metrics[hand][component] = {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "max_abs_error": float(np.max(np.abs(diff))),
            "bias_mean": float(np.mean(diff)),
        }
    return metrics


def _flatten_component_metrics(metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    flat = {}
    for hand, component in COMPONENT_ORDER:
        for metric_name, value in metrics[hand][component].items():
            flat[f"{hand}_{component}_{metric_name}"] = value
    return flat


def _plot_first_action_timeseries(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    frames: np.ndarray,
    metrics: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    component_labels = {
        "position": ("position Δ", ("x", "y", "z")),
        "rotation": ("rotation 6D", tuple(f"r6_{i}" for i in range(6))),
        "gripper": ("grip", ("grip",)),
    }
    colors = [f"C{i}" for i in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    for row, hand in enumerate(("right", "left")):
        for col, component in enumerate(("position", "rotation", "gripper")):
            ax = axes[row, col]
            sl = ACTION_COMPONENTS[hand][component]
            component_title, dim_labels = component_labels[component]
            for dim_idx, dim_label in enumerate(dim_labels):
                color = colors[dim_idx]
                ax.plot(
                    frames,
                    ground_truth[:, sl][:, dim_idx],
                    linestyle="--",
                    color=color,
                    alpha=0.85,
                    label=f"gt {dim_label}",
                )
                ax.plot(
                    frames,
                    predicted[:, sl][:, dim_idx],
                    linestyle="-",
                    color=color,
                    alpha=0.85,
                    label=f"pred {dim_label}",
                )

            m = metrics[hand][component]
            ax.set_title(
                f"{hand} {component_title}\nrmse={m['rmse']:.4g}, mae={m['mae']:.4g}, max={m['max_abs_error']:.4g}"
            )
            ax.grid(visible=True, alpha=0.25)
            ax.set_ylabel("action value")
            if row == 1:
                ax.set_xlabel("demo frame")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8, ncols=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_error_heatmap(errors: np.ndarray, output_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(errors.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xlabel("teacher-forced sample index")
    ax.set_ylabel("action dim (20)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _validate_demo(
    policy,
    hdf5_path: str,
    demo_key: str,
    *,
    prompt: str,
    frame_stride: int,
    max_frames_per_demo: int | None,
    compare_chunk_steps: int,
    output_dir: Path,
) -> dict:
    pred_by_chunk_step = [[] for _ in range(compare_chunk_steps)]
    gt_by_chunk_step = [[] for _ in range(compare_chunk_steps)]
    frame_by_chunk_step = [[] for _ in range(compare_chunk_steps)]
    infer_times_ms = []

    with h5py.File(hdf5_path, "r") as f:
        episode_group = f[f"data/{demo_key}"]
        obs_group = episode_group["obs"]
        gt_actions = _load_actions_20d(episode_group)
        frames = _frame_indices(
            len(gt_actions),
            stride=frame_stride,
            max_frames=max_frames_per_demo,
            compare_chunk_steps=compare_chunk_steps,
        )

        for frame in tqdm.tqdm(frames, desc=f"Teacher-forced {demo_key}", leave=False):
            obs_dict = _load_observation(obs_group, int(frame))
            observation = build_observation(obs_dict, prompt=prompt)
            t0 = time.monotonic()
            result = policy.infer(observation)
            infer_times_ms.append((time.monotonic() - t0) * 1000.0)
            chunk = _normalize_policy_actions(result["actions"])
            if len(chunk) < compare_chunk_steps:
                raise ValueError(
                    f"Policy returned action horizon {len(chunk)}, but --compare-chunk-steps={compare_chunk_steps}"
                )
            for chunk_step in range(compare_chunk_steps):
                pred_by_chunk_step[chunk_step].append(chunk[chunk_step])
                gt_by_chunk_step[chunk_step].append(gt_actions[int(frame) + chunk_step])
                frame_by_chunk_step[chunk_step].append(int(frame))

    demo_dir = output_dir / demo_key
    demo_dir.mkdir(parents=True, exist_ok=True)

    chunk_summaries = []
    for chunk_step in range(compare_chunk_steps):
        predicted = np.asarray(pred_by_chunk_step[chunk_step], dtype=np.float32)
        ground_truth = np.asarray(gt_by_chunk_step[chunk_step], dtype=np.float32)
        compared_frames = np.asarray(frame_by_chunk_step[chunk_step], dtype=np.int64)
        metrics = _component_metrics(predicted, ground_truth)
        chunk_summary = {
            "chunk_step": chunk_step,
            "num_samples": len(predicted),
            "metrics": metrics,
            "metrics_flat": _flatten_component_metrics(metrics),
        }
        chunk_summaries.append(chunk_summary)

        if chunk_step == 0:
            np.savez_compressed(
                demo_dir / "teacher_forced_first_action_arrays.npz",
                frames=compared_frames,
                predicted=predicted,
                ground_truth=ground_truth,
                error=predicted - ground_truth,
            )
            _plot_first_action_timeseries(
                predicted,
                ground_truth,
                compared_frames,
                metrics,
                demo_dir / "teacher_forced_first_action_timeseries.png",
                title=f"{demo_key}: teacher-forced first action prediction vs ground truth",
            )
            _plot_error_heatmap(
                predicted - ground_truth,
                demo_dir / "teacher_forced_first_action_error_heatmap.png",
                title=f"{demo_key}: first-action prediction error",
            )

    summary = {
        "demo_key": demo_key,
        "num_selected_frames": len(frame_by_chunk_step[0]) if frame_by_chunk_step else 0,
        "frame_stride": frame_stride,
        "compare_chunk_steps": compare_chunk_steps,
        "infer_ms_mean": float(np.mean(infer_times_ms)) if infer_times_ms else None,
        "infer_ms_p50": float(np.median(infer_times_ms)) if infer_times_ms else None,
        "infer_ms_p95": float(np.percentile(infer_times_ms, 95)) if infer_times_ms else None,
        "chunk_summaries": chunk_summaries,
    }
    with (demo_dir / "teacher_forced_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _aggregate_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        return {"num_demos": 0, "chunk_summaries": []}

    compare_chunk_steps = max(summary["compare_chunk_steps"] for summary in summaries)
    chunk_summaries = []
    for chunk_step in range(compare_chunk_steps):
        flat_values: dict[str, list[float]] = {}
        total_samples = 0
        for summary in summaries:
            if chunk_step >= len(summary["chunk_summaries"]):
                continue
            chunk_summary = summary["chunk_summaries"][chunk_step]
            total_samples += chunk_summary["num_samples"]
            for key, value in chunk_summary["metrics_flat"].items():
                flat_values.setdefault(key, []).append(value)
        chunk_summaries.append(
            {
                "chunk_step": chunk_step,
                "num_samples": total_samples,
                "mean_metrics_flat": {key: float(np.mean(values)) for key, values in flat_values.items()},
            }
        )
    return {
        "num_demos": len(summaries),
        "demo_keys": [summary["demo_key"] for summary in summaries],
        "chunk_summaries": chunk_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="pi0_dexmimicgen_two_arm_threading")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory with model.safetensors or params")
    parser.add_argument("--hdf5", required=True, help="DexMimicGen HDF5 dataset")
    parser.add_argument("--demos", nargs="+", default=None, help="Specific demo keys, e.g. demo_0 demo_1")
    parser.add_argument("--num-demos", type=int, default=5, help="First N demos to evaluate when --demos is unset")
    parser.add_argument("--frame-stride", type=int, default=10, help="Evaluate every Nth demo frame")
    parser.add_argument("--max-frames-per-demo", type=int, default=None)
    parser.add_argument("--compare-chunk-steps", type=int, default=1, help="Compare predicted chunk steps 0..N-1")
    parser.add_argument("--prompt", default="thread the needle with both arms")
    parser.add_argument("--output-dir", default="viz/teacher_forced_action_validation")
    parser.add_argument("--pytorch-device", default=None, help="Override policy PyTorch device, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.compare_chunk_steps < 1:
        raise ValueError("--compare-chunk-steps must be >= 1")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_keys = _select_demo_keys(args.hdf5, args.demos, args.num_demos)
    print(f"Loading policy {args.train_config} from {args.checkpoint}")
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.train_config),
        args.checkpoint,
        default_prompt=args.prompt,
        pytorch_device=args.pytorch_device,
    )

    summaries = []
    for demo_key in demo_keys:
        summary = _validate_demo(
            policy,
            args.hdf5,
            demo_key,
            prompt=args.prompt,
            frame_stride=args.frame_stride,
            max_frames_per_demo=args.max_frames_per_demo,
            compare_chunk_steps=args.compare_chunk_steps,
            output_dir=output_dir,
        )
        summaries.append(summary)
        first = summary["chunk_summaries"][0]["metrics_flat"]
        print(
            f"{demo_key}: first-action RMSE "
            f"r_pos={first['right_position_rmse']:.4g}, "
            f"r_rot={first['right_rotation_rmse']:.4g}, "
            f"r_grip={first['right_gripper_rmse']:.4g}, "
            f"l_pos={first['left_position_rmse']:.4g}, "
            f"l_rot={first['left_rotation_rmse']:.4g}, "
            f"l_grip={first['left_gripper_rmse']:.4g}"
        )

    aggregate = _aggregate_summaries(summaries)
    with (output_dir / "teacher_forced_metrics_summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Wrote teacher-forced validation outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
