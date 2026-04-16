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
import sys
import time
from pathlib import Path

import h5py
import imageio
import numpy as np
import robosuite
import websockets.sync.client
import dexmimicgen  # noqa: F401 -- registers custom environments
from openpi_client import msgpack_numpy

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))
from dexmimicgen_pi0_bridge import build_observation
from dexmimicgen_pi0_bridge import convert_action_20d_to_14d

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
    for ep_idx in range(args.num_episodes):
        demo_key = demo_keys[ep_idx % len(demo_keys)]
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{args.num_episodes} (using {demo_key} for initial state)")
        print(f"{'='*60}")
        with h5py.File(args.dataset, "r") as f:
            ep_grp = f[f"data/{demo_key}"]
            initial_state = {
                "states": ep_grp["states"][0],
                "model": ep_grp.attrs["model_file"],
            }
            if "ep_meta" in ep_grp.attrs:
                initial_state["ep_meta"] = ep_grp.attrs["ep_meta"]
        reset_to(env, initial_state)
        obs = env._get_observations()
        action_plan = collections.deque()
        video_frames = []
        success = False
        for step in range(args.max_steps):
            t0 = time.time()
            if not action_plan:
                observation = build_observation(obs, prompt=args.prompt)
                result = client.infer(observation)
                action_chunk_20d = result["actions"]
                n_use = min(args.replan_steps, len(action_chunk_20d))
                for i in range(n_use):
                    action_14d = convert_action_20d_to_14d(action_chunk_20d[i])
                    action_plan.append(action_14d)
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action)
            if env._check_success():
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
                print(f"  step {step:4d}/{args.max_steps}  "
                      f"inference={elapsed*1000:.0f}ms  "
                      f"queue={len(action_plan)}  "
                      f"success={success}")
            if success and args.stop_on_success:
                print(f"  SUCCESS at step {step}!")
                break
        if success:
            total_successes += 1
        print(f"  Episode result: {'SUCCESS' if success else 'FAILURE'} "
              f"({len(video_frames)} video frames captured)")
        video_path = args.video_path
        if args.num_episodes > 1:
            base, ext = os.path.splitext(args.video_path)
            video_path = f"{base}_ep{ep_idx}{ext}"
        if video_frames:
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            fps = max(1, int((1.0 / env.control_timestep) / args.video_skip))
            writer = imageio.get_writer(video_path, fps=fps)
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Saved video to {video_path} ({len(video_frames)} frames at {fps} fps)")
    print(f"\nDone. {total_successes}/{args.num_episodes} episodes succeeded.")
    env.close()
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run pi0 policy rollout in DexMimicGen simulation",
    )

    parser.add_argument("--host", type=str, default="localhost",
                        help="Policy server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Policy server port")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the DexMimicGen HDF5 dataset (for env metadata & initial states)")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of rollout episodes")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Maximum steps per episode")
    parser.add_argument("--replan-steps", type=int, default=10,
                        help="Execute this many steps from each action chunk before replanning")
    parser.add_argument("--video-path", type=str, default="rollout_output.mp4",
                        help="Output video file path")
    parser.add_argument("--video-skip", type=int, default=1,
                        help="Record every Nth frame (1 = every frame)")
    parser.add_argument("--camera-names", type=str, nargs="+",
                        default=["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
                        help="Camera views to render (concatenated horizontally)")
    parser.add_argument("--render-height", type=int, default=512,
                        help="Render height per camera")
    parser.add_argument("--render-width", type=int, default=512,
                        help="Render width per camera")
    
    parser.add_argument("--prompt", type=str, default="thread the needle with both arms",
                        help="Language prompt for the policy")
    parser.add_argument("--stop-on-success", action="store_true",
                        help="Stop episode early upon task success")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_rollout(args)
if __name__ == "__main__":
    main()