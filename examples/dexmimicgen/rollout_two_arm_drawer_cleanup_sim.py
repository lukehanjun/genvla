#!/usr/bin/env python3
"""Run a two-arm drawer cleanup pi0 policy in DexMimicGen simulation.

Start a matching policy server first, for example:
  uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi0_dexmimicgen_two_arm_drawer_cleanup \
    --policy.dir=checkpoints/pi0_dexmimicgen_two_arm_drawer_cleanup/drawer_cleanup_v1/10000

Then run this script in the DexMimicGen/robosuite environment:
  python examples/dexmimicgen/rollout_two_arm_drawer_cleanup_sim.py --host localhost --port 8000
"""

from __future__ import annotations

from pathlib import Path
import sys

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))

from rollout_sim import main as rollout_main  # noqa: E402

DEFAULT_DATASET = "/workspace/hjyu/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5"
DEFAULT_PROMPT = "clean up the drawer with both arms"


def main() -> None:
    default_args = [
        "--dataset",
        DEFAULT_DATASET,
        "--prompt",
        DEFAULT_PROMPT,
        "--policy-action-dim",
        "30",
        "--gripper-state-dim",
        "6",
        "--video-path",
        "drawer_cleanup_rollout.mp4",
        "--camera-names",
        "agentview",
        "robot0_eye_in_hand",
        "robot1_eye_in_hand",
    ]
    sys.argv = [sys.argv[0], *default_args, *sys.argv[1:]]
    rollout_main()


if __name__ == "__main__":
    main()
