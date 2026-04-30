#!/usr/bin/env python3
"""Task-specific converter for DexMimicGen TwoArmDrawerCleanup.

This is a thin wrapper around ``convert_dexmimicgen_data_to_lerobot.py`` with
safe defaults for the local drawer-cleanup HDF5, LeRobot repo id, and prompt.
User-provided CLI flags are appended after the defaults, so they override them.
"""

from __future__ import annotations

from pathlib import Path
import sys

import tyro

from openpi.policies import dexmimicgen_policy

_DEX_DIR = Path(__file__).resolve().parent
if str(_DEX_DIR) not in sys.path:
    sys.path.insert(0, str(_DEX_DIR))

from convert_dexmimicgen_data_to_lerobot import Args  # noqa: E402
from convert_dexmimicgen_data_to_lerobot import main as convert_main  # noqa: E402

DEFAULT_DATASET = "/workspace/hjyu/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5"
DEFAULT_REPO_ID = "local/dexmimicgen_two_arm_drawer_cleanup"


def main() -> None:
    default_args = [
        "--dataset-path",
        DEFAULT_DATASET,
        "--repo-id",
        DEFAULT_REPO_ID,
        "--default-prompt",
        dexmimicgen_policy.DRAWER_CLEANUP_PROMPT,
    ]
    sys.argv = [sys.argv[0], *default_args, *sys.argv[1:]]
    convert_main(tyro.cli(Args))


if __name__ == "__main__":
    main()
