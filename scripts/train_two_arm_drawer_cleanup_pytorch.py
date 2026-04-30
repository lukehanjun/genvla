#!/usr/bin/env python3
"""Launch PyTorch fine-tuning for DexMimicGen TwoArmDrawerCleanup.

Examples:
  uv run python scripts/train_two_arm_drawer_cleanup_pytorch.py
  uv run python scripts/train_two_arm_drawer_cleanup_pytorch.py --dry-run --save_interval 1000
  uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_two_arm_drawer_cleanup_pytorch.py --exp_name drawer_cleanup_v1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

DEFAULT_CONFIG = "pi0_dexmimicgen_two_arm_drawer_cleanup"
DEFAULT_EXP_NAME = "dexmimicgen_two_arm_drawer_cleanup_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper around scripts/train_pytorch.py for the two-arm drawer cleanup config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="OpenPI TrainConfig to use")
    parser.add_argument("--exp_name", default=DEFAULT_EXP_NAME, help="W&B/checkpoint experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    known, passthrough = parser.parse_known_args()

    train_script = Path(__file__).resolve().parent / "train_pytorch.py"
    cmd = [sys.executable, str(train_script), known.config_name]
    if "--exp_name" not in passthrough and not any(arg.startswith("--exp_name=") for arg in passthrough):
        cmd.extend(["--exp_name", known.exp_name])
    cmd.extend(passthrough)

    if known.dry_run:
        print(" ".join(cmd))
        return

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
