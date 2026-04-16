import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

DEFAULT_PROMPT = "thread the needle with both arms"


def make_dexmimicgen_example() -> dict:
    """Creates a random input example for the DexMimicGen bimanual policy."""
    return {
        "state": np.ones((20,), dtype=np.float32),
        "images": {
            "top": np.random.randint(256, size=(84, 84, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(84, 84, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(84, 84, 3), dtype=np.uint8),
        },
        "prompt": DEFAULT_PROMPT,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexMimicGenInputs(transforms.DataTransformFn):
    """Input adapter for DexMimicGen bimanual observations/actions."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {name: _parse_image(image) for name, image in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain only {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        top_image = in_images["top"]

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": in_images.get("left_wrist", np.zeros_like(top_image)),
                "right_wrist_0_rgb": in_images.get("right_wrist", np.zeros_like(top_image)),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.asarray("left_wrist" in in_images),
                "right_wrist_0_rgb": np.asarray("right_wrist" in in_images),
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DexMimicGenOutputs(transforms.DataTransformFn):
    """Output adapter for DexMimicGen bimanual actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20], dtype=np.float32)}
