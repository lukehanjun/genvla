import numpy as np

from openpi.policies import dexmimicgen_policy


def test_dexmimicgen_inputs_outputs():
    example = dexmimicgen_policy.make_dexmimicgen_example()
    example["actions"] = np.random.rand(10, 20).astype(np.float32)
    example["prompt"] = b"thread the needle with both arms"
    example["images"]["top"] = np.ones((3, 84, 84), dtype=np.float32)

    inputs = dexmimicgen_policy.DexMimicGenInputs()(example)

    assert inputs["state"].shape == (20,)
    assert inputs["image"]["base_0_rgb"].shape[-1] == 3
    assert inputs["image"]["left_wrist_0_rgb"].shape[-1] == 3
    assert inputs["image"]["right_wrist_0_rgb"].shape[-1] == 3
    assert inputs["actions"].shape == (10, 20)
    assert inputs["prompt"] == dexmimicgen_policy.DEFAULT_PROMPT

    outputs = dexmimicgen_policy.DexMimicGenOutputs()({"actions": np.random.rand(10, 32).astype(np.float32)})
    assert outputs["actions"].shape == (10, 20)

    drawer_outputs = dexmimicgen_policy.DexMimicGenOutputs(action_dim=30)(
        {"actions": np.random.rand(10, 32).astype(np.float32)}
    )
    assert drawer_outputs["actions"].shape == (10, 30)
