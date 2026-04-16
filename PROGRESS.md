# PROGRESS

## Current status
The custom DexMimicGen fine-tuning path is now scaffolded in the repo for `pi0_base`, and is now wired for the **PyTorch** training path rather than the JAX trainer.

## Confirmed decisions
- Use base checkpoint: `gs://openpi-assets/checkpoints/pi0_base`
- Use the **converted PyTorch checkpoint** at `checkpoints/pi0_base_pytorch/` for finetuning
- Do **not** use `pi0_droid`
- Do **not** reuse pretrained norm stats
- Do **not** reuse stock task configs as the final solution
- Build a **custom DexMimicGen bimanual** setup
- Scope for now: offline fine-tuning only
- Keep future policy-server integration easy by preserving an observation + prompt schema compatible with `docs/remote_inference.md`
- Use a **per-episode prompt field**, initially filled with one default string that can be edited later
- Remaining design details may be chosen autonomously if they preserve consistency and future policy-server friendliness

## Implemented files
- `examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py`
- `src/openpi/policies/dexmimicgen_policy.py`
- `src/openpi/policies/dexmimicgen_policy_test.py`
- `src/openpi/training/config.py`

## Repository findings that shaped the implementation
- `README.md` defines the training flow: convert dataset -> define config -> compute norm stats -> run training.
- `docs/norm_stats.md` documents norm-stat reuse, but this project will compute fresh stats instead.
- `docs/remote_inference.md` shows future remote serving expects observation dictionaries plus `prompt`.
- `examples/droid/convert_droid_data_to_lerobot.py` is single-arm and not a direct match.
- `examples/aloha_real/convert_aloha_data_to_lerobot.py` and `LeRobotAlohaDataConfig` are the closest built-in examples for dual-arm state/action layouts.

## Dataset findings
Source dataset:
- `/home/exx/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5`

Observed structure:
- 1025 episodes under `/data/demo_*`
- `actions`: shape `(T, 14)`
- dual-arm observations for `robot0_*` and `robot1_*`
- image observations including `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`
- no obvious language annotation field

Implication:
- The implementation should treat DexMimicGen as a **dual-arm custom pi0 task**, not as DROID.

## Implemented schema and wrappers
### LeRobot dataset schema
- `observation.images.top`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state` (20 dims)
- `action` (20 dims)
- per-frame `task` string used as the per-episode prompt seed

### State layout
The custom state vector is:
1. right arm (`robot0`) end-effector position: 3 dims
2. right arm (`robot0`) end-effector rotation: 6D rotation representation
3. right arm gripper opening scalar: 1 dim
4. left arm (`robot1`) end-effector position: 3 dims
5. left arm (`robot1`) end-effector rotation: 6D rotation representation
6. left arm gripper opening scalar: 1 dim

Total: 20 dims

Notes:
- quaternions are standardized to positive-`w` before conversion to rotation matrices / rot6d
- the state layout is intentionally aligned with the 20-dim action layout so the policy sees a consistent right-then-left ordering
- empirical check against end-effector deltas showed `robot0` aligns with the dataset's `right_*` action channels and `robot1` aligns with `left_*`

### Action layout
The custom action vector is reconstructed from `action_dict` and kept in dataset-native order:
1. right arm relative position delta: 3 dims
2. right arm relative rotation delta: 6D rotation representation
3. right arm gripper command: 1 dim
4. left arm relative position delta: 3 dims
5. left arm relative rotation delta: 6D rotation representation
6. left arm gripper command: 1 dim

Total: 20 dims

### Policy wrapper contract
`src/openpi/policies/dexmimicgen_policy.py` defines the runtime-facing observation schema:
- `images.top`
- `images.left_wrist`
- `images.right_wrist`
- `state`
- `prompt`

The training config repacks the LeRobot dataset fields into that runtime-facing schema so that future
policy-server integration can reuse the same observation layout.

## Custom train config
Added config name:
- `pi0_dexmimicgen_two_arm_threading`

Key properties:
- base model: `pi0_base`
- PyTorch initialization checkpoint: `checkpoints/pi0_base_pytorch`
- PyTorch training precision: `bfloat16` (recommended default to reduce memory use)
- dataset repo id: `local/dexmimicgen_two_arm_threading`
- prompt source: LeRobot `task` field
- default prompt fallback: `thread the needle with both arms`
- batch size: `32`
- train steps: `20_000`

Norm stats are expected at:
- `assets/pi0_dexmimicgen_two_arm_threading/local/dexmimicgen_two_arm_threading`

## Verification performed
- `uv run ruff check src/openpi/policies/dexmimicgen_policy.py src/openpi/policies/dexmimicgen_policy_test.py examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py src/openpi/training/config.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest src/openpi/policies/dexmimicgen_policy_test.py -q`
- smoke conversion:
  - `uv run examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py --dataset-path /home/exx/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5 --repo-id local/dexmimicgen_two_arm_threading_smoke --max-episodes 1`
- smoke LeRobot metadata check:
  - verified `fps == 20`
  - verified task mapping `{0: "thread the needle with both arms"}`
- smoke data-loader check against the converted smoke dataset:
  - state padded to `(batch, 32)`
  - actions chunked / padded to `(batch, 50, 32)`
  - prompt tokenization works
- config smoke check:
  - `pi0_dexmimicgen_two_arm_threading.pytorch_weight_path == "checkpoints/pi0_base_pytorch"`
  - `pi0_dexmimicgen_two_arm_threading.pytorch_training_precision == "bfloat16"`

## Important non-goals
- No real robot runtime integration yet
- No full policy-server client integration yet
- No reuse of pretrained DROID-specific assumptions

## CLI commands
### 1) Convert the DexMimicGen HDF5 to LeRobot
```bash
uv run python examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py \
  --dataset-path /home/exx/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5 \
  --repo-id local/dexmimicgen_two_arm_threading \
  --default-prompt "thread the needle with both arms"
```

Optional smoke test:
```bash
uv run python examples/dexmimicgen/convert_dexmimicgen_data_to_lerobot.py \
  --dataset-path /home/exx/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5 \
  --repo-id local/dexmimicgen_two_arm_threading_smoke \
  --max-episodes 1
```

### 2) Compute fresh normalization statistics
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_dexmimicgen_two_arm_threading
```

### 3) Prepare the PyTorch environment
Follow the repo's PyTorch setup from `README.md`:

```bash
uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

The custom config is already wired to load:

```text
checkpoints/pi0_base_pytorch/
```

so you do not need to pass the PyTorch checkpoint path on the CLI.

### 4) Run fine-tuning with PyTorch
```bash
uv run python scripts/train_pytorch.py pi0_dexmimicgen_two_arm_threading --exp_name dexmimicgen_two_arm_threading_v1
```

Optional multi-GPU:
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi0_dexmimicgen_two_arm_threading \
  --exp_name dexmimicgen_two_arm_threading_v1
```

### 5) Later, serve a trained PyTorch checkpoint with the custom config
```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi0_dexmimicgen_two_arm_threading \
  --policy.dir=checkpoints/pi0_dexmimicgen_two_arm_threading/dexmimicgen_two_arm_threading_v1/<step>
```

### 6) Visualizing
```bash
# Visualize dataset only
uv run examples/dexmimicgen/visualize_actions.py --ground-truth-only --episode-idx 0

# Visualize model output
uv run examples/dexmimicgen/visualize_actions.py \
  --checkpoint-dir checkpoints/pi0_dexmimicgen_two_arm_threading/dexmimicgen_two_arm_threading_rot6d/10000 \
  --episode-idx 0 --output-dir data/dexmimicgen/viz/
```

### 7) Running in Dexmimicgen
```bash
# Policy server
cd /home/horowitz3/pi0/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi0_dexmimicgen_two_arm_threading \
  --policy.dir=checkpoints/pi0_dexmimicgen_two_arm_threading/dexmimicgen_two_arm_threading_rot6d/10000


# Dexmimicgen
conda activate dexmimicgen
python examples/dexmimicgen/rollout_sim.py \
  --host localhost --port 8000 \
  --dataset /home/horowitz3/pi0/dexmimicgen/datasets/generated/two_arm_threading.hdf5 \
  --num-episodes 1 --max-steps 400 --replan-steps 10 \
  --video-path rollout_output.mp4 \
  --camera-names agentview robot0_eye_in_hand robot1_eye_in_hand \
  --stop-on-success
```

## Artifact references
- Context snapshot: `.omx/context/fine-tune-pi0-droid-on-dexmimicgen-20260410T214353Z.md`
- Deep-interview spec: `.omx/specs/deep-interview-fine-tune-pi0-droid-on-dexmimicgen.md`
- Deep-interview transcript: see `.omx/interviews/`
