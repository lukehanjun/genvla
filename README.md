## Running

### 0) Installing
```bash
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

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
Follow the repo's PyTorch setup from

```bash
uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

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