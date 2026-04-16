"""PyTorch LoRA implementation for HuggingFace Gemma models.

Mirrors the JAX LoRA in openpi/models/lora.py but operates on nn.Linear layers
in the HuggingFace PaliGemma / Gemma models used by the PyTorch training backend.
"""

import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)

ATTN_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    output = base_linear(x) + (x @ lora_A @ lora_B) * scaling
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, rslora: bool = False):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / math.sqrt(rank) if rslora else alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return base_out + lora_out * self.scaling

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


def inject_lora(model: nn.Module, lora_configs: dict, model_label: str = "") -> int:
    """Inject LoRA adapters into a HuggingFace Gemma model in-place.

    Args:
        model: A GemmaForCausalLM or PaliGemmaForConditionalGeneration model.
        lora_configs: Dict mapping target group names to LoRA config objects.
                      Expected keys: "attn" and/or "ffn", values are objects with
                      .rank, .alpha, and optionally .rslora attributes.
        model_label: Label for logging (e.g. "paligemma" or "action_expert").

    Returns:
        Number of LoRA adapters injected.
    """
    if not lora_configs:
        return 0

    attn_config = lora_configs.get("attn")
    ffn_config = lora_configs.get("ffn")

    target_modules = set()
    configs_by_name = {}
    if attn_config:
        target_modules.update(ATTN_TARGET_MODULES)
        for name in ATTN_TARGET_MODULES:
            configs_by_name[name] = attn_config
    if ffn_config:
        target_modules.update(FFN_TARGET_MODULES)
        for name in FFN_TARGET_MODULES:
            configs_by_name[name] = ffn_config

    count = 0
    for parent_name, parent_module in model.named_modules():
        for attr_name in list(target_modules):
            if not hasattr(parent_module, attr_name):
                continue
            original = getattr(parent_module, attr_name)
            if not isinstance(original, nn.Linear):
                continue

            cfg = configs_by_name[attr_name]
            lora_layer = LoRALinear(
                original,
                rank=cfg.rank,
                alpha=cfg.alpha,
                rslora=getattr(cfg, "rslora", False),
            )
            setattr(parent_module, attr_name, lora_layer)
            count += 1

    if count > 0:
        logger.info(f"Injected {count} LoRA adapters into {model_label or 'model'} "
                     f"(attn rank={attn_config.rank if attn_config else 'N/A'}, "
                     f"ffn rank={ffn_config.rank if ffn_config else 'N/A'})")
    return count


def get_lora_param_names(model: nn.Module) -> set[str]:
    """Return the set of parameter names that belong to LoRA adapters."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}." if name else ""
            names.add(f"{prefix}lora_A.weight")
            names.add(f"{prefix}lora_B.weight")
    return names


def freeze_base_for_lora(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except LoRA adapters and non-LLM parameters.

    For LoRA fine-tuning, we freeze the LLM backbone weights (everything under
    the paligemma language model and gemma expert) but keep LoRA adapters and
    task-specific heads (action projections, state projections, etc.) trainable.

    Returns:
        (num_trainable, num_frozen) parameter counts.
    """
    lora_names = get_lora_param_names(model)

    llm_prefixes = (
        "paligemma_with_expert.paligemma.language_model.",
        "paligemma_with_expert.paligemma.model.language_model.",
        "paligemma_with_expert.gemma_expert.",
    )

    num_trainable = 0
    num_frozen = 0
    for name, param in model.named_parameters():
        is_lora = name in lora_names
        is_llm = any(name.startswith(p) for p in llm_prefixes)

        if is_llm and not is_lora:
            param.requires_grad = False
            num_frozen += param.numel()
        else:
            param.requires_grad = True
            num_trainable += param.numel()

    logger.info(f"LoRA freeze: {num_trainable:,} trainable params, {num_frozen:,} frozen params")
    return num_trainable, num_frozen
