"""
FFN patch utilities independent of low-rank code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional, Tuple

import torch
from torch import nn

try:
    import transformer_engine.pytorch as te

    _HAS_TE = True
    _TE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency
    te = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = e

FFNPart = Literal["w1", "w2"]


@dataclass(frozen=True)
class FFNLinearRef:
    block_idx: int
    part: FFNPart  # w1 = activation proj, w2 = output proj
    name: str  # stable-ish module name for logging/serialization


def iter_ltx_ffn_linears(
    transformer: nn.Module,
) -> Iterator[Tuple[FFNLinearRef, nn.Module]]:
    """
    Yield references to the two FFN linear layers per transformer block:
    - w1: activation projection inside ff.net[0] (GEGLU/GELU/etc)
    - w2: output projection ff.net[2]
    """
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Expected transformer to have .transformer_blocks")

    for i, block in enumerate(blocks):
        ff = getattr(block, "ff", None)
        if ff is None or not hasattr(ff, "net"):
            continue

        # w1
        act = ff.net[0]
        proj = getattr(act, "proj", None)
        if proj is None:
            # Fall back: pick the first submodule with a `weight` shaped like linear.
            for _, m in act.named_modules():
                if hasattr(m, "weight") and isinstance(
                    getattr(m, "weight"), torch.Tensor
                ):
                    proj = m
                    break
        if proj is None:
            raise RuntimeError(f"Could not find FFN w1 proj in block {i}")
        yield FFNLinearRef(i, "w1", f"transformer_blocks.{i}.ff.w1"), proj

        # w2
        out = ff.net[2]
        yield FFNLinearRef(i, "w2", f"transformer_blocks.{i}.ff.w2"), out


def _get_weight_bias(m: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(m, "weight"):
        raise ValueError("Module has no .weight")
    W = getattr(m, "weight")
    b = getattr(m, "bias", None)
    return W, b


def _build_te_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    if not _HAS_TE:
        raise RuntimeError(f"transformer_engine not available ({_TE_IMPORT_ERROR})")
    try:
        return te.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            params_dtype=dtype,
        )
    except TypeError:
        # Older TE versions may not accept device/params_dtype in constructor.
        mod = te.Linear(in_features, out_features, bias=bias)
        return mod.to(device=device, dtype=dtype)


@torch.no_grad()
def apply_te_linear_to_transformer(
    transformer: nn.Module,
    *,
    which: Iterable[FFNPart] = ("w1", "w2"),
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> int:
    """
    In-place patching: replace FFN linear layers with TransformerEngine Linear.

    This is intended as a drop-in alternative to torch.nn.Linear for BF16 tests.
    """
    if not _HAS_TE:
        if strict:
            raise RuntimeError(f"transformer_engine not available ({_TE_IMPORT_ERROR})")
        print(f"[ffn_patch] transformer_engine not available; skipping TE patch ({_TE_IMPORT_ERROR})")
        return 0

    which_set = set(which)
    replaced = 0
    for ref, mod in iter_ltx_ffn_linears(transformer):
        if ref.part not in which_set:
            continue
        try:
            W, b = _get_weight_bias(mod)
        except ValueError:
            if strict:
                raise
            continue
        if not W.is_cuda:
            if strict:
                raise RuntimeError(
                    f"TE Linear requires CUDA tensors; got {ref.name} on {W.device}"
                )
            print(f"[ffn_patch] Skipping {ref.name} (device={W.device})")
            continue

        out_features, in_features = W.shape
        te_mod = _build_te_linear(
            in_features,
            out_features,
            bias=b is not None,
            device=W.device,
            dtype=dtype,
        )
        te_mod.weight.copy_(W.to(dtype=dtype))
        if b is not None and getattr(te_mod, "bias", None) is not None:
            te_mod.bias.copy_(b.to(dtype=dtype))
        te_mod.eval()

        block = transformer.transformer_blocks[ref.block_idx]
        ff = block.ff
        if ref.part == "w2":
            ff.net[2] = te_mod
        else:
            act = ff.net[0]
            if hasattr(act, "proj"):
                act.proj = te_mod
            else:
                if strict:
                    raise RuntimeError(
                        f"Cannot replace w1 for block {ref.block_idx}: activation has no .proj"
                    )
                continue
        replaced += 1

    return replaced

