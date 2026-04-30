"""Trimmed LiteLinear integration example based on Wan2.1.

This file is intentionally smaller than the production integration in
`wan/modules/model.py`. It shows the two pieces every host model needs today:

1. Build the target FFN layers with `LiteLinear`.
2. Resolve a sharded checkpoint directory to LiteLinear-compatible shards
   before the host loader consumes it.

This example documents the current integration shape. It is not a new public
API.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

from torch import nn

try:
    from lite_linear import LiteLinear
except Exception as exc:  # pragma: no cover - example code
    LiteLinear = None
    _IMPORT_ERROR = exc
    _PATCH_HELPERS_ERROR = exc
    default_cache_root = None
    get_safe_open_for_patch_build = None
    resolve_or_build_patched_checkpoint = None
    resolve_strict_checkpoint_path = None
else:
    _IMPORT_ERROR = None
    try:
        from lite_linear.checkpoint_patch_core import (
            default_cache_root,
            resolve_or_build_patched_checkpoint,
            resolve_strict_checkpoint_path,
        )
        from lite_linear.patched_checkpoint import get_safe_open_for_patch_build
    except Exception as exc:  # pragma: no cover - example code
        _PATCH_HELPERS_ERROR = exc
        default_cache_root = None
        get_safe_open_for_patch_build = None
        resolve_or_build_patched_checkpoint = None
        resolve_strict_checkpoint_path = None
    else:
        _PATCH_HELPERS_ERROR = None


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _BOOL_TRUE


def litelinear_disabled() -> bool:
    return _env_true("LITELINEAR_DISABLED", "0")


def litelinear_online_patch_enabled() -> bool:
    return (not litelinear_disabled()) and _env_true("LITELINEAR_ONLINE_PATCH", "0")


def litelinear_strict_mode_enabled() -> bool:
    return (not litelinear_disabled()) and (not litelinear_online_patch_enabled())


def litelinear_patch_tag() -> str:
    tag = os.environ.get("LITELINEAR_PATCH_TAG", "nocalib").strip().lower()
    return "calib" if tag == "calib" else "nocalib"


def litelinear_patch_config_path() -> Optional[Path]:
    raw = os.environ.get("LITELINEAR_PATCH_CONFIG", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def build_ffn_linear(in_features: int, out_features: int) -> nn.Module:
    """Return the FFN linear layer used by the host model.

    When LiteLinear is unavailable, this falls back to a dense `nn.Linear` so
    the host model can still run unchanged.
    """

    if LiteLinear is None:
        warnings.warn(
            "LiteLinear could not be imported; falling back to nn.Linear "
            f"({_IMPORT_ERROR}).",
            RuntimeWarning,
        )
        return nn.Linear(in_features, out_features)
    return LiteLinear(in_features, out_features)


def assign_litelinear_module_keys(module: nn.Module) -> int:
    """Populate `_lite_key` from `named_modules()` for all LiteLinear modules."""

    if LiteLinear is None:
        return 0

    assigned = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, LiteLinear):
            submodule._lite_key = name
            assigned += 1
    return assigned


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _sync_file_reference(src: Path, dst: Path) -> None:
    """Point `dst` at `src` with a symlink when possible, else copy it."""

    src = src.resolve()
    if dst.is_symlink():
        try:
            if dst.resolve() == src:
                return
        except FileNotFoundError:
            pass
        _unlink_if_exists(dst)
    elif dst.exists():
        if dst.resolve() == src:
            return
        _unlink_if_exists(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_patched_diffusers_dir(
    pretrained_model_name_or_path: Union[str, Path]
) -> Union[str, Path]:
    """Resolve a Diffusers checkpoint directory to LiteLinear-compatible shards.

    In online-patch mode, this will build cached patched shards on first use.
    In strict mode, this requires those patched shards to already exist.
    """

    if LiteLinear is None or _PATCH_HELPERS_ERROR is not None:
        return pretrained_model_name_or_path
    if not (litelinear_online_patch_enabled() or litelinear_strict_mode_enabled()):
        return pretrained_model_name_or_path

    try:
        checkpoint_dir = Path(pretrained_model_name_or_path).expanduser().resolve()
    except Exception:
        return pretrained_model_name_or_path

    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    config_path = checkpoint_dir / "config.json"
    if not checkpoint_dir.is_dir() or not index_path.is_file() or not config_path.is_file():
        return pretrained_model_name_or_path

    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return pretrained_model_name_or_path

    rank = int(getattr(LiteLinear, "DEFAULT_RANK", 64))
    tag = litelinear_patch_tag()
    patch_config_path = litelinear_patch_config_path()
    patch_config_digest = "none"
    if patch_config_path is not None:
        if patch_config_path.is_file():
            patch_config_digest = hashlib.sha1(
                patch_config_path.read_bytes()
            ).hexdigest()[:16]
        else:
            patch_config_digest = hashlib.sha1(
                str(patch_config_path).encode("utf-8")
            ).hexdigest()[:16]

    mode = "online" if litelinear_online_patch_enabled() else "strict"
    dir_key = hashlib.sha1(
        (
            f"{checkpoint_dir}|mode={mode}|rank={rank}|tag={tag}|"
            f"cfg={patch_config_digest}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = default_cache_root()
    patched_dir = cache_root / "patched_model_dirs" / dir_key
    patched_dir.mkdir(parents=True, exist_ok=True)

    safe_open_fn = get_safe_open_for_patch_build()
    if litelinear_online_patch_enabled():
        resolve_shard = resolve_or_build_patched_checkpoint
    else:
        resolve_shard = resolve_strict_checkpoint_path

    patched_weight_map = {}
    for shard_name in sorted({str(name) for name in weight_map.values()}):
        src_shard = checkpoint_dir / shard_name
        if litelinear_online_patch_enabled():
            resolved = resolve_shard(
                src_shard,
                rank=rank,
                tag=tag,
                cache_root=cache_root,
                safe_open_fn=safe_open_fn,
                log_prefix="[LiteLinear Example]",
                patch_config_path=patch_config_path,
                copy_original=_env_true("LITELINEAR_PATCH_COPY_ORIGINAL", "1"),
                force_rebuild=_env_true("LITELINEAR_PATCH_FORCE_REBUILD", "0"),
            )
        else:
            resolved = resolve_shard(
                src_shard,
                rank=rank,
                tag=tag,
                cache_root=cache_root,
                safe_open_fn=safe_open_fn,
                log_prefix="[LiteLinear Example]",
                patch_config_path=patch_config_path,
            )

        _sync_file_reference(resolved, patched_dir / resolved.name)
        with safe_open_fn(str(resolved), framework="pt", device="cpu") as shard_file:
            for key in shard_file.keys():
                patched_weight_map[str(key)] = resolved.name

    patched_index = dict(data)
    patched_index["weight_map"] = patched_weight_map
    (patched_dir / index_path.name).write_text(
        json.dumps(patched_index, indent=2) + "\n",
        encoding="utf-8",
    )
    _sync_file_reference(config_path, patched_dir / config_path.name)
    return patched_dir


class ExampleVideoBlock(nn.Module):
    """Host-model block with a direct `linear -> GELU -> linear` FFN."""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            build_ffn_linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            build_ffn_linear(ffn_dim, dim),
        )


class ExampleTransformer(nn.Module):
    """Minimal host model that mirrors the Wan FFN replacement pattern."""

    def __init__(self, dim: int, ffn_dim: int, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ExampleVideoBlock(dim, ffn_dim) for _ in range(num_layers)]
        )


def prepare_for_load(model: nn.Module, checkpoint_dir: Union[str, Path]) -> Union[str, Path]:
    """Populate LiteLinear keys and resolve the checkpoint directory.

    Typical host-loader usage:

    - construct the model architecture with LiteLinear FFNs,
    - assign stable keys,
    - redirect the checkpoint directory if LiteLinear patching is enabled,
    - then call the host framework's normal load function.
    """

    assign_litelinear_module_keys(model)
    return resolve_patched_diffusers_dir(checkpoint_dir)
