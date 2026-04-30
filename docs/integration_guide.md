# LiteLinear Integration Guide

This document uses the Wan2.1 integration in this tree as the concrete example.
The goal is to document the integration surfaces that exist today without
implying that a finalized high-level API has already been established.

## Recommended mental model

LiteLinear integration has two separate parts:

1. Architecture integration: replace the FFN linear layers you want to
   accelerate with `LiteLinear`.
2. Checkpoint integration: make sure checkpoint loading resolves to
   LiteLinear-compatible tensors before the model is asked to run in eval mode
   on CUDA.

These should be treated as distinct concerns. The first is model-architecture
specific. The second is checkpoint-format specific.

## What is already generic

The currently reusable components are:

- `lite_linear.LiteLinear`: the `nn.Linear` replacement used in the target FFN.
- `lite_linear.checkpoint_patch_core.resolve_or_build_patched_checkpoint()`: map
  an original `.safetensors` file to a cached LiteLinear-patched file,
  building it on first use when needed.
- `lite_linear.checkpoint_patch_core.resolve_strict_checkpoint_path()`: require
  an already-patched checkpoint and fail instead of decomposing from dense
  in-memory weights.
- `lite_linear.checkpoint_patch_core.generate_patch_config_from_checkpoint()`: generate a
  starting patch filter for a checkpoint.
- `python -m lite_linear.offline_patch`: CLI for producing a patched
  `.safetensors` file ahead of time.
- `LITELINEAR_PATCH_CONFIG`: JSON filter that selects which FFN pairs should be
  patched.
- `LITELINEAR_CACHE`: shared cache root for patched checkpoints and factor
  files.

## What is still model-specific

The parts that are still integration-specific are:

- Which modules should be replaced with `LiteLinear`.
- How to identify FFN pairs in checkpoint keys.
- How the host model redirects a checkpoint directory or shard list before
  `from_pretrained()`.
- Which environment defaults the host application wants to expose.

Today, `candidate_lite_prefix_pairs()` recognizes two FFN layouts:

- `.net.0.proj` + `.net.2` for LTX-style blocks.
- `.ffn.0` + `.ffn.2` for Wan-style blocks.

Accordingly, the offline patcher is reusable across both layouts, but it should
not yet be regarded as a universal API for automatic transformer patching.

## Recommended loading modes

### Strict mode

Strict mode is the recommended default mode.

It is active when:

- `LITELINEAR_DISABLED` is not set, and
- `LITELINEAR_ONLINE_PATCH` is not set.

Behavior:

- LiteLinear does not decompose from dense in-memory weights at `model.eval()`.
- If patchable LiteLinear modules still only have original dense checkpoint
  weights, the model raises instead of silently materializing.
- This keeps the old "load dense weights on GPU, decompose, save cache, then
  continue" path disabled.

Strict mode is appropriate when patched checkpoints are intended to be an
explicit requirement.

### Online patch mode

Set `LITELINEAR_ONLINE_PATCH=1` to enable first-load patching from original
checkpoint shards into cached patched shards.

Behavior:

- Original `.safetensors` shards are read once.
- Matching FFN weights are decomposed into `A`, `B`, `Q_fp8`, and
  `Q_scale_inv`.
- A patched shard is written into the LiteLinear cache.
- The model then loads the patched shard instead of loading the dense FFN
  weights and throwing them away.

This remains a first-run decomposition step, but it avoids the earlier
in-process path that first loaded dense weights into the model and then removed
them.

### Disabled mode

Set `LITELINEAR_DISABLED=1` to keep `LiteLinear` modules behaving like ordinary
dense linear layers for debugging.

This mode should be used only with original checkpoints that still contain
`.weight` and `.bias` tensors. Disabled mode does not accept factor-only
payloads.

## Wan2.1 as the concrete example

The Wan integration in this tree does three things:

1. It replaces only the video FFN path with `LiteLinear`.
2. It assigns stable LiteLinear module keys after model construction/load.
3. It resolves a sharded Diffusers checkpoint directory to patched shards before
   `WanModel.from_pretrained()` continues.

Relevant files:

- `wan/modules/model.py`
- `run_i2v.sh`
- `litelinear.config`
- `../examples/wan_integration.py`

### 1. Replace the target FFN modules

Wan uses explicit `nn.Sequential(linear, GELU, linear)` FFNs, so it can replace
the two FFN linears directly:

- `ffn.0`
- `ffn.2`

No activation-wrapper helper is required in this case. By contrast, LTX-style
FFNs often require `LiteLinear.replace_activation_proj_()` for the activation
projection.

### 2. Keep stable LiteLinear keys

LiteLinear uses stable module names to match factor payloads back to modules.
The standard state-dict load path usually populates `_lite_key`, but the Wan
integration also calls `assign_litelinear_module_keys(model)` after load so
that module names remain available even when the host loader path is atypical.

### 3. Resolve a checkpoint directory before load

Wan checkpoints are stored as Diffusers-style sharded directories. The Wan
integration reads the shard index, resolves each referenced shard to either:

- the original file,
- a cached patched shard built on first use, or
- an already-patched shard in strict mode.

Then it builds a patched temporary directory view with:

- remapped shard filenames,
- a rewritten `diffusion_pytorch_model.safetensors.index.json`, and
- the original `config.json`.

This patched directory is then passed to `WanModel.from_pretrained()`.

### Why Wan sets `low_cpu_mem_usage=False` for patched shards

Patched LiteLinear shards rely on `LiteLinear._load_from_state_dict()` consuming
factor payloads such as:

- `.A`
- `.B`
- `.Q_fp8`
- `.Q_scale_inv`

Diffusers' meta loader can bypass that hook, so the Wan integration switches to
the regular load path whenever a patched checkpoint directory is used.

## Minimal adaptation checklist for another model

At present, integration of another model can be decomposed into the following
steps:

1. Identify the exact FFN linears you want LiteLinear to own.
2. Replace those layers at model construction time.
3. Ensure each LiteLinear instance gets a stable module key.
4. Define how checkpoint keys map to patchable FFN pairs.
5. Add a checkpoint resolver before the model loads weights.
6. Decide which mode you want to expose:
   - strict/offline only,
   - online patch on first load,
   - or both.
7. Provide a patch-config JSON so users can narrow the integration to a subset
   of blocks while debugging.

## Offline patch flow

For a single `.safetensors` file:

```bash
python -m lite_linear.offline_patch \
  --source /path/to/model-00001-of-00008.safetensors \
  --rank 64 \
  --tag nocalib \
  --config /path/to/litelinear.config
```

If a patch config is not already available, an initial version can be generated
as follows:

```bash
python -m lite_linear.offline_patch \
  --source /path/to/model-00001-of-00008.safetensors \
  --config /path/to/litelinear.config \
  --ensure-config
```

For Diffusers-style sharded directories, the host integration should apply the
same logic shard by shard using the index file. This is the approach used by
the Wan resolver.

## Rationale For Documenting This Before Defining A New API

The existing code already has a concrete integration structure:

- module replacement,
- module naming,
- checkpoint discovery,
- checkpoint patching,
- and load-path redirection.

Documenting these pieces first makes the current constraints explicit:

- some pieces are genuinely generic,
- some remain layout-specific,
- and any future API should ideally wrap these existing steps rather than hide
  them behind an abstraction that does not align with current checkpoint
  formats.
