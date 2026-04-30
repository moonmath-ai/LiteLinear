# Runtime Notes

This release intentionally omits native runtime internals from the public
documentation. Use the packaged wheels for deployment and the build validation
scripts to confirm the wheel payload policy.

## Compatibility

- Wheels are platform-specific and must match the target Python ABI.
- Use a CUDA-enabled PyTorch environment compatible with the wheel build.
- Rebuild wheels when changing Python, platform, CUDA/PyTorch compatibility, or
  deployment hardware assumptions.

## Validation

The release wheel validation checks that:

- Required runtime modules are present.
- Source files for the native runtime are not included in the wheel payload.
- The public Python entrypoints needed by integration flows remain available.

Run validation with:

```bash
python scripts/validate_wheel_contents.py --wheel dist/<wheel>.whl
```

## Benchmarking

For stable timing comparisons:

- Warm up the target workload before collecting timings.
- Compare against the same model, prompt, scheduler settings, precision, and
  hardware.
- Treat first-run setup costs separately from steady-state inference timings.
