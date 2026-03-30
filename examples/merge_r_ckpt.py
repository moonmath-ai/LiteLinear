#!/usr/bin/env python3
"""
Merge per-rank checkpoint accumulators into raw xtx + n_rows (and optional R).

Usage:
  python ffn/merge_r_ckpt.py --out_raw merged.raw.pt ckpt.rank0 ckpt.rank1 ...
  python ffn/merge_r_ckpt.py --out_raw merged.raw.pt --out merged.pt ckpt.rank0 ckpt.rank1 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def _load_ckpt(path: Path) -> Tuple[int, Dict[str, Dict]]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    seq_idx = int(ckpt.get("seq_idx", ckpt.get("prompt_idx", 0)))
    accs = ckpt.get("accs", {}) or {}
    if not isinstance(accs, dict) or not accs:
        raise ValueError(f"Checkpoint has no accs: {path}")
    return seq_idx, accs


def _iter_accs(accs: Dict[str, Dict]) -> Iterable[Tuple[str, torch.Tensor, int]]:
    for name, entry in accs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid acc entry for {name}: {type(entry)}")
        xtx = entry.get("xtx")
        n_rows = entry.get("n_rows")
        if xtx is None or n_rows is None:
            raise ValueError(f"Missing xtx/n_rows for {name}")
        yield name, xtx, int(n_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="Paths to ckpt.rank* files")
    ap.add_argument("--out_raw", required=True, help="Output path for raw xtx + n_rows")
    ap.add_argument("--out", default=None, help="Optional output path for normalized R")
    ap.add_argument(
        "--allow_mismatch",
        action="store_true",
        help="Allow merging checkpoints with different seq_idx (not recommended)",
    )
    args = ap.parse_args()

    seqs = []
    accs_list = []
    for p in args.checkpoints:
        seq_idx, accs = _load_ckpt(Path(p))
        seqs.append(seq_idx)
        accs_list.append(accs)

    if len(set(seqs)) != 1 and not args.allow_mismatch:
        raise SystemExit(
            f"Checkpoint seq_idx mismatch: {seqs}. "
            "Re-run with --allow_mismatch to merge anyway."
        )

    total_xtx: Dict[str, torch.Tensor] = {}
    total_n_rows: Dict[str, int] = {}
    names = set(accs_list[0].keys())
    for accs in accs_list[1:]:
        if set(accs.keys()) != names:
            raise SystemExit("Checkpoint accs keys mismatch across ranks.")

    for accs in accs_list:
        for name, xtx, n_rows in _iter_accs(accs):
            xtx_cpu = xtx.detach().cpu()
            if name not in total_xtx:
                total_xtx[name] = xtx_cpu.clone()
                total_n_rows[name] = int(n_rows)
            else:
                total_xtx[name] += xtx_cpu
                total_n_rows[name] += int(n_rows)

    raw_payload = {
        "version": 1,
        "xtx": total_xtx,
        "n_rows": total_n_rows,
    }
    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw_payload, out_raw)
    print(f"Wrote raw: {out_raw}")

    if args.out:
        r_mats: Dict[str, torch.Tensor] = {}
        for name, xtx in total_xtx.items():
            n_rows = total_n_rows.get(name, 0)
            if n_rows <= 0:
                raise RuntimeError(f"No samples for {name}")
            r_mats[name] = (xtx / float(n_rows)).to(dtype=torch.float32)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in r_mats.items()}, out)
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
