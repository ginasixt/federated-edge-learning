#!/usr/bin/env python3
"""
Check whether PyTorch checkpoint files exist, are loadable, and change over FL rounds.

Expected filenames:
    model_round_<round>_run_<run_tag>.pt

Example:
    python check_checkpoint_dynamics.py \
      --checkpoint-dir "result/splits_iid_scaling/splits_iid_32768_clients.json/all_rounds" \
      --run-tag 1 \
      --expected-rounds 80 \
      --out-csv checkpoint_dynamics.csv

This script computes:
- whether each checkpoint is loadable
- whether tensors are finite
- SHA256 hash per checkpoint
- parameter L2 norm per checkpoint
- L2/relative L2/max-absolute difference to previous round
- cosine similarity to previous round
- whether consecutive checkpoints are exactly identical
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return a plain state_dict."""
    obj = torch.load(path, map_location="cpu")

    # Support plain state_dict and common wrapper format.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict: {type(obj)}")

    state: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor value for key {key!r}: {type(value)}")
        state[str(key)] = value.detach().cpu().float().contiguous()

    if not state:
        raise ValueError("Empty state_dict")

    return state


def flatten_state_dict(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten tensors in deterministic key order."""
    parts = []
    for key in sorted(state.keys()):
        parts.append(state[key].reshape(-1))
    return torch.cat(parts)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash keys, shapes, and raw tensor bytes in deterministic order."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def discover_rounds(checkpoint_dir: Path, run_tag: str) -> dict[int, Path]:
    pattern = re.compile(rf"^model_round_(\d+)_run_{re.escape(run_tag)}\.pt$")
    found: dict[int, Path] = {}
    for path in checkpoint_dir.glob(f"model_round_*_run_{run_tag}.pt"):
        match = pattern.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def safe_float(x: torch.Tensor | float) -> float:
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing model_round_...pt files")
    parser.add_argument("--run-tag", default="1", help="Run tag used in filenames, e.g. 1")
    parser.add_argument("--expected-rounds", type=int, default=None, help="Expected number of rounds, e.g. 80")
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--min-delta", type=float, default=1e-12, help="Warn if consecutive L2 delta <= this value")
    parser.add_argument("--plot", default=None, help="Optional PNG path for a delta-over-rounds plot")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint directory does not exist: {checkpoint_dir}")
        return 2

    rounds = discover_rounds(checkpoint_dir, str(args.run_tag))
    if not rounds:
        print(f"ERROR: no checkpoints found in {checkpoint_dir} for run tag {args.run_tag}")
        return 2

    if args.expected_rounds is not None:
        expected = set(range(1, args.expected_rounds + 1))
        found = set(rounds.keys())
        missing = sorted(expected - found)
        extra = sorted(found - expected)
    else:
        missing = []
        extra = []

    rows: list[dict[str, Any]] = []
    previous_vec: torch.Tensor | None = None
    previous_hash: str | None = None
    previous_keys: list[str] | None = None
    invalid: list[tuple[int, str]] = []

    for rnd, path in rounds.items():
        row: dict[str, Any] = {
            "round": rnd,
            "path": str(path),
            "filename": path.name,
            "file_size_bytes": path.stat().st_size,
        }

        try:
            state = load_checkpoint(path)
            keys = sorted(state.keys())
            vec = flatten_state_dict(state)
            h = state_hash(state)

            finite = bool(torch.isfinite(vec).all().item())
            param_l2 = safe_float(torch.linalg.vector_norm(vec))
            param_mean_abs = safe_float(vec.abs().mean())
            param_max_abs = safe_float(vec.abs().max())

            row.update({
                "loadable": True,
                "finite": finite,
                "num_tensors": len(state),
                "num_parameters": int(vec.numel()),
                "sha256": h,
                "param_l2": param_l2,
                "param_mean_abs": param_mean_abs,
                "param_max_abs": param_max_abs,
            })

            if previous_vec is None:
                row.update({
                    "delta_l2_from_prev": None,
                    "relative_delta_l2_from_prev": None,
                    "max_abs_delta_from_prev": None,
                    "mean_abs_delta_from_prev": None,
                    "cosine_similarity_to_prev": None,
                    "identical_to_prev": None,
                    "keys_same_as_prev": None,
                })
            else:
                keys_same = keys == previous_keys
                if vec.numel() != previous_vec.numel():
                    raise ValueError(
                        f"Parameter vector length changed: previous={previous_vec.numel()}, current={vec.numel()}"
                    )

                delta = vec - previous_vec
                delta_l2 = safe_float(torch.linalg.vector_norm(delta))
                prev_l2 = safe_float(torch.linalg.vector_norm(previous_vec))
                rel_delta = delta_l2 / prev_l2 if prev_l2 > 0 else math.nan
                max_abs_delta = safe_float(delta.abs().max())
                mean_abs_delta = safe_float(delta.abs().mean())

                denom = safe_float(torch.linalg.vector_norm(vec) * torch.linalg.vector_norm(previous_vec))
                cosine = safe_float(torch.dot(vec, previous_vec) / denom) if denom > 0 else math.nan
                identical = h == previous_hash

                row.update({
                    "delta_l2_from_prev": delta_l2,
                    "relative_delta_l2_from_prev": rel_delta,
                    "max_abs_delta_from_prev": max_abs_delta,
                    "mean_abs_delta_from_prev": mean_abs_delta,
                    "cosine_similarity_to_prev": cosine,
                    "identical_to_prev": identical,
                    "keys_same_as_prev": keys_same,
                })

            previous_vec = vec
            previous_hash = h
            previous_keys = keys

        except Exception as exc:
            invalid.append((rnd, str(exc)))
            row.update({
                "loadable": False,
                "error": str(exc),
            })

        rows.append(row)

    # Summary
    loaded_rows = [r for r in rows if r.get("loadable")]
    duplicate_hashes = len({r.get("sha256") for r in loaded_rows}) < len(loaded_rows)
    identical_consecutive = [
        int(r["round"]) for r in loaded_rows
        if r.get("identical_to_prev") is True
    ]
    near_zero_delta = [
        int(r["round"]) for r in loaded_rows
        if r.get("delta_l2_from_prev") is not None
        and float(r["delta_l2_from_prev"]) <= args.min_delta
    ]

    print("\n=== CHECKPOINT DYNAMICS SUMMARY ===")
    print(f"Directory: {checkpoint_dir}")
    print(f"Run tag: {args.run_tag}")
    print(f"Found checkpoint rounds: {min(rounds)}-{max(rounds)} ({len(rounds)} files)")

    if args.expected_rounds is not None:
        print(f"Expected rounds: 1-{args.expected_rounds}")
        print(f"Missing rounds: {missing if missing else 'none'}")
        print(f"Extra rounds: {extra if extra else 'none'}")

    print(f"Invalid/load errors: {invalid if invalid else 'none'}")
    print(f"Duplicate checkpoint hashes anywhere: {duplicate_hashes}")
    print(f"Consecutive identical checkpoints: {identical_consecutive if identical_consecutive else 'none'}")
    print(f"Near-zero consecutive deltas <= {args.min_delta:g}: {near_zero_delta if near_zero_delta else 'none'}")

    # Print compact trajectory table
    print("\nround | param_l2 | delta_l2_prev | rel_delta_prev | max_abs_delta | cosine_prev | identical")
    print("-" * 96)
    for r in loaded_rows:
        def fmt(v: Any, digits: int = 6) -> str:
            if v is None:
                return "-"
            if isinstance(v, bool):
                return str(v)
            try:
                return f"{float(v):.{digits}g}"
            except Exception:
                return str(v)

        print(
            f"{int(r['round']):5d} | "
            f"{fmt(r.get('param_l2')):>9} | "
            f"{fmt(r.get('delta_l2_from_prev')):>14} | "
            f"{fmt(r.get('relative_delta_l2_from_prev')):>14} | "
            f"{fmt(r.get('max_abs_delta_from_prev')):>13} | "
            f"{fmt(r.get('cosine_similarity_to_prev')):>11} | "
            f"{fmt(r.get('identical_to_prev')):>9}"
        )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {out_csv}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plot_rows = [
                r for r in loaded_rows
                if r.get("delta_l2_from_prev") is not None
            ]
            xs = [int(r["round"]) for r in plot_rows]
            ys = [float(r["delta_l2_from_prev"]) for r in plot_rows]

            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Round")
            plt.ylabel("L2 difference to previous checkpoint")
            plt.title("Checkpoint parameter change over rounds")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"Plot written: {args.plot}")
        except Exception as exc:
            print(f"WARNING: Could not create plot: {exc}")

    ok = not invalid and not missing and not identical_consecutive and not near_zero_delta
    print(f"\nRESULT: {'OK' if ok else 'CHECK WARNINGS ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
