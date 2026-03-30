#!/usr/bin/env python3
"""
Unit tests for online (on-the-fly) low-rank decomposition.

These tests validate the decomposition route used by `extras/decompose.py`
when factors are *not* precomputed (i.e. direct `decompose_weight(...)`).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

try:
    from lite_linear.ffn_delta import decompose_weight, r_sqrt_and_inv
except ModuleNotFoundError:
    # Allow direct execution from a source checkout without installation.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from lite_linear.ffn_delta import decompose_weight, r_sqrt_and_inv


class TestOnlineDecomposition(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def test_reconstruction_without_r(self) -> None:
        d_out, d_in, rank = 64, 48, 16
        W = torch.randn(d_out, d_in, dtype=torch.float32)

        A, B, Q = decompose_weight(W, rank=rank)
        W_recon = A @ B + Q

        self.assertEqual(A.shape, (d_out, rank))
        self.assertEqual(B.shape, (rank, d_in))
        self.assertEqual(Q.shape, (d_out, d_in))
        self.assertTrue(torch.allclose(W, W_recon, atol=1e-5, rtol=1e-5))

    def test_reconstruction_with_r(self) -> None:
        d_out, d_in, rank = 80, 80, 20
        W = torch.randn(d_out, d_in, dtype=torch.float32)

        # Construct SPD calibration matrix R for the "with-R" online path.
        M = torch.randn(d_in, d_in, dtype=torch.float32)
        R = (M.transpose(0, 1) @ M) / float(d_in) + 1e-3 * torch.eye(
            d_in, dtype=torch.float32
        )
        r_sqrt, r_sqrt_inv = r_sqrt_and_inv(R)

        A, B, Q = decompose_weight(W, rank=rank, r_sqrt=r_sqrt, r_sqrt_inv=r_sqrt_inv)
        W_recon = A @ B + Q

        self.assertEqual(A.shape, (d_out, rank))
        self.assertEqual(B.shape, (rank, d_in))
        self.assertEqual(Q.shape, (d_out, d_in))
        self.assertTrue(torch.allclose(W, W_recon, atol=1e-4, rtol=1e-4))

    def test_requires_both_rsqrt_and_inverse(self) -> None:
        W = torch.randn(16, 16, dtype=torch.float32)
        R = torch.eye(16, dtype=torch.float32)
        r_sqrt, _ = r_sqrt_and_inv(R)

        with self.assertRaises(ValueError):
            _ = decompose_weight(W, rank=8, r_sqrt=r_sqrt, r_sqrt_inv=None)


if __name__ == "__main__":
    unittest.main()
