# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global HM shares -> exact integer quotas with protected floors and available caps."""

import numpy as np


def allocate_budget(shares: np.ndarray, available: np.ndarray, protected: np.ndarray, total: int) -> np.ndarray:
    shares = np.asarray(shares, dtype=np.float64)
    available = np.asarray(available, dtype=np.int64)
    protected = np.asarray(protected, dtype=np.int64)
    if shares.shape != available.shape or shares.shape != protected.shape:
        raise ValueError("Shares, availability and protection must have the same layer/head shape.")
    if not np.isfinite(shares).all() or np.any(shares <= 0):
        raise ValueError("HM shares must be finite and strictly positive.")
    if np.any(protected < 0) or np.any(available < protected):
        raise ValueError("Invalid protected floors or available lengths.")
    target = min(int(total), int(available.sum()))
    if target < protected.sum():
        raise ValueError("The total budget cannot accommodate all protected entries.")
    if target == available.sum():
        return available.copy()
    if target == protected.sum():
        return protected.copy()
    shares = shares / shares.sum()
    lo, hi = 0.0, float(np.max(available / shares))
    for _ in range(80):
        scale = (lo + hi) / 2
        if np.clip(scale * shares, protected, available).sum() < target:
            lo = scale
        else:
            hi = scale
    ideal = np.clip(hi * shares, protected, available)
    quota = np.floor(ideal).astype(np.int64)
    remainder = target - int(quota.sum())
    fractions = np.where(quota < available, ideal - quota, -np.inf)
    order = np.argsort(-fractions.ravel(), kind="stable")
    if remainder < 0 or remainder > np.count_nonzero(quota < available):
        raise RuntimeError("Failed to round head quotas to the exact global target.")
    quota.ravel()[order[:remainder]] += 1
    return quota
