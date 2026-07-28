"""Fixed quantization grids, expressed in the normalized space of blocked.py
(a block's max-abs element maps to ~±SCALE_REF = 6).

E2M1 is the hardware FP4 grid. The 3-bit grids are MSE-optimal look-up tables fit
on a Gaussian in that same normalized space (see ../../grids.ipynb):

  relative MSE on N(0,1), blocks of 16
    NVFP4 (E2M1, 4-bit)                    0.0091
    FP4-pair centers of mass (3-bit)       0.0334   signed
    Lloyd / k-means (3-bit)                0.0218   signed   <- best 3-bit
"""

import torch

# ---------------------------------------------------------------------------
# E2M1 — the 4-bit hardware FP4 grid
# ---------------------------------------------------------------------------
# Positive magnitudes and the midpoints between them (round-to-nearest bounds).
E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
E2M1_MAX = 6.0

# Full signed grid, ascending. Index == the 4-bit code is NOT true here (that is
# handled by the packing code in nvfp4.py); this is for generic nearest-rounding.
FP4_GRID = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
     +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0]
)

# Decode table indexed by the 4-bit E2M1 code (sign bit 3, magnitude bits 0-2).
E2M1_DECODE = torch.tensor(E2M1_LEVELS + [-v for v in E2M1_LEVELS])

# ---------------------------------------------------------------------------
# 3-bit look-up tables (8 levels) for the SIGNED normalization, where the scale
# absorbs the sign of the block's max-abs element so the extreme is always +6.
# Both are asymmetric for exactly that reason.
# ---------------------------------------------------------------------------
# Centers of mass of adjacent FP4 pairs (a pure downcast of an FP4 index >> 1).
FP4_DOWNCAST_SIGNED_3BIT = torch.tensor(
    [-4.506, -2.550, -1.236, -0.372, +0.373, +1.236, +2.550, +5.120]
)

# Lloyd's algorithm (1-D k-means) optimum — the best 3-bit grid of the three.
LLOYD_SIGNED_3BIT = torch.tensor(
    [-4.795, -3.062, -1.631, -0.339, +0.927, +2.265, +3.797, +5.788]
)


def grid_spacing(grid: torch.Tensor) -> float:
    """Mean gap between adjacent levels of a sorted grid.

    Used to make grid-proximity logit initialisation scale-free: distances are
    measured in units of the grid's own spacing, so the same (std, strength)
    hyperparameters give the same softness on a ±4 integer grid as on the ±6
    Lloyd grid. Returns 1.0 for a unit-spaced integer grid, so the symmetric
    uniform GSQ grids behave exactly as before.
    """
    return float(grid.float().diff().mean())


def lloyd_grid(x: torch.Tensor, n_levels: int = 8, n_iters: int = 200) -> torch.Tensor:
    """Fit an MSE-optimal scalar grid with Lloyd's algorithm (1-D k-means).

    Used to derive the constants above; kept so a grid can be re-fit on a real
    weight distribution instead of a Gaussian.
    """
    from .blocked import index_nearest

    x_flat = x.flatten()
    qs = torch.linspace(0, 1, n_levels + 2)[1:-1]
    # init from quantiles of a subsample (torch.quantile has a 2**24 size limit)
    sample = x_flat[torch.randperm(x_flat.numel())[:1_000_000]]
    centers = torch.quantile(sample, qs.to(x_flat.dtype))
    for _ in range(n_iters):
        inds = index_nearest(x_flat, centers)
        sums = torch.bincount(inds, weights=x_flat, minlength=n_levels).double()
        counts = torch.bincount(inds, minlength=n_levels).double()
        # Keep empty clusters at their previous center: zeroing them would break the
        # sorted-grid invariant that index_nearest (bucketize) relies on.
        new_centers = centers.double().clone()
        nonempty = counts > 0
        new_centers[nonempty] = sums[nonempty] / counts[nonempty]
        new_centers = new_centers.to(centers.dtype)
        if torch.allclose(new_centers, centers, atol=1e-7):
            return new_centers
        centers = new_centers
    return centers
