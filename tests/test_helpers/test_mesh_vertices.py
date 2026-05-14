"""Tests for graded_vertices mesh generation.

Verifies that graded_vertices produces a geometrically-graded 1D mesh
with correct start/end positions, monotonicity, and growth ratio.
"""
import numpy as np
from hisp.festim_models.new_mb_model import graded_vertices


class TestGradedVertices:
    def test_starts_at_zero(self):
        """Mesh always starts at x = 0 (plasma-facing surface)."""
        verts = graded_vertices(L=1e-3, h0=1e-10, r=1.05)
        assert verts[0] == 0.0

    def test_ends_at_L(self):
        """Mesh ends exactly at the specified thickness L."""
        L = 5e-4
        verts = graded_vertices(L=L, h0=1e-10, r=1.05)
        assert np.isclose(verts[-1], L)

    def test_monotonically_increasing(self):
        """All vertices are strictly increasing (no duplicates or reversals)."""
        verts = graded_vertices(L=1e-3, h0=1e-9, r=1.1)
        diffs = np.diff(verts)
        assert np.all(diffs > 0)

    def test_growth_ratio_approximately_r(self):
        """Consecutive cell sizes grow by the specified ratio r."""
        r = 1.08
        verts = graded_vertices(L=1e-3, h0=1e-10, r=r)
        diffs = np.diff(verts)
        # Check ratio between consecutive step sizes (skip last step which may be clipped)
        ratios = diffs[1:-1] / diffs[:-2]
        assert np.allclose(ratios, r, atol=1e-10)

    def test_small_domain(self):
        """Works correctly for very thin domains (micrometers)."""
        L = 1e-6
        verts = graded_vertices(L=L, h0=1e-10, r=1.05)
        assert verts[0] == 0.0
        assert np.isclose(verts[-1], L)
        assert len(verts) >= 2

    def test_h0_larger_than_L(self):
        """If initial cell size h0 >= L, mesh degenerates to [0, L] gracefully."""
        """If h0 >= L, mesh should still start at 0 and end at L."""
        L = 1e-10
        verts = graded_vertices(L=L, h0=1e-9, r=1.05)
        assert verts[0] == 0.0
        assert np.isclose(verts[-1], L)
