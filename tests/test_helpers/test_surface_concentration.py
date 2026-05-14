"""Tests for make_surface_concentration_time_function.

Verifies the analytical surface concentration BC formula c_s = φ·Rp/D
used for the 'Dirichlet - Analytical implantation approximation' BC,
including the optional recombination-limited correction term.
"""
import pytest
import numpy as np
from hisp.festim_models.new_mb_model import make_surface_concentration_time_function


def _constant_T(x, t):
    """Temperature function returning 500 K everywhere."""
    return np.full_like(x[0], 500.0)


class TestSurfaceConcentration:
    def test_zero_flux_returns_zero(self):
        """Zero incoming flux gives zero surface concentration."""
        c_S = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: 0.0,
            D0=4.1e-7,
            E_eV=0.39,
            R_p=3e-9,
        )
        assert c_S(1.0) == 0.0

    def test_positive_flux_gives_positive_concentration(self):
        """Non-zero flux produces a positive surface concentration."""
        c_S = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: 1e20,
            D0=4.1e-7,
            E_eV=0.39,
            R_p=3e-9,
        )
        val = c_S(1.0)
        assert val > 0.0

    def test_no_recombination_term_when_Kr0_is_None(self):
        """Without recombination (Kr0=None), c_s = φ·Rp / D(T) exactly."""
        flux = 1e20
        D0 = 4.1e-7
        E_eV = 0.39
        R_p = 3e-9
        T = 500.0

        c_S = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: flux,
            D0=D0,
            E_eV=E_eV,
            R_p=R_p,
            Kr0=None,
        )
        val = c_S(1.0)

        # Manual calculation
        kB_J = 1.380649e-23
        eV_to_J = 1.602176634e-19
        D_T = D0 * np.exp(-(E_eV * eV_to_J) / (kB_J * T))
        expected = (flux * R_p) / D_T
        assert val == pytest.approx(expected, rel=1e-10)

    def test_with_recombination_term(self):
        """Recombination-limited term (sqrt(φ/Kr)) adds to base concentration."""
        flux = 1e20
        D0 = 4.1e-7
        E_eV = 0.39
        R_p = 3e-9
        Kr0 = 7.94e-17
        E_Kr = -2.0

        c_S_no_kr = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: flux,
            D0=D0,
            E_eV=E_eV,
            R_p=R_p,
            Kr0=None,
        )

        c_S_with_kr = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: flux,
            D0=D0,
            E_eV=E_eV,
            R_p=R_p,
            flux_tot_fun=lambda t: flux,
            Kr0=Kr0,
            E_Kr=E_Kr,
        )

        # With recombination, concentration should be higher (extra term adds)
        assert c_S_with_kr(1.0) > c_S_no_kr(1.0)

    def test_zero_total_flux_with_Kr(self):
        """If total flux is zero, recombination term is skipped (avoids division by zero)."""
        c_S = make_surface_concentration_time_function(
            T_fun=_constant_T,
            flux_fun=lambda t: 1e20,
            D0=4.1e-7,
            E_eV=0.39,
            R_p=3e-9,
            flux_tot_fun=lambda t: 0.0,
            Kr0=7.94e-17,
            E_Kr=-2.0,
        )
        # Should still return finite positive value (just diffusion term)
        val = c_S(1.0)
        assert val > 0.0
        assert np.isfinite(val)
