"""Tests for boundary condition setup in make_dynamic_mb_model.

Verifies that each combination of plasma-facing and rear-surface BC
options produces the correct FESTIM BC objects (FixedConcentrationBC,
SurfaceReactionBC) with correct parameters on the correct surfaces.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import festim as F
from hisp.festim_models.new_mb_model import make_dynamic_mb_model
from conftest import FakeBin, FakeBinConfig, FakeMaterial, FakeTrap


def _build_model(bc_plasma_facing, bc_rear, material=None):
    """Helper to build a model with specified BCs."""
    mat = material or FakeMaterial()
    config = FakeBinConfig(
        bc_plasma_facing_surface=bc_plasma_facing,
        bc_rear_surface=bc_rear,
    )
    bin_ = FakeBin(material=mat, bin_configuration=config)

    T_fn = lambda x, t: np.full_like(x[0], 500.0)
    flux_fn = lambda t: 1e20

    model, quantities = make_dynamic_mb_model(
        bin=bin_,
        temperature=T_fn,
        deuterium_ion_flux=flux_fn,
        tritium_ion_flux=flux_fn,
        deuterium_atom_flux=flux_fn,
        tritium_atom_flux=flux_fn,
        final_time=100.0,
        folder="/tmp/test_bcs",
        exports=False,
        profile_export=False,
    )
    return model, quantities


class TestPlasmaFacingBC:
    def test_robin_implantation(self):
        """Robin PFS: 3 SurfaceReactionBC (DD, DT, TT) + 4 volumetric sources."""
        model, _ = _build_model(
            "Robin - Surf. Rec. + Implantation",
            "Dirichlet - 0 concentration",
        )
        # Should have SurfaceReactionBC objects in boundary_conditions
        srbc = [bc for bc in model.boundary_conditions if isinstance(bc, F.SurfaceReactionBC)]
        assert len(srbc) == 3  # DD, DT, TT
        # Should have volumetric sources
        assert hasattr(model, 'sources') and len(model.sources) == 4

    def test_dirichlet_zero_implantation(self):
        """Dirichlet PFS: c=0 at surface + volumetric implantation sources."""
        model, _ = _build_model(
            "Dirichlet - 0 concentration + Implantation",
            "Dirichlet - 0 concentration",
        )
        # Should have FixedConcentrationBC at inlet with value 0
        fixed_bcs = [bc for bc in model.boundary_conditions if isinstance(bc, F.FixedConcentrationBC)]
        # At least 2 for inlet (D, T) + 2 for outlet
        inlet_zeros = [bc for bc in fixed_bcs if bc.value == 0.0]
        assert len(inlet_zeros) >= 2
        # Should have volumetric sources
        assert hasattr(model, 'sources') and len(model.sources) == 4

    def test_dirichlet_analytical(self):
        """Analytical PFS: time-dependent c_s(t) Dirichlet BCs, no vol. sources."""
        model, _ = _build_model(
            "Dirichlet - Analyttical implantation approximation",
            "Dirichlet - 0 concentration",
        )
        # Should have FixedConcentrationBC with callable values (not zero)
        fixed_bcs = [bc for bc in model.boundary_conditions if isinstance(bc, F.FixedConcentrationBC)]
        # Should have at least 2 for inlet + 2 for outlet = 4
        assert len(fixed_bcs) >= 4
        # Inlet BCs should have callable values (time-dependent concentration)
        callable_bcs = [bc for bc in fixed_bcs if callable(bc.value)]
        assert len(callable_bcs) >= 2

    def test_unsupported_plasma_facing_raises(self):
        """Unsupported PFS BC string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported plasma-facing BC"):
            _build_model("Invalid BC", "Dirichlet - 0 concentration")


class TestRearBC:
    def test_dirichlet_zero(self):
        """Dirichlet rear: c=0 FixedConcentrationBC at rear surface."""
        model, _ = _build_model(
            "Dirichlet - 0 concentration + Implantation",
            "Dirichlet - 0 concentration",
        )
        fixed_bcs = [bc for bc in model.boundary_conditions if isinstance(bc, F.FixedConcentrationBC)]
        # Should include zeros at outlet
        assert len(fixed_bcs) >= 2

    def test_neumann_no_flux(self):
        """Neumann rear: natural BC (no explicit BC added), model builds OK."""
        model, _ = _build_model(
            "Dirichlet - 0 concentration + Implantation",
            "Neumann - no flux",
        )
        # The Neumann BC currently extends with empty list, so no additional BCs
        # compared to just inlet BCs
        # Just verify model builds without error
        assert model is not None

    def test_robin_surf_rec_rear(self):
        """Robin rear: 3 SurfaceReactionBC (DD,DT,TT) at outlet with material K_R/E_R."""
        mat = FakeMaterial(K_R=1.5e-16, E_R=-1.8)
        model, _ = _build_model(
            "Dirichlet - 0 concentration + Implantation",
            "Robin - Surf. Rec.",
            material=mat,
        )
        srbc = [bc for bc in model.boundary_conditions if isinstance(bc, F.SurfaceReactionBC)]
        assert len(srbc) == 3

        # Verify subdomain is outlet (x = L)
        for bc in srbc:
            assert bc.subdomain.x == model.mesh.vertices[-1]

        # Verify reactant pairs
        reactant_pairs = []
        for bc in srbc:
            names = sorted([r.name for r in bc.reactant])
            reactant_pairs.append(tuple(names))
        assert ("D", "D") in reactant_pairs
        assert ("T", "T") in reactant_pairs
        assert ("D", "T") in reactant_pairs

        # Verify k_r0 and E_kr read from material
        for bc in srbc:
            assert bc.k_r0 == 1.5e-16
            assert bc.E_kr == -1.8

    def test_robin_surf_rec_rear_defaults(self):
        """Robin rear without K_R/E_R in material falls back to W defaults."""
        mat = FakeMaterial()
        # Remove K_R to trigger getattr default
        del mat.K_R
        del mat.E_R
        model, _ = _build_model(
            "Dirichlet - 0 concentration + Implantation",
            "Robin - Surf. Rec.",
            material=mat,
        )
        srbc = [bc for bc in model.boundary_conditions if isinstance(bc, F.SurfaceReactionBC)]
        for bc in srbc:
            assert bc.k_r0 == 7.94e-17
            assert bc.E_kr == -2.0

    def test_unsupported_rear_raises(self):
        """Unsupported rear BC string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported rear BC"):
            _build_model("Dirichlet - 0 concentration + Implantation", "Invalid rear BC")
