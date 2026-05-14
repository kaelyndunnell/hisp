"""Tests for model exports and quantities setup.

Verifies that make_dynamic_mb_model creates the correct TotalVolume
and SurfaceFlux export objects for all species, and that VTX/profile
exports are toggled correctly by the exports/profile_export flags.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import festim as F
from hisp.festim_models.new_mb_model import make_dynamic_mb_model
from conftest import FakeBin, FakeBinConfig, FakeMaterial


def _build_model(profile_export=True, exports=True):
    """Helper to build a model with given export settings."""
    bin_ = FakeBin()
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
        folder="/tmp/test_exports",
        exports=exports,
        profile_export=profile_export,
    )
    return model, quantities


class TestModelExports:
    def test_quantities_contain_all_species(self):
        """Quantities dict has keys for every species (D, T, trap1_D, trap1_T)."""
        _, quantities = _build_model()
        # Should contain keys for all species (D, T, trap1_D, trap1_T)
        assert "D" in quantities
        assert "T" in quantities
        assert "trap1_D" in quantities
        assert "trap1_T" in quantities

    def test_mobile_species_have_flux_entries(self):
        """Mobile species get inlet and outlet SurfaceFlux exports."""
        _, quantities = _build_model()
        assert "D_inlet_flux" in quantities
        assert "D_outlet_flux" in quantities
        assert "T_inlet_flux" in quantities
        assert "T_outlet_flux" in quantities

    def test_profile_export_present_when_true(self):
        """profile_export=True adds XDMFExport for each species."""
        _, quantities = _build_model(profile_export=True)
        assert "D_profile" in quantities
        assert "T_profile" in quantities
        assert "trap1_D_profile" in quantities
        assert "trap1_T_profile" in quantities

    def test_profile_export_absent_when_false(self):
        """profile_export=False omits profile export quantities."""
        _, quantities = _build_model(profile_export=False)
        assert "D_profile" not in quantities
        assert "T_profile" not in quantities

    def test_exports_false_no_vtx(self):
        """exports=False suppresses VTXSpeciesExport objects."""
        model, _ = _build_model(exports=False)
        vtx_exports = [e for e in model.exports if isinstance(e, F.VTXSpeciesExport)]
        assert len(vtx_exports) == 0

    def test_exports_true_has_vtx(self):
        """exports=True creates at least one VTXSpeciesExport."""
        model, _ = _build_model(exports=True)
        vtx_exports = [e for e in model.exports if isinstance(e, F.VTXSpeciesExport)]
        assert len(vtx_exports) >= 1
