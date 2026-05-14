"""Integration tests using real Bin and Material objects from PFC-Tritium-Transport.

Verifies that make_dynamic_mb_model works with real PFC-TT classes
(Bin, Material, BinConfiguration, Trap) rather than fake/mock objects.
"""
import pytest
import numpy as np
import tempfile

# Trigger PFC-TT path resolution so PFC-TT packages become importable
import hisp.bin  # noqa: F401

from bins_from_csv.csv_bin import Bin, BinConfiguration
from materials.materials import Material, Trap
from hisp.festim_models.new_mb_model import make_dynamic_mb_model


def _make_real_bin():
    """Build a real PFC-TT Bin with a W material and 1 trap."""
    material = Material(
        name="W",
        Mat_density=6.3e28,
        D0=4.1e-7,
        E_D=0.39,
        K_R=7.94e-17,
        E_R=-2.0,
        N_traps=1,
        traps=[Trap(Trap_density=1e-3, k_0=8.96e-17, E_k=0.2, p_0=1e13, E_p=0.87)],
    )
    config = BinConfiguration(
        rtol=1e-10,
        atol=1e10,
        fp_max_stepsize=5.0,
        max_stepsize_no_fp=100.0,
        bc_plasma_facing_surface="Dirichlet - 0 concentration + Implantation",
        bc_rear_surface="Dirichlet - 0 concentration",
    )
    return Bin(
        flux_id=0,
        material=material,
        thickness=6e-3,
        cu_thickness=2e-3,
        mode="high_wetted",
        parent_bin_surf_area=26.0,
        surface_area=0.886,
        f_ion_flux_fraction=0.919,
        location="FW",
        bin_configuration=config,
        calculate_implantation_params=False,
    )


@pytest.mark.integration
class TestWithRealBin:
    """Integration tests using real PFC-TT Bin/Material objects."""

    def test_real_bin_has_required_attributes(self):
        """PFC-TT Bin exposes the attributes make_dynamic_mb_model needs."""
        bin_ = _make_real_bin()
        assert hasattr(bin_, "material")
        assert hasattr(bin_, "thickness")
        assert hasattr(bin_, "bin_configuration")
        assert hasattr(bin_, "bin_number")
        assert bin_.material.N_traps == 1

    def test_real_bin_model_construction(self):
        """Full model builds from a real PFC-TT Bin without crashing."""
        bin_ = _make_real_bin()
        T_fn = lambda x, t: np.full_like(x[0], 500.0)
        flux_fn = lambda t: 1e20

        with tempfile.TemporaryDirectory() as tmpdir:
            model, quantities = make_dynamic_mb_model(
                bin=bin_,
                temperature=T_fn,
                deuterium_ion_flux=flux_fn,
                tritium_ion_flux=flux_fn,
                deuterium_atom_flux=flux_fn,
                tritium_atom_flux=flux_fn,
                final_time=100.0,
                folder=tmpdir,
                exports=False,
                profile_export=False,
            )

        assert model is not None
        assert len(model.species) >= 2
        assert "D" in quantities
        assert "T" in quantities

    def test_real_bin_model_initialises(self):
        """Model initialises and its internals reflect the bin properties."""
        bin_ = _make_real_bin()
        T_fn = lambda x, t: np.full_like(x[0], 500.0)
        flux_fn = lambda t: 1e20

        with tempfile.TemporaryDirectory() as tmpdir:
            model, _ = make_dynamic_mb_model(
                bin=bin_,
                temperature=T_fn,
                deuterium_ion_flux=flux_fn,
                tritium_ion_flux=flux_fn,
                deuterium_atom_flux=flux_fn,
                tritium_atom_flux=flux_fn,
                final_time=100.0,
                folder=tmpdir,
                exports=False,
                profile_export=False,
            )

            model.initialise()

            # Mesh spans [0, bin.thickness]
            assert model.mesh.vertices[0] == 0.0
            assert np.isclose(model.mesh.vertices[-1], bin_.thickness)

            # Material diffusivity matches bin material
            vol = model.subdomains[0]
            assert vol.material.D_0 == bin_.material.D0
            assert vol.material.E_D == bin_.material.E_D

            # Species count: 2 mobile + 2 × N_traps
            n_traps = bin_.material.N_traps
            assert len(model.species) == 2 + 2 * n_traps

            # Solver tolerances match bin configuration
            assert model.settings.rtol == bin_.bin_configuration.rtol
            assert model.settings.atol == bin_.bin_configuration.atol
