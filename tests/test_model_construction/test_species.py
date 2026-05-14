"""Tests for species and trap creation in make_dynamic_mb_model.

Verifies that mobile species (D, T) and trapped species (trap1_D, trap1_T, ...)
are created with correct names and counts based on the number of traps
in the material definition.
"""
import pytest
import numpy as np
from hisp.festim_models.new_mb_model import make_dynamic_mb_model


@pytest.fixture
def simple_model(fake_bin, T_fn, flux_fn):
    """Build a model without running it."""
    model, quantities = make_dynamic_mb_model(
        bin=fake_bin,
        temperature=T_fn,
        deuterium_ion_flux=flux_fn,
        tritium_ion_flux=flux_fn,
        deuterium_atom_flux=flux_fn,
        tritium_atom_flux=flux_fn,
        final_time=100.0,
        folder="/tmp/test_species",
        exports=False,
        profile_export=False,
    )
    return model, quantities


class TestModelSpecies:
    def test_species_count(self, simple_model):
        """Total species = 2 mobile (D,T) + 2×N_traps (trapped D+T per trap)."""
        model, _ = simple_model
        n_traps = 1  # default FakeBin has 1 trap
        assert len(model.species) == 2 + 2 * n_traps

    def test_mobile_species_names(self, simple_model):
        """Mobile species are named 'D' and 'T'."""
        model, _ = simple_model
        mobile_names = [s.name for s in model.species if s.mobile]
        assert "D" in mobile_names
        assert "T" in mobile_names
        assert len(mobile_names) == 2

    def test_trapped_species_naming(self, simple_model):
        """Trapped species follow naming convention 'trap{i}_{isotope}'."""
        model, _ = simple_model
        trapped = [s for s in model.species if not s.mobile]
        names = {s.name for s in trapped}
        assert "trap1_D" in names
        assert "trap1_T" in names

    def test_multiple_traps(self, fake_bin, T_fn, flux_fn, fake_material_2traps):
        """With 2 traps: 2 mobile + 4 trapped = 6 species total."""
        fake_bin.material = fake_material_2traps
        model, _ = make_dynamic_mb_model(
            bin=fake_bin,
            temperature=T_fn,
            deuterium_ion_flux=flux_fn,
            tritium_ion_flux=flux_fn,
            deuterium_atom_flux=flux_fn,
            tritium_atom_flux=flux_fn,
            final_time=100.0,
            folder="/tmp/test_species_2traps",
            exports=False,
            profile_export=False,
        )
        assert len(model.species) == 6
        trapped_names = {s.name for s in model.species if not s.mobile}
        assert "trap1_D" in trapped_names
        assert "trap1_T" in trapped_names
        assert "trap2_D" in trapped_names
        assert "trap2_T" in trapped_names
