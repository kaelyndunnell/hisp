from hisp.festim_models.mb_model import make_W_mb_model, make_B_mb_model, make_DFW_mb_model

import festim as F

import pytest


@pytest.mark.parametrize("temp", [400, lambda t: 400 + 1, lambda x, t: 400 + t - x[0]])
def test_mb_model(temp):
    """Builds a festim tungsten model, run it, and tests the output."""
    (mb_model, quantities) = make_W_mb_model(
        temperature=temp,
        deuterium_ion_flux=lambda _: 1e22,
        deuterium_atom_flux=lambda _: 1e22,
        tritium_ion_flux=lambda _: 1e22,
        tritium_atom_flux=lambda _: 1e22,
        final_time=50,
        folder=".",
    )

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, F.TotalVolume)
        assert len(value.data) > 0


@pytest.mark.parametrize("temp", [400, lambda t: 400 + 1, lambda x, t: 400 + t - x[0]])
def test_mb_model(temp):
    """Builds a festim boron model, run it, and tests the output."""
    (mb_model, quantities) = make_B_mb_model(
        temperature=temp,
        deuterium_ion_flux=lambda _: 1e22,
        deuterium_atom_flux=lambda _: 1e22,
        tritium_ion_flux=lambda _: 1e22,
        tritium_atom_flux=lambda _: 1e22,
        final_time=50,
        folder=".",
    )

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, F.TotalVolume)
        assert len(value.data) > 0

@pytest.mark.parametrize("temp", [400, lambda t: 400 + 1, lambda x, t: 400 + t - x[0]])
def test_mb_model(temp):
    """Builds a festim tungsten model, run it, and tests the output."""
    (mb_model, quantities) = make_DFW_mb_model(
        temperature=temp,
        deuterium_ion_flux=lambda _: 1e22,
        deuterium_atom_flux=lambda _: 1e22,
        tritium_ion_flux=lambda _: 1e22,
        tritium_atom_flux=lambda _: 1e22,
        final_time=50,
        folder=".",
    )

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, F.TotalVolume)
        assert len(value.data) > 0