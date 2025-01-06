from hisp.festim_models.mb_model import (
    make_W_mb_model,
    make_B_mb_model,
    make_DFW_mb_model,
)

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
        L=6e-3,
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
        L=6e-3,
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
        L=6e-3,
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


def test_model_last_timestep_overshoot():

    from hisp.scenario import Scenario, Pulse

    pulse1 = Pulse(
        pulse_type="FP",
        nb_pulses=1,
        ramp_up=0,
        steady_state=50,
        ramp_down=0,
        waiting=0,
        tritium_fraction=0.5,
    )

    scenario = Scenario([pulse1])

    def fun(t):
        scenario.get_pulse(t)
        return None

    (my_model, quantities) = make_DFW_mb_model(
        temperature=1000,
        deuterium_ion_flux=lambda _: 0,
        deuterium_atom_flux=lambda _: 0,
        tritium_ion_flux=lambda _: 0,
        tritium_atom_flux=lambda _: 0,
        final_time=50,
        L=6e-3,
        folder=".",
    )

    my_model.settings.stepsize.growth_factor = 1.2
    my_model.settings.stepsize.cutback_factor = 0.9
    my_model.settings.stepsize.target_nb_iterations = 4
    my_model.settings.stepsize.max_stepsize = fun

    my_model.initialise()

    my_model.t.value = 49
    my_model.settings.stepsize.value = 20
    my_model.dt.value = 20

    my_model.show_progress_bar = False

    my_model.iterate()
