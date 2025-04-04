from hisp.festim_models.mb_model import (
    make_W_mb_model,
    make_B_mb_model,
    make_DFW_mb_model,
)

from hisp.settings import CustomSettings
import festim as F
import pytest


@pytest.mark.parametrize(
    "temp", [1000, lambda t: 1000 + 1, lambda x, t: 1000 + t - x[0]]
)
def test_mb_W_model(temp):
    """Builds a festim tungsten model, run it, and tests the output."""
    (mb_model, quantities) = make_W_mb_model(
        temperature=1000,
        deuterium_ion_flux=lambda _: 2e10,
        deuterium_atom_flux=lambda _: 2e10,
        tritium_ion_flux=lambda _: 2e10,
        tritium_atom_flux=lambda _: 2e10,
        L=6e-3,
        final_time=50,
        folder=".",
    )
    mb_model.settings.stepsize.initial_value = 1

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, (F.TotalVolume, F.SurfaceQuantity, F.SurfaceTemperature)) # to change with SurfaceTemp PR merged
        assert len(value.data) > 0


@pytest.mark.parametrize(
    "temp", [1000, lambda t: 1000 + 1, lambda x, t: 1000 + t - x[0]]
)
def test_mb_model_B(temp):
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
    mb_model.settings.stepsize.initial_value = 1

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, (F.TotalVolume, F.SurfaceQuantity, F.SurfaceTemperature)) # to change with SurfaceTemp PR merged
        assert len(value.data) > 0


@pytest.mark.parametrize(
    "temp", [1000, lambda t: 1000 + 1, lambda x, t: 1000 + t - x[0]]
)
def test_mb_model_DFW(temp):
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

    mb_model.settings.stepsize.initial_value = 1

    mb_model.initialise()
    mb_model.run()

    # TEST
    assert isinstance(quantities, dict)
    for key, value in quantities.items():
        assert isinstance(key, str)
        assert isinstance(value, (F.TotalVolume, F.SurfaceQuantity, F.SurfaceTemperature)) # to change with SurfaceTemp PR merged
        assert len(value.data) > 0


def test_model_last_timestep_overshoot():
    """
    Test to check that a simulation can run even with max_stepsize calling "get_pulse"
    at a timestep that is greater than the maximum time in the scenario.

    See PR #77 for more details.
    """

    # Define a scenario with a single pulse
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

    # Define a function that will be called by the model to get the pulse at each timestep
    def fun(t):
        scenario.get_pulse(t)
        return None

    # Create a model
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

    my_model.show_progress_bar = False  # Disable the progress bar

    # give the stepsize adaptivity settings
    my_model.settings.stepsize.growth_factor = 1.2
    my_model.settings.stepsize.cutback_factor = 0.9
    my_model.settings.stepsize.target_nb_iterations = 4

    # cap the stepsize with the function we created
    my_model.settings.stepsize.max_stepsize = fun

    # Initialise the model
    my_model.initialise()

    # Artificially set the next timestep to be greater than the final time
    my_model.t.value = 49
    my_model.settings.stepsize.value = 20
    my_model.dt.value = 20

    my_model.iterate()



@pytest.mark.parametrize(
        "rtol", [1e-10, lambda t: 1e-8 if t<10 else 1e-10]
        )
def test_callable_rtol(rtol): 
    """Builds B model to test custom rtol."""
    (my_model, quantities) = make_B_mb_model(
        temperature=500,
        deuterium_ion_flux=lambda _: 1e22,
        deuterium_atom_flux=lambda _: 1e22,
        tritium_ion_flux=lambda _: 1e22,
        tritium_atom_flux=lambda _: 1e22,
        final_time=50,
        L=6e-3,
        custom_rtol=rtol,
        folder=".",
    )

    # initialise the model
    my_model.initialise()

    assert my_model.settings.rtol == rtol

@pytest.mark.parametrize(
        "atol", [1e10, lambda t: 1e12 if t<10 else 1e10]
        )
def test_callable_atol(atol): 
    """Builds B model to test custom rtol."""
    (my_model, quantities) = make_B_mb_model(
        temperature=500,
        deuterium_ion_flux=lambda _: 1e22,
        deuterium_atom_flux=lambda _: 1e22,
        tritium_ion_flux=lambda _: 1e22,
        tritium_atom_flux=lambda _: 1e22,
        final_time=50,
        L=6e-3,
        custom_atol=atol,
        custom_rtol=1e-10,
        folder=".",
    )

    # initialise the model
    my_model.initialise()

    assert my_model.settings.atol == atol
