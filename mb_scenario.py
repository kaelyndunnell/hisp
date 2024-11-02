# simple monoblock simulation in festim
import numpy as np
import matplotlib.pyplot as plt

from dolfinx.fem.function import Constant
from numpy.typing import NDArray

from hisp.helpers import Scenario
from hisp import make_mb_model
from hisp.dina import get_particle_flux, heat

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

NB_FP_PULSES_PER_DAY = 13
COOLANT_TEMP = 343  # 70 degree C cooling water

# tritium fraction = T/D
PULSE_TYPE_TO_TRITIUM_FRACTION = {
    "FP": 0.5,
    "ICWC": 0,
    "RISP": 0,
    "GDC": 0,
    "BAKE": 0,
}

if __name__ == "__main__":
    my_scenario = Scenario("scenario_test.txt")

    nb_mb = 64
    L = 6e-3  # m

    def T_function(x: NDArray, t: Constant) -> float:
        """Monoblock temperature function.

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed monoblock temperature in K
        """
        resting_value = np.full_like(x[0], COOLANT_TEMP)
        pulse_row = my_scenario.get_row(float(t))
        pulse_type = my_scenario.get_pulse_type(float(t))
        t_rel = t - my_scenario.get_time_till_row(pulse_row)

        if pulse_type == "BAKE":
            T_bake = 483.15  # K
            flat_top_value = np.full_like(x[0], T_bake)
        else:
            heat_flux = heat(pulse_type, nb_mb, t_rel)
            T_surface = 1.1e-4 * heat_flux + COOLANT_TEMP
            T_rear = 2.2e-5 * heat_flux + COOLANT_TEMP
            a = (T_rear - T_surface) / L
            b = T_surface
            flat_top_value = a * x[0] + b

        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)

        relative_time = t - time_start_current_pulse
        return (
            flat_top_value
            if relative_time % total_time_pulse < total_time_on
            and relative_time % total_time_pulse != 0.0
            else resting_value
        )

    def deuterium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        ion_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * (1 - tritium_fraction)
        resting_value = 0

        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def tritium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        ion_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * tritium_fraction
        resting_value = 0.0

        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def deuterium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse
        atom_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * (1 - tritium_fraction)
        resting_value = 0.0
        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def tritium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        atom_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * tritium_fraction
        resting_value = 0.0
        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    my_model, quantities = make_mb_model(
        temperature=T_function,
        deuterium_ion_flux=deuterium_ion_flux,
        tritium_ion_flux=tritium_ion_flux,
        deuterium_atom_flux=deuterium_atom_flux,
        tritium_atom_flux=tritium_atom_flux,
        final_time=my_scenario.get_maximum_time(),
        L=6e-3,
        folder=f"mb{nb_mb}_results",
    )
    ############# Run Simu #############

    my_model.initialise()
    my_model.run()
    my_model.progress_bar.close()

    ############# Results Plotting #############

    for name, quantity in quantities.items():
        plt.plot(quantity.t, quantity.data, label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.yscale("log")

    plt.show()

    fig, ax = plt.subplots()

    ax.stackplot(
        quantity.t,
        [quantity.data for quantity in quantities.values()],
        labels=quantities.keys(),
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.show()
