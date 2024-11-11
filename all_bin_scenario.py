import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from hisp.plamsa_data_handling.main import (
    PlasmaDataHandling,
    read_wetted_data,
    compute_wetted_frac,
    find_length
)
from hisp.festim_models import make_W_mb_model, make_B_mb_model, make_DFW_mb_model

# from ITER_scenario import my_scenario
from hisp.helpers import periodic_step_function
from hisp.scenario import Scenario, Pulse
from fw_sub_bins import (
    sub_3_bins,
    dfw,
    fw_bins,
)
import pandas as pd

# import dolfinx

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

    fp = Pulse(
        pulse_type="FP",
        nb_pulses=1,
        ramp_up=10,
        steady_state=10,
        ramp_down=10,
        waiting=100,
    )
    icwc = Pulse(
        pulse_type="ICWC",
        nb_pulses=1,
        ramp_up=10,
        steady_state=10,
        ramp_down=10,
        waiting=100,
    )
    risp = Pulse(
        pulse_type="RISP",
        nb_pulses=1,
        ramp_up=10,
        steady_state=10,
        ramp_down=10,
        waiting=100,
    )

    my_scenario = Scenario(pulses=[fp])

    data_folder = "data"
    plasma_data_handling = PlasmaDataHandling(
        pulse_type_to_data={
            "FP": np.loadtxt(data_folder + "/Binned_Flux_Data.dat", skiprows=1),
            "ICWC": np.loadtxt(data_folder + "/ICWC_data.dat", skiprows=1),
            "GDC": np.loadtxt(data_folder + "/GDC_data.dat", skiprows=1),
        },
        path_to_ROSP_data=data_folder + "/ROSP_data",
        path_to_RISP_data=data_folder + "/RISP_data",
        path_to_RISP_wall_data=data_folder + "/RISP_Wall_data.dat",
    )

    def T_function(x: NDArray, t: float, frac=0) -> float:  # default shadowed
        """Monoblock temperature function.

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed monoblock temperature in K
        """
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        t_rel = t - my_scenario.get_time_start_current_pulse(t)

        if pulse.pulse_type == "BAKE":
            T_bake = 483.15  # K
            flat_top_value = np.full_like(x[0], T_bake)
        else:
            heat_total, heat_ion = plasma_data_handling.heat(
                pulse.pulse_type, nb_mb, t_rel
            )
            if nb_mb in fw_bins:
                frac = fw_frac
            heat_flux = heat_total - heat_ion + heat_ion * frac
            T_surface = 1.1e-4 * heat_flux + COOLANT_TEMP
            T_rear = 2.2e-5 * heat_flux + COOLANT_TEMP
            a = (T_rear - T_surface) / length
            b = T_surface
            flat_top_value = a * x[0] + b

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        return periodic_step_function(
            t_rel,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=np.full_like(x[0], COOLANT_TEMP),
        )

    def deuterium_ion_flux(t: float, frac=1) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        if nb_mb in fw_bins:
            frac = fw_frac

        ion_flux = frac * plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * (1 - tritium_fraction)
        resting_value = 0
        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    def tritium_ion_flux(t: float, frac=1) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        if nb_mb in fw_bins:
            frac = fw_frac

        ion_flux = frac * plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * tritium_fraction
        resting_value = 0.0

        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    def deuterium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        atom_flux = plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * (1 - tritium_fraction)
        resting_value = 0.0
        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    def tritium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        atom_flux = plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * tritium_fraction
        resting_value = 0.0
        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    ############# RUN FW BIN SIMUS #############
    # TODO: adjust to run monoblocks in parallel
    for nb_mb in fw_bins:
        wetted_data = read_wetted_data("Wetted_Frac_Bin_Data.csv", nb_mb=nb_mb)

        Slow = wetted_data[0]
        Stot = wetted_data[1]
        Shigh = wetted_data[2]
        f = wetted_data[3]

        section = "low_wetted"
        length, material = find_length(nb_mb, section)
        fw_frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, low_wet=True)
        print(nb_mb, material, section, length)

        if material == "W":
            my_model, quantities = make_W_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()
        elif material == "B":
            my_model, quantities = make_B_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

        section = "shadowed"
        length, material = find_length(nb_mb, section)
        fw_frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, shadowed=True)
        print(nb_mb, material, section, length)

        if material == "W":
            my_model, quantities = make_W_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()
        elif material == "B":
            my_model, quantities = make_B_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

        if nb_mb in sub_3_bins:
            section = "high_wetted"
            length, material = find_length(nb_mb, section)
            fw_frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, high_wet=True)
            print(nb_mb, material, section, length)

            my_model, quantities = make_W_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

        if nb_mb in dfw:
            section = "dfw"
            length, material = find_length(nb_mb, section)
            fw_frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, shadowed = True)
            print(nb_mb, material, section, length)

            my_model, quantities = make_DFW_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_{section}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

    ############# RUN DIV BIN SIMUS #############
    for nb_mb in list(range(19, 65)):
        length, material = find_length(nb_mb)
        print(nb_mb, material, length)

        if material == "W":
            my_model, quantities = make_W_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

        elif material == "B":
            my_model, quantities = make_B_mb_model(
                temperature=T_function,
                deuterium_ion_flux=deuterium_ion_flux,
                tritium_ion_flux=tritium_ion_flux,
                deuterium_atom_flux=deuterium_atom_flux,
                tritium_atom_flux=tritium_atom_flux,
                # FIXME: -1s here to avoid last time step spike
                final_time=my_scenario.get_maximum_time() - 1,
                L=length,
                folder=f"mb{nb_mb}_results",
            )
            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()

    ############# Results Plotting #############
    # TODO: add capability to add all inventories together and plot at the end
    # TODO: add a graph that computes grams 

    for name, quantity in quantities.items():
        plt.plot(quantity.t, quantity.data, label=name, marker="o")

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
