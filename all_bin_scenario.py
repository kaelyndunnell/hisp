import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.festim_models import make_W_mb_model, make_B_mb_model, make_DFW_mb_model
from make_iter_bins import FW_bins, Div_bins, total_fw_bins, total_nb_bins

from hisp.helpers import periodic_step_function
from hisp.scenario import Scenario, Pulse
from hisp.bin import SubBin, DivBin

# import dolfinx

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

NB_FP_PULSES_PER_DAY = 13
COOLANT_TEMP = 343  # 70 degree C cooling water

if __name__ == "__main__":

    fp = Pulse(
        pulse_type="FP",
        nb_pulses=1,
        ramp_up=10,
        steady_state=10,
        ramp_down=10,
        waiting=100,
        tritium_fraction=0.5,
        tritium_fraction=0.5,
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

    ############# CREATE EMPTY NP ARRAYS TO STORE ALL DATA #############
    global_data = {}

    def T_function(x: NDArray, t: float) -> float:
        """W Monoblock temperature function.

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed W monoblock temperature in K
        """
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        t_rel = t - my_scenario.get_time_start_current_pulse(t)

        if pulse.pulse_type == "BAKE":
            T_bake = 483.15  # K
            flat_top_value = np.full_like(x[0], T_bake)
        else:
            heat_flux = plasma_data_handling.get_heat(pulse.pulse_type, sub_bin, t_rel)
            T_surface = 1.1e-4 * heat_flux + COOLANT_TEMP
            T_rear = 2.2e-5 * heat_flux + COOLANT_TEMP
            a = (T_rear - T_surface) / sub_bin.thickness
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

    def T_function_B(x: NDArray, t: float) -> float:
        """W Monoblock temperature function.

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed W monoblock temperature in K
        """
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        t_rel = t - my_scenario.get_time_start_current_pulse(t)

        if pulse.pulse_type == "BAKE":
            T_bake = 483.15  # K
            flat_top_value = np.full_like(x[0], T_bake)
        else:
            heat_flux = plasma_data_handling.get_heat(pulse.pulse_type, sub_bin, t_rel)
            T_rear_tungsten = (
                2.2e-5 * heat_flux + COOLANT_TEMP
            )  # boron layers based off of rear temp of W mbs
            flat_top_value = np.full_like(x[0], 5e-4 * heat_flux + T_rear_tungsten)
            T_rear = 2.2e-5 * heat_flux + COOLANT_TEMP # rear temp for tungsten, used in all calculations for now
            if sub_bin.material == "W":
                T_surface = 1.1e-4 * heat_flux + COOLANT_TEMP
                a = (T_rear - T_surface) / sub_bin.thickness
                b = T_surface
                flat_top_value = a * x[0] + b
            else: # currently both B and DFW, later separate these when get DFW data 
                flat_top_value = np.full_like(x[0], 5e-4 * heat_flux + T_rear) # boron layers based off of rear temp of W mbs

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        return periodic_step_function(
            t_rel,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=np.full_like(x[0], COOLANT_TEMP),
        )

    def deuterium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        ion_flux = plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, bin=sub_bin, t_rel=relative_time, ion=True
        )
        tritium_fraction = pulse.tritium_fraction
        flat_top_value = ion_flux * (1 - tritium_fraction)
        resting_value = 0
        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    def tritium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration
        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        ion_flux = plasma_data_handling.get_particle_flux(
            pulse_type=pulse_type, bin=sub_bin, t_rel=relative_time, ion=True
        )

        tritium_fraction = pulse.tritium_fraction
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
            pulse_type=pulse_type, bin=sub_bin, t_rel=relative_time, ion=False
        )

        tritium_fraction = pulse.tritium_fraction
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
            pulse_type=pulse_type, bin=sub_bin, t_rel=relative_time, ion=False
        )
        tritium_fraction = pulse.tritium_fraction
        flat_top_value = atom_flux * tritium_fraction
        resting_value = 0.0
        return periodic_step_function(
            relative_time,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=flat_top_value,
            value_off=resting_value,
        )

    def max_stepsize(t: float) -> float:
        pulse = my_scenario.get_pulse(t)
        relative_time = t - my_scenario.get_time_start_current_pulse(t)
        return periodic_step_function(
            relative_time,
            period_on=pulse.duration_no_waiting,
            period_total=pulse.total_duration,
            value=pulse.duration_no_waiting / 10,
            value_off=None,
        )

    def which_model(subbin: SubBin | DivBin):
        """Returns the correct model for the subbin.

        Args:
            subbin: The bin/subbin to get the model for

        Returns:
            festim.HTransportModel, dict: The model and the quantities to plot
        """
    def ion_implantation_range(t: float) -> float:
        # assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        incident_energy = plasma_data_handling.get_incident_energy(
            pulse_type=pulse_type,
            bin=sub_bin,
            t_rel=relative_time,
        )
        # TODO add reference pls
        implantation_range = 1.9e-10 * incident_energy**0.59  # m
        return implantation_range

    def atom_implantation_range(t: float) -> float:
        # assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse = my_scenario.get_pulse(t)
        pulse_type = pulse.pulse_type

        time_start_current_pulse = my_scenario.get_time_start_current_pulse(t)
        relative_time = t - time_start_current_pulse

        incident_energy = plasma_data_handling.get_incident_energy(
            pulse_type=pulse_type, bin=sub_bin, t_rel=relative_time, ion=False
        )
        # TODO add reference pls
        implantation_range = 1.9e-10 * incident_energy**0.59  # m
        return implantation_range

    def which_model(nb_bin: int, material: str):
        common_args = {
            "temperature": T_function,
            "deuterium_ion_flux": deuterium_ion_flux,
            "tritium_ion_flux": tritium_ion_flux,
            "deuterium_atom_flux": deuterium_atom_flux,
            "tritium_atom_flux": tritium_atom_flux,
            "final_time": my_scenario.get_maximum_time() - 1,
            "L": subbin.thickness,
            "L": sub_bin.thickness,
            "ion_implantation_range": ion_implantation_range,
            "atom_implantation_range": atom_implantation_range,
        }

        if isinstance(subbin, DivBin):
            parent_bin_index = subbin.index
        elif isinstance(subbin, SubBin):
            parent_bin_index = subbin.parent_bin_index

        if subbin.material == "W":
            return make_W_mb_model(
                **common_args,
                temperature=T_function_W,
                folder=f"mb{parent_bin_index+1}_{sub_bin.mode}_results",
            )
        elif subbin.material == "B":
            return make_B_mb_model(
                **common_args,
                temperature=T_function_B,
                folder=f"mb{parent_bin_index+1}_{sub_bin.mode}_results",
            )
        elif subbin.material == "SS":
            return make_DFW_mb_model(
                **common_args,
                temperature=T_function_W,  # TODO Change temperature to SS
                folder=f"mb{parent_bin_index+1}_dfw_results",
            )
        if material == "W":
            common_args["folder"] = f"mb{fw_bin.index+1}_{sub_bin.mode}_results"
            my_model, quantities = make_W_mb_model(**common_args)
        elif material == "B":
            common_args["folder"] = f"mb{fw_bin.index+1}_{sub_bin.mode}_results"
            my_model, quantities = make_B_mb_model(**common_args)
        elif material == "SS":
            common_args["folder"] = f"mb{fw_bin.index+1}_dfw_results"
            my_model, quantities = make_DFW_mb_model(**common_args)

        return my_model, quantities

    ############# RUN FW BIN SIMUS #############
    # TODO: adjust to run monoblocks in parallel
    for fw_bin in FW_bins.bins[:3]:  # only running 3 fw_bins to demonstrate capability
        for sub_bin in fw_bin.sub_bins:
            my_model, quantities = which_model(sub_bin)

            # add milestones for stepsize and adaptivity
            milestones = [pulse.total_duration for pulse in my_scenario.pulses]
            milestones += [pulse.duration_no_waiting for pulse in my_scenario.pulses]
            milestones.append(my_model.settings.final_time)
            milestones = sorted(np.unique(milestones))
            my_model.settings.stepsize.milestones = milestones
            my_model.settings.stepsize.growth_factor = 1.2
            my_model.settings.stepsize.cutback_factor = 0.9
            my_model.settings.stepsize.target_nb_iterations = 4

            my_model.settings.stepsize.max_stepsize = max_stepsize

            my_model.initialise()
            my_model.run()
            my_model.progress_bar.close()
            global_data.update(quantities)

    ############# RUN DIV BIN SIMUS #############
    # for div_bin in Div_bins.bins:
    for div_bin in Div_bins.bins[
        :4
    ]:  # only running 4 div bins to demonstrate capability
        my_model, quantities = which_model(div_bin)

        # add milestones for stepsize and adaptivity
        milestones = [pulse.total_duration for pulse in my_scenario.pulses]
        milestones += [pulse.duration_no_waiting for pulse in my_scenario.pulses]
        milestones.append(my_model.settings.final_time)
        milestones = sorted(np.unique(milestones))
        my_model.settings.stepsize.milestones = milestones
        my_model.settings.stepsize.growth_factor = 1.2
        my_model.settings.stepsize.cutback_factor = 0.9
        my_model.settings.stepsize.target_nb_iterations = 4

        my_model.settings.stepsize.max_stepsize = max_stepsize

        my_model.initialise()
        my_model.run()
        my_model.progress_bar.close()
        global_data.update(quantities)

    ############# Results Plotting #############
    # TODO: add a graph that computes grams

    for name, quantity in global_data.items():
        plt.plot(quantity.t, quantity.data, label=name, marker="o")

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.yscale("log")

    plt.show()

    fig, ax = plt.subplots()

    ax.stackplot(
        quantity.t,
        [quantity.data for quantity in global_data.values()],
        labels=global_data.keys(),
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.show()
