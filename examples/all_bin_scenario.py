import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.festim_models import (
    make_W_mb_model,
    make_B_mb_model,
    make_DFW_mb_model,
    make_temperature_function,
    make_particle_flux_function,
)
from make_iter_bins import FW_bins, Div_bins

from hisp.helpers import periodic_step_function
from hisp.scenario import Scenario, Pulse
from hisp.bin import SubBin, DivBin

# import dolfinx

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

COOLANT_TEMP = 343  # 70 degree C cooling water
fp = Pulse(
    pulse_type="FP",
    nb_pulses=1,
    ramp_up=10,
    steady_state=10,
    ramp_down=10,
    waiting=100,
    tritium_fraction=0.5,
)

my_scenario = Scenario(pulses=[fp])

data_folder = "data"
plasma_data_handling = PlasmaDataHandling(
    pulse_type_to_data={
        "FP": pd.read_csv(data_folder + "/Binned_Flux_Data.dat", delimiter=","),
        "ICWC": pd.read_csv(data_folder + "/ICWC_data.dat", delimiter=","),
        "GDC": pd.read_csv(data_folder + "/GDC_data.dat", delimiter=","),
    },
    path_to_ROSP_data=data_folder + "/ROSP_data",
    path_to_RISP_data=data_folder + "/RISP_data",
    path_to_RISP_wall_data=data_folder + "/RISP_Wall_data.dat",
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


def which_model(bin: SubBin | DivBin):
    """Returns the correct model for the subbin.

    Args:
        bin: The bin/subbin to get the model for

    Returns:
        festim.HTransportModel, dict: The model and the quantities to plot
    """
    temperature_fuction = make_temperature_function(
        scenario=my_scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        coolant_temp=COOLANT_TEMP,
    )
    d_ion_incident_flux = make_particle_flux_function(
        scenario=my_scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        ion=True,
        tritium=False,
    )
    tritium_ion_flux = make_particle_flux_function(
        scenario=my_scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        ion=True,
        tritium=True,
    )
    deuterium_atom_flux = make_particle_flux_function(
        scenario=my_scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        ion=False,
        tritium=False,
    )
    tritium_atom_flux = make_particle_flux_function(
        scenario=my_scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        ion=False,
        tritium=True,
    )
    common_args = {
        "deuterium_ion_flux": d_ion_incident_flux,
        "tritium_ion_flux": tritium_ion_flux,
        "deuterium_atom_flux": deuterium_atom_flux,
        "tritium_atom_flux": tritium_atom_flux,
        "final_time": my_scenario.get_maximum_time() - 1,
        "temperature": temperature_fuction,
        "L": bin.thickness,
    }

    if isinstance(bin, DivBin):
        parent_bin_index = bin.index
    elif isinstance(bin, SubBin):
        parent_bin_index = bin.parent_bin_index

    if bin.material == "W":
        return make_W_mb_model(
            **common_args,
            folder=f"mb{parent_bin_index+1}_{bin.mode}_results",
        )
    elif bin.material == "B":
        return make_B_mb_model(
            **common_args,
            folder=f"mb{parent_bin_index+1}_{bin.mode}_results",
        )
    elif bin.material == "SS":
        return make_DFW_mb_model(
            **common_args,
            folder=f"mb{parent_bin_index+1}_dfw_results",
        )


def run_bin(bin: SubBin | DivBin):
    """
    Runs the FESTIM model for the given bin.

    Args:
        bin: The bin to run the model for

    Returns:
        The model, the quantities
    """
    # create the FESTIM model
    my_model, quantities = which_model(bin)

    # add milestones for stepsize and adaptivity
    milestones = []
    current_time = 0
    for pulse in my_scenario.pulses:
        start_of_pulse = my_scenario.get_time_start_current_pulse(current_time)
        for i in range(pulse.nb_pulses):
            milestones.append(start_of_pulse + pulse.total_duration * (i + 1))
            milestones.append(
                start_of_pulse + pulse.total_duration * i + pulse.duration_no_waiting
            )

        current_time = start_of_pulse + pulse.total_duration * pulse.nb_pulses
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

    return my_model, quantities


if __name__ == "__main__":
    global_data = {}
    processed_data = []

    # running only a subset of the bins for demonstration purposes

    ############# RUN FW BIN SIMUS #############
    for fw_bin in FW_bins.bins[:3]:
        global_data[fw_bin] = {}
        fw_bin_data = {"bin_index": fw_bin.index, "sub_bins": []}

        for sub_bin in fw_bin.sub_bins:
            my_model, quantities = run_bin(sub_bin)

            global_data[fw_bin][sub_bin] = quantities

            subbin_data = {
                key: {"t": value.t, "data": value.data}
                for key, value in quantities.items()
            }
            subbin_data["mode"] = sub_bin.mode
            subbin_data["parent_bin_index"] = sub_bin.parent_bin_index

            fw_bin_data["sub_bins"].append(subbin_data)

        processed_data.append(fw_bin_data)

    ############# RUN DIV BIN SIMUS #############
    for div_bin in Div_bins.bins[15:18]:
        my_model, quantities = run_bin(div_bin)

        global_data[div_bin] = quantities

        bin_data = {
            key: {"t": value.t, "data": value.data} for key, value in quantities.items()
        }
        bin_data["bin_index"] = div_bin.index

        processed_data.append(bin_data)

    # write the processed data to JSON
    with open("processed_data.json", "w+") as f:
        json.dump(processed_data, f, indent=4)
