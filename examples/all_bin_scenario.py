import json
import pandas as pd

from hisp.plamsa_data_handling import PlasmaDataHandling

from make_iter_bins import FW_bins, Div_bins, my_reactor

from hisp.scenario import Scenario, Pulse

from hisp.model import Model

# import dolfinx
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

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

my_hisp_model = Model(
    reactor=my_reactor,
    scenario=my_scenario,
    plasma_data_handling=plasma_data_handling,
    coolant_temp=343.0,
)

if __name__ == "__main__":
    global_data = {}
    processed_data = []

    # running only a subset of the bins for demonstration purposes

    ############# RUN FW BIN SIMUS #############
    for fw_bin in FW_bins.bins[:3]:
        global_data[fw_bin] = {}
        fw_bin_data = {"bin_index": fw_bin.index, "sub_bins": []}

        for sub_bin in fw_bin.sub_bins:
            my_model, quantities = my_hisp_model.run_bin(sub_bin)

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
        my_model, quantities = my_hisp_model.run_bin(div_bin)

        global_data[div_bin] = quantities

        bin_data = {
            key: {"t": value.t, "data": value.data} for key, value in quantities.items()
        }
        bin_data["bin_index"] = div_bin.index

        processed_data.append(bin_data)

    # write the processed data to JSON
    with open("processed_data.json", "w+") as f:
        json.dump(processed_data, f, indent=4)
