"""Scenario C, capability_test — DT/DD alternation + GDC + bake.

5 DT pulses, 1 D-only pulse, 5 DT pulses, 1 D-only pulse,
then 2-day glow discharge cleaning, then baking at 493 K.
Uses Binned_Flux_Data_rad.dat for FP fluxes.
"""
from pathlib import Path
import pandas as pd

from hisp.scenario import Pulse, Scenario
from hisp.plasma_data_handling import PlasmaDataHandling

data_folder = str(Path(__file__).parent / "data")

plasma_data_handling = PlasmaDataHandling(
    pulse_type_to_data={
        "FP": pd.read_csv(data_folder + "/Binned_Flux_Data_rad.dat", delimiter=","),
        "FP_D": pd.read_csv(data_folder + "/Binned_Flux_Data_just_D_pulse.dat", delimiter=",", comment="#"),
        "GDC": pd.read_csv(data_folder + "/GDC_data.dat", delimiter=","),
    },
    path_to_RISP_data=data_folder + "/RISP_data",
    path_to_ROSP_data=data_folder + "/ROSP_data",
    path_to_RISP_wall_data=data_folder + "/RISP_Wall_data.dat",
)

fp = Pulse(
    pulse_type="FP",
    nb_pulses=5,
    ramp_up=429,
    steady_state=650,
    ramp_down=455,
    waiting=3600,
    tritium_fraction=0.5,
)

fp_d = Pulse(
    pulse_type="FP_D",
    nb_pulses=1,
    ramp_up=429,
    steady_state=650,
    ramp_down=455,
    waiting=3600,
    tritium_fraction=0.0,
)

gdc = Pulse(
    pulse_type="GDC",
    nb_pulses=1,
    ramp_up=1,
    steady_state=172798,   # 2-day glow
    ramp_down=1,
    waiting=172800,        # 2-day waiting
    tritium_fraction=0.0,
)

bake = Pulse(
    pulse_type="BAKE",
    nb_pulses=1,
    ramp_up=151200,      # 5 °C/hour; 42 hours total
    steady_state=345600,
    ramp_down=108000,    # -7 °C/hour; 30 hours total
    waiting=11,          # HISP expects at least 10 s of waiting
    tritium_fraction=0.0,
)

scenario = Scenario(pulses=[fp, fp_d, fp, fp_d, gdc, bake], baking_temp=273.15 + 220)  # 220 °C in Kelvin
