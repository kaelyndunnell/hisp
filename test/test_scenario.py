from src.hisp.helpers import Scenario
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
scenario_path = os.path.join(current_dir, "scenario_test.txt")


def test_maximum_time():
    # BUILD
    my_scenario = Scenario(scenario_path)
    expected_maximum_time = 2 * (455 + 455 + 650 + 1000) + 2 * (36 + 36 + 180 + 1000)

    # RUN
    computed_maximum_time = my_scenario.get_maximum_time()

    # TEST
    assert computed_maximum_time == expected_maximum_time


@pytest.mark.parametrize("t, expected_row", [(0, 0), (6000, 1)])
def test_get_pulse_row(t, expected_row):
    my_scenario = Scenario(scenario_path)

    pulse_row = my_scenario.get_row(t=t)

    assert pulse_row == expected_row


def test_reading_a_file():
    my_scenario = Scenario(scenario_path)

    times = np.linspace(0, my_scenario.get_maximum_time(), 1000)
    pulse_types = []
    for t in times:
        pulse_type = my_scenario.get_pulse_type(t)
        pulse_types.append(pulse_type)

    # color the line based on the pulse type
    color = {
        "FP": "red",
        "ICWC": "blue",
        "RISP": "green",
        "GDC": "orange",
        "BAKE": "purple",
    }

    colors = [color[pulse_type] for pulse_type in pulse_types]

    for i in range(len(times) - 1):
        plt.plot(times[i : i + 2], np.ones_like(times[i : i + 2]), c=colors[i])
    # plt.xscale("log")
    # plt.show()

one_line_scenario_path = os.path.join(current_dir, "one_line_scenario.txt")

@pytest.mark.parametrize("t, expected_row", [(100, 0)])
def test_one_line_scenario(t, expected_row):
    my_scenario = Scenario(one_line_scenario_path)

    pulse_row = my_scenario.get_row(t=t)

    assert pulse_row == expected_row 

@pytest.mark.parametrize("row, expected_duration", [(0,2560), (1,1252)])
def test_get_pulse_duration(row, expected_duration):
    my_scenario = Scenario(scenario_path)

    duration = my_scenario.get_pulse_duration(row=row)

    assert duration == expected_duration 

@pytest.mark.parametrize("row, expected_duration", [(0,1560), (1,252)])
def test_get_pulse_duration_no_waiting(row, expected_duration):
    my_scenario = Scenario(scenario_path)

    duration = my_scenario.get_pulse_duration_no_waiting(row=row)

    assert duration == expected_duration 