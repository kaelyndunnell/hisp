from hisp.helpers import Scenario
import os
import pytest

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
