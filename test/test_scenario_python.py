from hisp.scenario import Scenario, Pulse
import pytest

def test_scenario():
    scenario = Scenario()
    assert len(scenario.pulses) == 0

    scenario.pulses.append(
        Pulse(
            pulse_type="FP",
            nb_pulses=2,
            ramp_up=0.1,
            steady_state=0.2,
            ramp_down=0.3,
            waiting=0.4,
        )
    )

    assert len(scenario.pulses) == 1
    assert scenario.pulses[0].pulse_type == "FP"
    assert scenario.pulses[0].nb_pulses == 2
    assert scenario.pulses[0].ramp_up == 0.1
    assert scenario.pulses[0].steady_state == 0.2
    assert scenario.pulses[0].ramp_down == 0.3
    assert scenario.pulses[0].waiting == 0.4

    scenario.to_txt_file("test_scenario.txt")
    scenario2 = Scenario.from_txt_file("test_scenario.txt")

    assert len(scenario2.pulses) == 1
    assert scenario2.pulses[0].pulse_type == "FP"
    assert scenario2.pulses[0].nb_pulses == 2
    assert scenario2.pulses[0].ramp_up == 0.1
    assert scenario2.pulses[0].steady_state == 0.2
    assert scenario2.pulses[0].ramp_down == 0.3
    assert scenario2.pulses[0].waiting == 0.4

def test_scenario_several_pulses():
    scenario = Scenario()
    assert len(scenario.pulses) == 0

    scenario.pulses.append(
        Pulse(
            pulse_type="FP",
            nb_pulses=2,
            ramp_up=0.1,
            steady_state=0.2,
            ramp_down=0.3,
            waiting=0.4,
        )
    )

    scenario.pulses.append(
        Pulse(
            pulse_type="ICWC",
            nb_pulses=3,
            ramp_up=0.5,
            steady_state=0.6,
            ramp_down=0.7,
            waiting=0.8,
        )
    )

    assert len(scenario.pulses) == 2
    assert scenario.pulses[0].pulse_type == "FP"
    assert scenario.pulses[0].nb_pulses == 2
    assert scenario.pulses[0].ramp_up == 0.1
    assert scenario.pulses[0].steady_state == 0.2
    assert scenario.pulses[0].ramp_down == 0.3
    assert scenario.pulses[0].waiting == 0.4

    assert scenario.pulses[1].pulse_type == "ICWC"
    assert scenario.pulses[1].nb_pulses == 3
    assert scenario.pulses[1].ramp_up == 0.5
    assert scenario.pulses[1].steady_state == 0.6
    assert scenario.pulses[1].ramp_down == 0.7
    assert scenario.pulses[1].waiting == 0.8

    scenario.to_txt_file("test_scenario.txt")
    scenario2 = Scenario.from_txt_file("test_scenario.txt")

    assert len(scenario2.pulses) == 2
    assert scenario2.pulses[0].pulse_type == "FP"
    assert scenario2.pulses[0].nb_pulses == 2
    assert scenario2.pulses[0].ramp_up == 0.1
    assert scenario2.pulses[0].steady_state == 0.2
    assert scenario2.pulses[0].ramp_down == 0.3
    assert scenario2.pulses[0].waiting == 0.4

    assert scenario2.pulses[1].pulse_type == "ICWC"
    assert scenario2.pulses[1].nb_pulses == 3
    assert scenario2.pulses[1].ramp_up == 0.5
    assert scenario2.pulses[1].steady_state == 0.6
    assert scenario2.pulses[1].ramp_down == 0.7
    assert scenario2.pulses[1].waiting == 0.8


def test_maximum_time():
    # BUILD

    pulse1 = Pulse(
        pulse_type="FP",
        nb_pulses=2,
        ramp_up=455,
        steady_state=455,
        ramp_down=650,
        waiting=1000,
    )
    pulse2 = Pulse(
        pulse_type="ICWC",
        nb_pulses=2,
        ramp_up=36,
        steady_state=36,
        ramp_down=180,
        waiting=1000,
    )
    my_scenario = Scenario([pulse1, pulse2])


    expected_maximum_time = 2 * (455 + 455 + 650 + 1000) + 2 * (36 + 36 + 180 + 1000)

    # RUN
    computed_maximum_time = my_scenario.get_maximum_time()

    # TEST
    assert computed_maximum_time == expected_maximum_time


pulse1 = Pulse(
        pulse_type="FP",
        nb_pulses=2,
        ramp_up=455,
        steady_state=455,
        ramp_down=650,
        waiting=1000,
    )
pulse2 = Pulse(
    pulse_type="ICWC",
    nb_pulses=2,
    ramp_up=36,
    steady_state=36,
    ramp_down=180,
    waiting=1000,
)
@pytest.mark.parametrize("t, expected_pulse", [(0, pulse1), (6000, pulse2), (1e5, None)])
def test_get_pulse(t, expected_pulse):

    my_scenario = Scenario([pulse1, pulse2])

    if  expected_pulse is None:
        with pytest.raises(ValueError):
            my_scenario.get_pulse(t=t)
    else:
        pulse = my_scenario.get_pulse(t=t)
        assert pulse == expected_pulse

@pytest.mark.parametrize("t, expected_pulse", [(100, pulse1)])
def test_one_pulse_scenario(t, expected_pulse):
    my_scenario = Scenario([expected_pulse])

    pulse = my_scenario.get_pulse(t=t)

    assert pulse == expected_pulse

@pytest.mark.parametrize("row, expected_duration", [(0, 2560), (1, 1252)])
def test_get_pulse_duration(row, expected_duration):
    my_scenario = Scenario([pulse1, pulse2])

    duration = my_scenario.get_pulse_duration(row=row)

    assert duration == expected_duration

@pytest.mark.parametrize("row, expected_duration", [(0, 1560), (1, 252)])
def test_get_pulse_duration_no_waiting(row, expected_duration):
    my_scenario = Scenario([pulse1, pulse2])

    duration = my_scenario.get_pulse_duration_no_waiting(row=row)

    assert duration == expected_duration

@pytest.mark.parametrize("row, expected_time", [(0, 0.0), (1, 5120.0)])
def test_get_time_till_row(row, expected_time):
    my_scenario = Scenario([pulse1, pulse2])

    elapsed_time = my_scenario.get_time_till_row(row=row)

    assert elapsed_time == expected_time