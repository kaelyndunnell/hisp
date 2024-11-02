from hisp.scenario import Scenario, Pulse


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
    