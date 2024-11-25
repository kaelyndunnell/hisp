from hisp import Pulse

import pytest


def test_pulse_initialization():
    pulse = Pulse(
        pulse_type="FP",
        nb_pulses=2,
        ramp_up=0.1,
        steady_state=0.2,
        ramp_down=0.3,
        waiting=0.4,
        tritium_fraction=0.5,
    )

    assert pulse.pulse_type == "FP"
    assert pulse.nb_pulses == 2
    assert pulse.ramp_up == 0.1
    assert pulse.steady_state == 0.2
    assert pulse.ramp_down == 0.3
    assert pulse.waiting == 0.4


def test_pulse_total_duration():
    pulse = Pulse(
        pulse_type="FP",
        nb_pulses=2,
        ramp_up=0.1,
        steady_state=0.2,
        ramp_down=0.3,
        waiting=0.4,
        tritium_fraction=0.5,
    )

    assert pulse.total_duration == 1.0


def test_pulse_total_duration_with_zeros():
    pulse = Pulse(
        pulse_type="RISP",
        nb_pulses=1,
        ramp_up=0.0,
        steady_state=0.0,
        ramp_down=0.0,
        waiting=0.0,
        tritium_fraction=0.0,
    )

    assert pulse.total_duration == 1800.0


def test_pulse_total_duration_no_waiting_with_zeros():
    pulse = Pulse(
        pulse_type="RISP",
        nb_pulses=1,
        ramp_up=0.0,
        steady_state=0.0,
        ramp_down=0.0,
        waiting=0.0,
        tritium_fraction=0.0,
    )

    assert pulse.duration_no_waiting == 270.0


def test_pulse_risp_with_zeros_raises_warning():
    with pytest.warns(UserWarning):
        pulse = Pulse(
            pulse_type="RISP",
            nb_pulses=1,
            ramp_up=0.0,
            steady_state=0.0,
            ramp_down=0.0,
            waiting=0.0,
            tritium_fraction=0.0,
        )
        pulse.total_duration
    assert pulse.ramp_up != 0.0
    assert pulse.steady_state != 0.0
    assert pulse.ramp_down != 0.0
    assert pulse.waiting != 0.0


def test_pulse_duration_no_waiting():
    pulse = Pulse(
        pulse_type="FP",
        nb_pulses=2,
        ramp_up=0.1,
        steady_state=0.2,
        ramp_down=0.3,
        waiting=0.4,
        tritium_fraction=0.5,
    )

    assert pulse.duration_no_waiting == 0.6
    assert pulse.duration_no_waiting == pulse.total_duration - pulse.waiting
