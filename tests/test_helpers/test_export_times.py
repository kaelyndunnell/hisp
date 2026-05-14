"""Tests for compute_export_times.

Verifies that the function generates correct time-sampling points
for data export across pulse scenarios, ensuring proper spacing
and cumulative offsets when multiple pulses are chained.
"""
import pytest
from unittest.mock import MagicMock
from hisp.festim_models.new_mb_model import compute_export_times


def _make_pulse(ramp_up, steady_state, ramp_down, waiting=0, nb_pulses=1):
    """Create a mock pulse with given timing parameters."""
    pulse = MagicMock()
    pulse.ramp_up = ramp_up
    pulse.steady_state = steady_state
    pulse.ramp_down = ramp_down
    pulse.waiting = waiting
    pulse.total_duration = ramp_up + steady_state + ramp_down + waiting
    pulse.nb_pulses = nb_pulses
    return pulse


def _make_scenario(pulses):
    """Create a mock scenario with given pulse list."""
    scenario = MagicMock()
    scenario.pulses = pulses
    return scenario


class TestComputeExportTimes:
    def test_single_pulse_3_samples(self):
        """3 samples span: start of ramp-up, mid-steady-state, end of pulse."""
        pulse = _make_pulse(10, 100, 10, waiting=0, nb_pulses=1)
        scenario = _make_scenario([pulse])

        times = compute_export_times(scenario, samples_per_pulse=3)

        assert len(times) == 3
        assert times[0] == 0.0  # start of ramp-up
        assert times[1] == pytest.approx(10 + 50)  # middle of steady-state
        assert times[2] == pytest.approx(120)  # end of pulse

    def test_multiple_pulses_cumulative_offsets(self):
        """Multiple pulse occurrences stack in time (no overlap)."""
        pulse1 = _make_pulse(5, 50, 5, waiting=0, nb_pulses=2)
        pulse2 = _make_pulse(10, 100, 10, waiting=0, nb_pulses=1)
        scenario = _make_scenario([pulse1, pulse2])

        times = compute_export_times(scenario, samples_per_pulse=3)

        # pulse1 has nb_pulses=2, pulse2 has nb_pulses=1 -> 3 occurrences total
        assert len(times) == 9

        # First occurrence of pulse1: [0, 5+25, 60]
        assert times[0] == 0.0
        assert times[1] == pytest.approx(5 + 25)
        assert times[2] == pytest.approx(60)

        # Second occurrence of pulse1: starts at 60
        assert times[3] == pytest.approx(60)
        assert times[4] == pytest.approx(60 + 5 + 25)
        assert times[5] == pytest.approx(120)

        # pulse2: starts at 120
        assert times[6] == pytest.approx(120)

    def test_custom_samples_per_pulse(self):
        """Samples are evenly spaced within the pulse duration."""
        pulse = _make_pulse(10, 80, 10, waiting=0, nb_pulses=1)
        scenario = _make_scenario([pulse])

        times = compute_export_times(scenario, samples_per_pulse=5)

        assert len(times) == 5
        # Evenly spaced within pulse duration
        duration = 100
        for i, t in enumerate(times):
            expected = (i + 0.5) * duration / 5
            assert t == pytest.approx(expected)

    def test_len_equals_nb_pulses_times_samples(self):
        """Total number of export times = nb_pulses × samples_per_pulse."""
        pulse = _make_pulse(2, 20, 2, waiting=0, nb_pulses=4)
        scenario = _make_scenario([pulse])
        samples = 7

        times = compute_export_times(scenario, samples_per_pulse=samples)

        assert len(times) == 4 * samples
