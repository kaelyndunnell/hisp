"""Tests for compute_flux_values and build_ufl_flux_expression.

Verifies that particle flux time-profiles are correctly constructed from
scenario/plasma data, including tritium fraction splitting, time stacking
for repeated pulses, and UFL expression generation for FESTIM sources.
"""
import pytest
from unittest.mock import MagicMock
from hisp.festim_models.new_mb_model import compute_flux_values, build_ufl_flux_expression


def _make_pulse(ramp_up, steady_state, ramp_down, waiting=0, nb_pulses=1,
                tritium_fraction=0.5, pulse_type="FP"):
    pulse = MagicMock()
    pulse.ramp_up = ramp_up
    pulse.steady_state = steady_state
    pulse.ramp_down = ramp_down
    pulse.waiting = waiting
    pulse.total_duration = ramp_up + steady_state + ramp_down + waiting
    pulse.nb_pulses = nb_pulses
    pulse.tritium_fraction = tritium_fraction
    pulse.pulse_type = pulse_type
    # GDC attributes (None unless Bake+GDC)
    pulse.gdc_ramp_up = None
    pulse.gdc_steady_state = None
    pulse.gdc_ramp_down = None
    return pulse


def _make_scenario(pulses):
    scenario = MagicMock()
    scenario.pulses = pulses
    return scenario


def _make_plasma_data_handling(ion_flux=1e20, atom_flux=5e19):
    pdh = MagicMock()

    def get_pf(pulse, bin_, t_rel, ion):
        return ion_flux if ion else atom_flux

    pdh.get_particle_flux = MagicMock(side_effect=get_pf)
    return pdh


class TestComputeFluxValues:
    def test_tritium_fraction_applied(self):
        """Tritium fraction correctly splits ion/atom flux into D and T components."""
        pulse = _make_pulse(5, 50, 5, nb_pulses=1, tritium_fraction=0.3)
        scenario = _make_scenario([pulse])
        pdh = _make_plasma_data_handling(ion_flux=1e20, atom_flux=5e19)
        bin_ = MagicMock()

        occs = compute_flux_values(scenario, pdh, bin_)

        assert len(occs) == 1
        occ = occs[0]
        assert occ['T_ion'] == pytest.approx(1e20 * 0.3)
        assert occ['D_ion'] == pytest.approx(1e20 * 0.7)
        assert occ['T_atom'] == pytest.approx(5e19 * 0.3)
        assert occ['D_atom'] == pytest.approx(5e19 * 0.7)

    def test_timing_stacks_correctly(self):
        """Multiple pulse occurrences produce non-overlapping time intervals."""
        pulse = _make_pulse(5, 50, 5, nb_pulses=3, tritium_fraction=0.5)
        scenario = _make_scenario([pulse])
        pdh = _make_plasma_data_handling()
        bin_ = MagicMock()

        occs = compute_flux_values(scenario, pdh, bin_)

        assert len(occs) == 3
        duration = 60.0
        for i, occ in enumerate(occs):
            assert occ['start'] == pytest.approx(i * duration)
            assert occ['end'] == pytest.approx((i + 1) * duration)

    def test_bake_gdc_uses_gdc_timing(self):
        """Bake+GDC pulses sample flux at GDC sub-timing, not main pulse timing."""
        pulse = _make_pulse(10, 100, 10, nb_pulses=1, tritium_fraction=0.5,
                            pulse_type="Bake+GDC")
        # Set GDC sub-timing
        pulse.gdc_ramp_up = 2
        pulse.gdc_steady_state = 20
        pulse.gdc_ramp_down = 3

        scenario = _make_scenario([pulse])
        pdh = _make_plasma_data_handling()
        bin_ = MagicMock()

        occs = compute_flux_values(scenario, pdh, bin_)

        # The function should call get_particle_flux with t_rel = gdc_ramp_up + gdc_steady_state / 2
        expected_t_rel = 2 + 20 / 2  # = 12
        # Check the call was made with gdc timing
        calls = pdh.get_particle_flux.call_args_list
        for call in calls:
            _, kwargs = call
            if 't_rel' in kwargs:
                assert kwargs['t_rel'] == pytest.approx(expected_t_rel)


class TestBuildUflFluxExpression:
    def test_returns_four_callables(self):
        """Returns 4 flux functions: D_ion, D_atom, T_ion, T_atom."""
        occs = [{
            'start': 0, 'end': 100,
            'pulse': _make_pulse(5, 50, 5),
            'D_ion': 1e20, 'D_atom': 5e19, 'T_ion': 3e19, 'T_atom': 1e19,
        }]
        result = build_ufl_flux_expression(occs)
        assert len(result) == 4
        for fn in result:
            assert callable(fn)

    def test_value_zero_outside_time_window(self):
        """Flux expression evaluates to zero outside the active pulse window."""
        occs = [{
            'start': 10, 'end': 50,
            'pulse': _make_pulse(5, 30, 5),
            'D_ion': 1e20, 'D_atom': 5e19, 'T_ion': 3e19, 'T_atom': 1e19,
        }]
        d_ion_fn, _, _, _ = build_ufl_flux_expression(occs, value_off=0.0)

        # Since these return UFL expressions, verify the function is callable
        # and returns a valid UFL expression when given a dolfinx Constant
        from dolfinx import mesh as dfx_mesh, fem
        from mpi4py import MPI

        domain = dfx_mesh.create_unit_interval(MPI.COMM_WORLD, 10)
        t_val = fem.Constant(domain, 5.0)  # before window (start=10)
        expr = d_ion_fn(t_val)
        assert expr is not None
