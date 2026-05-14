import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np
import numpy.typing as npt
from hisp.scenario import Pulse
from dolfinx import fem
import math
from festim import XDMFExport


class PulsedSource(F.ParticleSource):
    def __init__(self, flux, distribution, volume, species):
        """Initalizes flux and distribution for PulsedSource.

        Args:
            flux (callable): the input flux value from DINA data
            distribution (function of x): distribution of flux throughout mb
            volume (F.VolumeSubdomain1D): volume where this flux is imposed
            species (F.species): species of flux (e.g. D/T)

        Returns:
            flux and distribution of species.
        """
        self.flux = flux
        self.distribution = distribution
        super().__init__(None, volume, species)

    @property
    def time_dependent(self):
        return True

    def create_value_fenics(self, mesh, temperature, t: Constant):
        self.flux_fenics = F.as_fenics_constant(self.flux(t.value), mesh)
        x = ufl.SpatialCoordinate(mesh)
        self.distribution_fenics = self.distribution(x)

        self.value_fenics = self.flux_fenics * self.distribution_fenics

    def update(self, t: float):
        self.flux_fenics.value = self.flux(t)

# we override Stepsize to control the precision of milestones detection
# TODO remove this when https://github.com/festim-dev/FESTIM/issues/933 is fixed
class Stepsize(F.Stepsize):
    def modify_value(self, value, nb_iterations, t=None):
        if not self.is_adapt(t):
            return value

        if nb_iterations < self.target_nb_iterations:
            updated_value = value * self.growth_factor
        elif nb_iterations > self.target_nb_iterations:
            updated_value = value * self.cutback_factor
        else:
            updated_value = value

        if max_step := self.get_max_stepsize(t):
            if updated_value > max_step:
                updated_value = max_step

        next_milestone = self.next_milestone(t)
        if next_milestone is not None:
            time_to_milestone = next_milestone - t
            if updated_value > time_to_milestone and not np.isclose(
                t, next_milestone, atol=0.0001, rtol=0
            ):
                updated_value = time_to_milestone

        return updated_value

def gaussian_distribution(
    x: npt.NDArray, mean: float, width: float, mod=ufl
) -> ufl.core.expr.Expr:
    """Generates a gaussian distribution for particle sources.

    Args:
        x (npt.NDArray): x values along the length of given bin.
        mean (float): Mean of the distribution.
        width (float): Width of the gaussian distribution.
        mod (_type_, optional): Module used to express gaussian distribution. Defaults to ufl.

    Returns:
        ufl.core.expr.Expr: Gaussian distribution with area 1.  
    """
    return mod.exp(-((x[0] - mean) ** 2) / (2 * width**2)) / (
        np.sqrt(2 * np.pi * width**2)
    )


def periodic_step_function(x, period_on, period_total, value, value_off=0.0):
    """
    Creates a periodic step function with two periods.
    """

    if period_total < period_on:
        raise ValueError("period_total must be greater than period_on")

    if x % period_total < period_on:
        return value
    else:
        return value_off
    
def periodic_pulse_function(current_time: float, pulse: Pulse, value, value_off=343.0):
    """Creates bake function with ramp up rate and ramp down rate.

    Args:
        current_time (float): time within the pulse 
        pulse (Pulse): pulse of HISP Pulse class
        value (float): steady-state value 
        value_off (float): value at t=0 and t=final time. 
    """
    
    if current_time == pulse.total_duration:
        return value_off
    elif current_time % pulse.total_duration < pulse.ramp_up: # ramp up 
        return (value - value_off) / (pulse.ramp_up) * current_time + value_off # y = mx + b, slope is temp/ramp up time
    elif current_time % pulse.total_duration < pulse.ramp_up + pulse.steady_state: # steady state
        return value
    else: # ramp down, waiting
        lower_value = value - (value - value_off)/pulse.ramp_down * (current_time - (pulse.ramp_up + pulse.steady_state)) # y = mx + b, slope is temp/ramp down time
        if lower_value >= value_off: 
            return lower_value
        else: 
            return value_off

class XDMFExportEveryDt(XDMFExport):
    """
    Write to XDMF only if enough time has elapsed since the last write.
    Uses min_dt1 for t <= switch and min_dt2 for t > switch.

    Parameters
    ----------
    filename : str
        Path for the XDMF file(s).
    field : str or festim Field
        What to export (same as base XDMFExport).
    min_dt1 : float
        Minimum time spacing before `switch` (inclusive).
    min_dt2 : float
        Minimum time spacing after `switch` (strictly greater).
    switch : float
        Time at which the cadence changes from min_dt1 to min_dt2.
    atol : float, optional
        Small tolerance to account for floating point accumulation.
        Default is 0.0 (set e.g. 1e-12 if needed).
    """
    def __init__(self, filename, field, min_dt1: float, min_dt2: float, switch: float, atol: float = 0.0):
        super().__init__(filename, field)
        self._min_dt1 = float(min_dt1)
        self._min_dt2 = float(min_dt2)
        self._switch = float(switch)
        self._atol = float(atol)
        self._last_t = None

    def _current_min_dt(self, t: float) -> float:
        return self._min_dt2 if t > self._switch else self._min_dt1

    def write(self, t: float):
        t = float(t)
        min_dt = self._current_min_dt(t)

        if (self._last_t is None) or ((t - self._last_t) >= (min_dt - self._atol)):
            super().write(t)
            self._last_t = t

def gaussian_implantation_ufl(Rp, sigma, axis=0, thickness=None):
    """
    Returns callable value(x, t) -> UFL expression S(x,t) [m^-3 s^-1]
    - Rp, sigma in meters
    - axis in {0,1,2} selects x[axis] as depth coordinate
    - If thickness is not None (meters), renormalize over [0, thickness] to conserve J(t)
    """
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    if thickness is None:
        C = 0.5 * (1.0 + erf(Rp / (sigma * sqrt(2.0))))  # Gaussian mass in [0, +inf)
        C = max(C, 1e-12)  # numerical safeguard
        norm = inv_sqrt_2pi / sigma
        def value(x):
            xi = x[axis]
            z  = (xi - Rp) / sigma
            return norm * ufl.exp(-0.5 * z * z)
        return value
    else:
        # Renormalize over [0, thickness]
        from math import erf, sqrt
        a = (0.0 - Rp) / (sigma * sqrt(2.0))
        b = (thickness - Rp) / (sigma * sqrt(2.0))
        C = max(0.5 * (erf(b) - erf(a)), 1e-12)        # in-domain Gaussian mass
        norm = (inv_sqrt_2pi / sigma) / C
        def value(x):
            xi = x[axis]
            z  = (xi - Rp) / sigma
            return norm * ufl.exp(-0.5 * z * z)
        return value
    
def periodic_pulse_ufl(t, pulse, value, value_off=343.0):
    """
    UFL symbolic version of periodic_pulse_function.
    Args:
        t: UFL expression (time)
        pulse: Pulse object with ramp_up, steady_state, ramp_down, waiting
        value: UFL expression or Constant (steady-state value)
        value_off: float or UFL Constant (off value)
    Returns:
        UFL expression representing the piecewise ramp profile.
    """

    # Compute relative time within one pulse cycle (no modulo in UFL)
    tau = t  # Assume t is within [0, pulse.total_duration] for now

    # Conditions for each phase
    within_up = ufl.lt(tau, pulse.ramp_up)
    within_steady = ufl.And(ufl.ge(tau, pulse.ramp_up),
                             ufl.lt(tau, pulse.ramp_up + pulse.steady_state))
    within_down = ufl.And(ufl.ge(tau, pulse.ramp_up + pulse.steady_state),
                          ufl.lt(tau, pulse.ramp_up + pulse.steady_state + pulse.ramp_down))
    # Waiting phase: tau >= ramp_up + steady_state + ramp_down

    # Ramp-up: linear interpolation
    up_val = (value - value_off) / pulse.ramp_up * tau + value_off

    # Ramp-down: linear decrease
    down_val = value - (value - value_off) / pulse.ramp_down * (tau - (pulse.ramp_up + pulse.steady_state))
    down_val = ufl.conditional(ufl.ge(down_val, value_off), down_val, value_off)

    # Piecewise conditional chain
    shape = ufl.conditional(
        within_up, up_val,
        ufl.conditional(
            within_steady, value,
            ufl.conditional(within_down, down_val, value_off)
        )
    )

    return shape

def make_ufl_flux_function(scalar_flux_function):
    """
    Convert a scalar flux function to a UFL expression function.
    
    Args:
        scalar_flux_function: A function that takes time (float) and returns flux value (float)
        
    Returns:
        A function that takes time (UFL expression) and returns UFL expression
    """
    def ufl_flux(t):
        # For time-dependent behavior, we need to create conditional expressions
        # This is a simplified version - for complex time dependencies,
        # you might need to sample at specific times and interpolate
        return ufl.Constant(1.0)  # Placeholder - this needs proper implementation
    
    return ufl_flux