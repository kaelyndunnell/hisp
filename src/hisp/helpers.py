import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np
import numpy.typing as npt
from hisp.scenario import Pulse


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
        self.flux_fenics = F.as_fenics_constant(self.flux(float(t)), mesh)
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
    
    if current_time % pulse.total_duration < pulse.ramp_up: # ramp up 
        return (value - value_off) / (pulse.ramp_up) * current_time + value_off # y = mx + b, slope is temp/ramp up time
    elif current_time % pulse.total_duration < pulse.ramp_up + pulse.steady_state: # steady state
        return value
    else: # ramp down, waiting
        current_temp = value - (value - value_off)/pulse.ramp_down * (current_time - (pulse.ramp_up + pulse.steady_state)) # y = mx + b, slope is temp/ramp down time
        
        if current_temp >= value_off: 
            return current_temp
        else: 
            return value_off