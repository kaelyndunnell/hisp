import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np
import numpy.typing as npt

from typing import Callable


class PulsedSource(F.ParticleSource):
    def __init__(
        self, flux: Callable, mean: Callable, width: Callable, reflection_coeff: float, volume, species
    ):
        """Initalizes flux and distribution for PulsedSource. Encorces distribution
        as a gaussian.

        Args:
            flux (callable): the input flux value from DINA data
            mean (callable): the mean of the distribution as a function of time
            width (callable): the width of the distribution as a function of time
            reflection_coeff (float): the reflection coefficient 
            volume (F.VolumeSubdomain1D): volume where this flux is imposed
            species (F.species): species of flux (e.g. D/T)
        """
        self.flux = flux
        self.mean = mean
        self.width = width
        self.reflection_coeff = reflection_coeff
        super().__init__(None, volume, species)

    @property
    def time_dependent(self):
        return True

    def create_value_fenics(self, mesh, temperature, t: Constant):
        self.incident_flux = F.as_fenics_constant(self.flux(float(t)), mesh)
        self.mean_fenics = F.as_fenics_constant(self.mean(float(t)), mesh)
        self.width_fenics = F.as_fenics_constant(self.width(float(t)), mesh)

        x = ufl.SpatialCoordinate(mesh)
        distribution = gaussian_distribution(
            x, self.mean_fenics, self.width_fenics, mod=ufl
        )

        implanted_flux = (1-self.reflection_coeff)*self.incident_flux
        self.value_fenics = implanted_flux * distribution

    def update(self, t: float):
        self.incident_flux = self.flux(t)
        self.mean_fenics.value = self.mean(t)
        self.width_fenics.value = self.width(t)


def gaussian_distribution(
    x: npt.NDArray, mean: float | ufl.Constant, width: float | ufl.Constant, mod=ufl
) -> ufl.core.expr.Expr:
    return mod.exp(-((x[0] - mean) ** 2) / (2 * width**2))


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
