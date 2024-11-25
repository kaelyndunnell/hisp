import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np
import numpy.typing as npt


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


def gaussian_distribution(
    x: npt.NDArray, mean: callable, width: float, mod=ufl
) -> ufl.core.expr.Expr:
    return mod.exp(-((x[0] - mean(x[0])) ** 2) / (2 * width**2))


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
