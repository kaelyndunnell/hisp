import festim as F
from dolfinx.fem.function import Constant
import ufl


class PulsedSource(F.ParticleSource):
    def __init__(self, flux, distribution, volume, species):
        """Initalizes flux and distribution for PulsedSource. 

        Args:
            flux (float): the input flux value from DINA data
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
        self.flux_fenics = Constant(mesh, float(self.flux(t)))
        x = ufl.SpatialCoordinate(mesh)
        self.distribution_fenics = self.distribution(x)

        self.value_fenics = self.flux_fenics * self.distribution_fenics

    def update(self, t: float):
        self.flux_fenics.value = self.flux(t)
