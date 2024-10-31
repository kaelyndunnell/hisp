import festim as F

import dolfinx.fem as fem
import ufl
import basix


class CustomProblem(F.HydrogenTransportProblem):
    def define_temperature(self):
        # check if temperature is None
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        # if temperature is a float or int, create a fem.Constant
        elif isinstance(self.temperature, (float, int)):
            self.temperature_fenics = F.as_fenics_constant(
                self.temperature, self.mesh.mesh
            )
        # if temperature is a fem.Constant or function, pass it to temperature_fenics
        elif isinstance(self.temperature, (fem.Constant, fem.Function)):
            self.temperature_fenics = self.temperature

        # if temperature is callable, process accordingly
        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), (float, int)):
                    raise ValueError(
                        f"self.temperature should return a float or an int, not {type(self.temperature(t=float(self.t)))} "
                    )
                # only t is an argument
                self.temperature_fenics = F.as_fenics_constant(
                    mesh=self.mesh.mesh, value=self.temperature(t=float(self.t))
                )
            else:
                x = ufl.SpatialCoordinate(self.mesh.mesh)
                degree = 1
                element_temperature = basix.ufl.element(
                    basix.ElementFamily.P,
                    self.mesh.mesh.basix_cell(),
                    degree,
                    basix.LagrangeVariant.equispaced,
                )
                function_space_temperature = fem.functionspace(
                    self.mesh.mesh, element_temperature
                )
                self.temperature_fenics = fem.Function(function_space_temperature)
                self.temperature_fenics.interpolate(
                    lambda x: self.temperature(x, float(self.t))
                )

    def update_time_dependent_values(self):

        # this is for the last time step, don't update the fluxes to avoid overshoot in the scenario file
        if float(self.t) > self.settings.final_time:
            return

        F.ProblemBase.update_time_dependent_values(self)

        if not self.temperature_time_dependent:
            return

        t = float(self.t)

        if isinstance(self.temperature_fenics, fem.Constant):
            self.temperature_fenics.value = self.temperature(t=t)
        elif isinstance(self.temperature_fenics, fem.Function):
            self.temperature_fenics.interpolate(
                lambda x: self.temperature(x, float(self.t))
            )

        for bc in self.boundary_conditions:
            if isinstance(bc, (F.FixedConcentrationBC, F.ParticleFluxBC)):
                if bc.temperature_dependent:
                    bc.update(t=t)

        for source in self.sources:
            if source.temperature_dependent:
                source.update(t=t)
