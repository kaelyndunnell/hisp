"""Custom HydrogenTransportProblem subclass for HISP.

The only reason this subclass exists is that HISP's temperature functions
are numpy-based callables ``T(x, t)`` (where *x* is a numpy coordinate
array), whereas FESTIM 2.0's ``define_temperature`` expects the callable
to accept UFL ``SpatialCoordinate`` objects and return a UFL expression.

The overrides here:

1. ``define_temperature``  – uses ``fem.Function.interpolate(callable)``
   instead of building a ``fem.Expression`` from a UFL expression.
2. ``update_time_dependent_values`` – re-interpolates the numpy-based
   temperature at each timestep, and adds a guard against updating past
   ``final_time`` (to avoid scenario overshoot).

Everything else (exports, ``post_processing``, ``Profile1DExport`` timing,
etc.) is handled by FESTIM 2.0 natively.
"""

import festim as F

import basix
import dolfinx.fem as fem
import numpy as np


class CustomProblem(F.HydrogenTransportProblem):
    """Thin wrapper around :class:`festim.HydrogenTransportProblem`.

    Only ``define_temperature`` and ``update_time_dependent_values`` are
    overridden; all export / post-processing logic is delegated to FESTIM.
    """

    def define_temperature(self):
        """Define temperature field from a numpy-based callable ``T(x, t)``.

        FESTIM's base implementation builds a ``fem.Expression`` from a UFL
        expression, but HISP temperature functions use numpy operations
        (``np.full_like``, conditionals, etc.) that cannot run on UFL
        ``SpatialCoordinate`` objects.  We therefore use
        ``fem.Function.interpolate(callable)`` instead.
        """
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        if isinstance(self.temperature, (float, int)):
            self.temperature_fenics = F.as_fenics_constant(
                self.temperature, self.mesh.mesh
            )

        elif isinstance(self.temperature, (fem.Constant, fem.Function)):
            self.temperature_fenics = self.temperature

        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), (float, int)):
                    raise ValueError(
                        "self.temperature should return a float or an int, not "
                        f"{type(self.temperature(t=float(self.t)))}"
                    )
                self.temperature_fenics = F.as_fenics_constant(
                    mesh=self.mesh.mesh, value=self.temperature(t=float(self.t))
                )
            else:
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

                # Store as a numpy-based callable so that the base-class
                # update_time_dependent_values can call
                #   self.temperature_fenics.interpolate(self.temperature_expr)
                # and dolfinx will pass numpy coordinates to the callable.
                self.temperature_expr = (
                    lambda x: self.temperature(x, float(self.t))
                )
                self.temperature_fenics.interpolate(self.temperature_expr)

    def update_time_dependent_values(self):
        """Update all time-dependent values.

        Adds a guard: once ``t > final_time`` we skip updates to avoid
        overshoot in scenario-derived flux / temperature look-ups.
        Then delegates to FESTIM's base implementation which handles
        BCs, sources, reactions, and temperature (via
        ``self.temperature_expr``).
        """
        if float(self.t) > self.settings.final_time:
            return

        super().update_time_dependent_values()
