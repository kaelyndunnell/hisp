import festim as F
from collections.abc import Callable
import dolfinx
import numpy.typing as npt


class CustomSettings(F.Settings):
    """Custom Settings for a festim Boron simulation.

    Args:
        atol (float, Callable): Absolute tolerance for the solver.
        rtol (float, Callable): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the
            solver. Defaults to 30.
        transient (bool, optional): Whether the simulation is transient or not.
        final_time (float, optional): Final time for a transient simulation.
            Defaults to None
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation. Defaults to None

    Attributes:
        atol (float, Callable): Absolute tolerance for the solver.
        rtol (float, Callable): Relative tolerance for the solver.
        max_iterations (int): Maximum number of iterations for the solver.
        transient (bool): Whether the simulation is transient or not.
        final_time (float): Final time for a transient simulation.
        stepsize (festim.Stepsize): stepsize for a transient
            simulation.
    """

    def __init__(
        self,
        atol: (
            float
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type]],
                npt.NDArray[dolfinx.default_scalar_type],
            ]
            | None
        ) = None,
        rtol: (
            float
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type]],
                npt.NDArray[dolfinx.default_scalar_type],
            ]
            | None
        ) = None,
        max_iterations=30,
        transient=True,
        final_time=None,
        stepsize=None,
        element_degree= 1,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_iterations = max_iterations
        self.transient = transient
        self.final_time = final_time
        self.stepsize = stepsize
        self.element_degree = element_degree

    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value is None:
            self._stepsize = None
        elif isinstance(value, (float, int)):
            self._stepsize = F.Stepsize(initial_value=value)
        elif isinstance(value, F.Stepsize):
            self._stepsize = value
        else:
            raise TypeError("stepsize must be an of type int, float or festim.Stepsize")
        
    @property
    def rtol(self):
        return self._rtol
    
    @rtol.setter
    def rtol(self, value):
        if value is None:
            self._rtol = value
        elif isinstance(value, float):
            self._rtol = value
        elif callable(value):
            self._rtol = value
        else:
            raise TypeError(
                "Value must be a float, or callable"
            )
        
    @property
    def rtol_time_dependent(self):
        if self.rtol is None:
            return False
        if callable(self.rtol):
            arguments = self.rtol.__code__.co_varnames
            return "t" in arguments
        else:
            return False
        
    @property
    def atol(self):
        return self._atol
    
    @atol.setter
    def atol(self, value):
        if value is None:
            self._atol = value
        elif isinstance(value, float):
            self._atol = value
        elif callable(value):
            self._atol = value
        else:
            raise TypeError(
                "Value must be a float, or callable"
            )
        
    @property
    def atol_time_dependent(self):
        if self.atol is None:
            return False
        if callable(self.atol):
            arguments = self.atol.__code__.co_varnames
            return "t" in arguments
        else:
            return False
        
    
    @property
    def element_degree(self) -> int | None:
        return getattr(self, "_element_degree", None)

    @element_degree.setter
    def element_degree(self, value: int | None):
        if value is None:
            self._element_degree = None
        elif isinstance(value, int) and value >= 0:
            self._element_degree = value
        else:
            raise TypeError("element_degree must be None or a non-negative int")
