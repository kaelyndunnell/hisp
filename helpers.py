import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np


class PulsedSource(F.ParticleSource):
    def __init__(self, flux, distribution, volume, species):
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


class Scenario:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = np.genfromtxt(filename, dtype=str, comments="#")

    def get_row(self, t: float):
        """Returns the row of the scenario file that corresponds to the time t.

        Args:
            t (float): the time in seconds

        Returns:
            int: the row index of the scenario file corresponding to the time t
        """
        current_time = 0
        for i, row in enumerate(self.data):
            nb_pulses = int(row[1])
            phase_duration = nb_pulses * self.get_pulse_duration(i)
            if t <= current_time + phase_duration:
                return i
            else:
                current_time += phase_duration

        raise ValueError("Time t is out of bounds of the scenario file")

    def get_pulse_type(self, t: float) -> str:
        """Returns the pulse type as a string at time t.

        Args:
            t (float): time in seconds

        Returns:
            str: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
        """
        row_idx = self.get_row(t)
        return self.data[row_idx][0]

    def get_pulse_duration(self, row: int) -> float:
        """Returns the total duration of a pulse in seconds for a given row in the file.

        Args:
            row (int): the row index in the scenario file

        Returns:
            float: the total duration of the pulse in seconds
        """
        row_data = self.data[row]
        pulse_type = row_data[0]
        if pulse_type == "RISP":  # hard coded because it's zero in the files
            ramp_up = 10
            steady_state = 250
            ramp_down = 10
            waiting = 1530
            total_duration = ramp_up + steady_state + ramp_down + waiting
            return total_duration

        ramp_up = float(row_data[2])
        steady_state = float(row_data[4])
        ramp_down = float(row_data[3])
        waiting = float(row_data[5])

        total_duration = ramp_up + steady_state + ramp_down + waiting
        return total_duration

    def get_maximum_time(self) -> float:
        """Returns the maximum time in seconds for the scenario file.

        Returns:
            float: the maximum time in seconds
        """
        max_time = 0
        for i, row in enumerate(self.data):
            nb_pulses = int(row[1])
            max_time += nb_pulses * self.get_pulse_duration(i)
        return max_time
