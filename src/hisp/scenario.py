import pandas as pd
from typing import List
import warnings


class Pulse:
    pulse_type: str
    nb_pulses: int
    ramp_up: float
    steady_state: float
    ramp_down: float
    waiting: float

    def __init__(
        self,
        pulse_type: str,
        nb_pulses: int,
        ramp_up: float,
        steady_state: float,
        ramp_down: float,
        waiting: float,
        tritium_fraction: float,  # tritium fraction = T/D
    ):
        self.pulse_type = pulse_type
        self.nb_pulses = nb_pulses
        self.ramp_up = ramp_up
        self.steady_state = steady_state
        self.ramp_down = ramp_down
        self.waiting = waiting
        self.tritium_fraction = tritium_fraction

    @property
    def total_duration(self) -> float:
        all_zeros = (
            self.ramp_up == 0
            and self.steady_state == 0
            and self.ramp_down == 0
            and self.waiting == 0
        )
        if self.pulse_type == "RISP" and all_zeros:
            msg = "RISP pulse has all zeros for ramp_up, steady_state, ramp_down, waiting. "
            msg += "Setting hardcoded values. Please check the values in the scenario file."
            warnings.warn(msg, UserWarning)

            self.ramp_up = 10
            self.steady_state = 250
            self.ramp_down = 10
            self.waiting = 1530

        return self.ramp_up + self.steady_state + self.ramp_down + self.waiting

    @property
    def duration_no_waiting(self) -> float:
        return self.total_duration - self.waiting


class Scenario:
    def __init__(self, pulses: List[Pulse] = None):
        """Initializes a Scenario object containing several pulses.

        Args:
            pulses: The list of pulses in the scenario. Each pulse is a Pulse object.
        """
        self._pulses = pulses if pulses is not None else []

    @property
    def pulses(self) -> List[Pulse]:
        return self._pulses

    def to_txt_file(self, filename: str):
        df = pd.DataFrame(
            [
                {
                    "pulse_type": pulse.pulse_type,
                    "nb_pulses": pulse.nb_pulses,
                    "ramp_up": pulse.ramp_up,
                    "steady_state": pulse.steady_state,
                    "ramp_down": pulse.ramp_down,
                    "waiting": pulse.waiting,
                    "tritium_fraction": pulse.tritium_fraction,
                }
                for pulse in self.pulses
            ]
        )
        df.to_csv(filename, index=False)

    @staticmethod
    def from_txt_file(filename: str, old_format=False) -> "Scenario":
        if old_format:
            pulses = []
            with open(filename, "r") as f:
                for line in f:
                    # skip first line
                    if line.startswith("#"):
                        continue

                    # skip empty lines
                    if not line.strip():
                        continue

                    # assume this is the format
                    pulse_type, nb_pulses, ramp_up, steady_state, ramp_down, waiting = (
                        line.split()
                    )
                    pulses.append(
                        Pulse(
                            pulse_type=pulse_type,
                            nb_pulses=int(nb_pulses),
                            ramp_up=float(ramp_up),
                            steady_state=float(steady_state),
                            ramp_down=float(ramp_down),
                            waiting=float(waiting),
                        )
                    )
            return Scenario(pulses)
        df = pd.read_csv(filename)
        pulses = [
            Pulse(
                pulse_type=row["pulse_type"],
                nb_pulses=int(row["nb_pulses"]),
                ramp_up=float(row["ramp_up"]),
                steady_state=float(row["steady_state"]),
                ramp_down=float(row["ramp_down"]),
                waiting=float(row["waiting"]),
                tritium_fraction=float(row["tritium_fraction"]),
            )
            for _, row in df.iterrows()
        ]
        return Scenario(pulses)

    def get_row(self, t: float) -> int:
        """
        Returns the index of the pulse at time t.
        If t is greater than the maximum time in the scenario, a
        warning is raised and the last pulse index is returned.

        Args:
            t: the time in seconds

        Returns:
            the index of the pulse at time t
        """
        current_time = 0
        for i, pulse in enumerate(self.pulses):
            phase_duration = pulse.nb_pulses * pulse.total_duration
            if t < current_time + phase_duration:
                return i
            else:
                current_time += phase_duration

        warnings.warn(
            f"Time t {t} is out of bounds of the scenario file. Valid times are t < {self.get_maximum_time()}",
            UserWarning,
        )
        return i

    def get_pulse(self, t: float) -> Pulse:
        """
        Returns the pulse at time t.
        If t is greater than the maximum time in the scenario, a
        warning is raised and the last pulse is returned.

        Args:
            t: the time in seconds

        Returns:
            Pulse: the pulse at time t
        """
        row_idx = self.get_row(t)
        return self.pulses[row_idx]

    def get_pulse_type(self, t: float) -> str:
        """Returns the pulse type as a string at time t.

        Args:
            t: time in seconds

        Returns:
            pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
        """
        return self.get_pulse(t).pulse_type

    def get_maximum_time(self) -> float:
        """Returns the maximum time of the scenario in seconds.

        Returns:
            the maximum time of the scenario in seconds
        """
        return sum([pulse.nb_pulses * pulse.total_duration for pulse in self.pulses])

    def get_time_start_current_pulse(self, t: float):
        """Returns the time (s) at which the current pulse started.

        Args:
            t: the time in seconds

        Returns:
            the time at which the current pulse started
        """
        pulse_index = self.get_row(t)
        return sum(
            [
                pulse.nb_pulses * pulse.total_duration
                for pulse in self.pulses[:pulse_index]
            ]
        )

    # TODO this is the same as get_time_start_current_pulse, remove
    def get_time_till_row(self, row: int) -> float:
        """Returns the time (s) until the row in the scenario file.

        Args:
            row: the row index in the scenario file

        Returns:
            the time until the row in the scenario file
        """
        return sum(
            [pulse.nb_pulses * pulse.total_duration for pulse in self.pulses[:row]]
        )

    # TODO remove
    def get_pulse_duration_no_waiting(self, row: int) -> float:
        """Returns the total duration (without the waiting time) of a pulse in seconds for a given row in the file.

        Args:
            row: the row index in the scenario file

        Returns:
            the total duration of the pulse in seconds
        """
        return self.pulses[row].duration_no_waiting

    # TODO remove
    def get_pulse_duration(self, row: int) -> float:
        """Returns the total duration of a pulse in seconds for a given row in the file.

        Args:
            row: the row index in the scenario file

        Returns:
            the total duration of the pulse in seconds
        """
        return self.pulses[row].total_duration
