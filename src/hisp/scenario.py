import pandas as pd
from typing import List

class Pulse:
    pulse_type: str
    nb_pulses: int
    ramp_up: float
    steady_state: float
    ramp_down: float
    waiting: float

    def __init__(self, pulse_type: str, nb_pulses: int, ramp_up: float, steady_state: float, ramp_down: float, waiting: float):
        self.pulse_type = pulse_type
        self.nb_pulses = nb_pulses
        self.ramp_up = ramp_up
        self.steady_state = steady_state
        self.ramp_down = ramp_down
        self.waiting = waiting
    
    @property
    def total_duration(self) -> float:
        return self.ramp_up + self.steady_state + self.ramp_down + self.waiting

class Scenario:
    def __init__(self, pulses = None):
        self._pulses = pulses if pulses is not None else []
    
    @property
    def pulses(self) -> List[Pulse]:
        return self._pulses

    def to_txt_file(self, filename: str):
        df = pd.DataFrame([{
            "pulse_type": pulse.pulse_type,
            "nb_pulses": pulse.nb_pulses,
            "ramp_up": pulse.ramp_up,
            "steady_state": pulse.steady_state,
            "ramp_down": pulse.ramp_down,
            "waiting": pulse.waiting
        } for pulse in self.pulses])
        df.to_csv(filename, index=False)

    @staticmethod
    def from_txt_file(filename: str):
        df = pd.read_csv(filename)
        pulses = [
            Pulse(
                pulse_type=row["pulse_type"],
                nb_pulses=int(row["nb_pulses"]),
                ramp_up=float(row["ramp_up"]),
                steady_state=float(row["steady_state"]),
                ramp_down=float(row["ramp_down"]),
                waiting=float(row["waiting"]),
            )
            for _, row in df.iterrows()
        ]
        return Scenario(pulses)
