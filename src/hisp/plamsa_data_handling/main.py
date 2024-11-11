import numpy as np
from numpy.typing import NDArray
import pandas as pd

from typing import Dict
from fw_sub_bins import (
    sub_2_bins,
    sub_3_bins,
    fw_bins,
    high_w_6mm,
    low_w_6mm,
    shadow_w_6mm,
    w_10mm,
    high_w_12mm,
    low_w_12mm,
    shadow_w_12mm,
    b_1um,
    b_100nm,
    b_5um,
    ss_5mm,
)


def read_wetted_data(filename, nb_mb):
    """Reads wetted/shadowed data from csv file for first wall.

    Args:
        filename (str): filename of csv file with wetted FW data
        monoblock (int): monoblock number

    Returns:
        Slow/Shigh, Stot, f, DFW for nb

    """

    data = pd.read_csv(filename, skiprows=1, names=range(5))
    data = data.to_numpy()
    return data[nb_mb - 1]


def compute_wetted_frac(
    nb_mb, Slow, Stot, Shigh, f, low_wet=False, high_wet=False, shadowed=False
):
    """Computes fraction of wetted-ness for first wall sub-bins.

    Args:
        nb_mb (int): monoblock number
        Slow (float): surface area of low wetted area.
        Stot (float): total surface area of bin.
        Shigh (float): surface area of high wetted area.
        f (float): fraction of heat values in low wetted area from SMITER csv files.
        low_wet (Boolean): True if solving for low wetted bin.
        high_wet (Boolean): True if solving for high wetted bin.
        shadowed (Boolean): True if solving for shadowed bin.

    Returns:
        frac: fraction of wetted-ness for sub-bin.

    """
    if nb_mb in sub_3_bins:
        if low_wet:
            frac = f * Stot / Slow
        elif high_wet:
            frac = (1 - f) * Stot / Shigh
        elif shadowed:
            frac = 0.0

    elif nb_mb in sub_2_bins:
        if low_wet:
            frac = Stot / Slow
        elif shadowed:
            frac = 0.0

    else:  # div blocks
        frac = 1

    return frac

# TODO: make this an optional class that requires specification from the user? 
# TODO: add tests for find_length
def find_length(nb_mb, section=None):
    """Finds length and material of given monoblock.

    Args:
        nb_mb (int): monoblock number
        section (str): for fw_bins, monoblock section. can be either "high_wetted", "low_wetted", "shadowed", or "dfw".

    Returns:
        length (float): length of given nb_mb.
        material (str): material of given nb_mb.

    """
    if section not in ["high_wetted", "low_wetted", "shadowed", "dfw", None]: 
        raise ValueError("Section must be either 'high_wetted', 'low-wetted', 'shadowed', 'dfw', or None.")

    if nb_mb in fw_bins:
        if section == "high_wetted":
            if nb_mb in high_w_6mm:
                length = 6e-3  # m

            elif nb_mb in w_10mm:
                length = 10e-3

            elif nb_mb in high_w_12mm:
                length = 12e-3

            material = "W"

        elif section == "low_wetted":
            if nb_mb in low_w_6mm:
                length = 6e-3  # m
                material = "W"

            elif nb_mb in w_10mm:
                length = 10e-3
                material = "W"

            elif nb_mb in low_w_12mm:
                length = 12e-3
                material = "W"

            elif nb_mb in b_100nm:
                length = 100e-9
                material = "B"

        elif section == "dfw": 
            length = 5e-3
            material = "SS"

        else:
            if nb_mb in shadow_w_6mm:
                length = 6e-3  # m
                material = "W"

            elif nb_mb in w_10mm:
                length = 10e-3
                material = "W"

            elif nb_mb in shadow_w_12mm:
                length = 12e-3
                material = "W"

            elif nb_mb in b_1um:
                length = 1e-6
                material = "B"

    else:
        if nb_mb in high_w_6mm:
            length = 6e-3  # m
            material = "W"

        elif nb_mb in b_1um:
            length = 1e-6
            material = "B"

        elif nb_mb in b_5um:
            length = 5e-6
            material = "B"

    return length, material

class PlasmaDataHandling:
    def __init__(
        self,
        pulse_type_to_data: Dict[str, NDArray],
        path_to_RISP_data: str,
        path_to_ROSP_data: str,
        path_to_RISP_wall_data: str,
    ):
        self.pulse_type_to_data = pulse_type_to_data
        self.path_to_RISP_data = path_to_RISP_data
        self.path_to_ROSP_data = path_to_ROSP_data
        self.path_to_RISP_wall_data = path_to_RISP_wall_data

        self._time_to_RISP_data = {}

    def get_particle_flux(
        self, pulse_type: str, nb_mb: int, t_rel: float, ion=True
    ) -> float:
        """Returns the particle flux for a given pulse type

        Args:
            pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
            nb_mb: monoblock number
            t_rel: t_rel as an integer (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds
            ion (bool, optional): _description_. Defaults to True.

        Returns:
            float: particle flux in part/m2/s
        """
        if ion:
            FP_index = 2
            other_index = 0
        if not ion:
            FP_index = 3
            other_index = 1

        if pulse_type == "FP":
            flux = self.pulse_type_to_data[pulse_type][:, FP_index][nb_mb - 1]
        elif pulse_type == "RISP":
            assert isinstance(
                t_rel, float
            ), f"t_rel should be a float, not {type(t_rel)}"
            flux = self.RISP_data(nb_mb=nb_mb, t_rel=t_rel)[other_index]
        elif pulse_type == "BAKE":
            flux = 0.0
        else:
            flux = self.pulse_type_to_data[pulse_type][:, other_index][nb_mb - 1]

        return flux

    def RISP_data(self, nb_mb: int, t_rel: float | int) -> NDArray:
        """Returns the correct RISP data file for indicated monoblock

        Args:
            nb_mb: mb number
            t_rel: t_rel as an integer(in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

        Returns:
            data: data from correct file as a numpy array
        """
        inner_swept_bins = list(range(46, 65))
        outer_swept_bins = list(range(19, 34))

        if nb_mb in inner_swept_bins:
            folder = self.path_to_RISP_data
            div = True
            offset_mb = 46
        elif nb_mb in outer_swept_bins:
            folder = self.path_to_ROSP_data
            div = True
            offset_mb = 19
        else:
            div = False
            offset_mb = 0

        t_rel = int(t_rel)
        # NOTE: what is the point of this test since it takes nb_mb as an argument?
        if div:
            if 1 <= t_rel <= 9:
                key = f"{folder}_1_9"
                if key not in self._time_to_RISP_data.keys():
                    self._time_to_RISP_data[key] = np.loadtxt(
                        f"{folder}/time0.dat", skiprows=1
                    )
                data = self._time_to_RISP_data[key]
            elif 10 <= t_rel <= 98:
                key = f"{folder}_10_98"
                if key not in self._time_to_RISP_data.keys():
                    self._time_to_RISP_data[key] = np.loadtxt(
                        f"{folder}/time10.dat", skiprows=1
                    )
                data = self._time_to_RISP_data[key]
            elif 100 <= t_rel <= 260:
                key = f"{folder}_{t_rel}"
                if key not in self._time_to_RISP_data.keys():
                    self._time_to_RISP_data[key] = np.loadtxt(
                        f"{folder}/time{t_rel}.dat", skiprows=1
                    )
                data = self._time_to_RISP_data[key]
            elif 261 <= t_rel <= 269:
                key = f"{folder}_261_269"
                if key not in self._time_to_RISP_data.keys():
                    self._time_to_RISP_data[key] = np.loadtxt(
                        f"{folder}/time260.dat", skiprows=1
                    )
                data = self._time_to_RISP_data[key]
            else:  # NOTE: so if time is too large a MB transforms into a FW element???
                key = "wall_data"
                if key not in self._time_to_RISP_data.keys():
                    self._time_to_RISP_data[key] = np.loadtxt(
                        self.path_to_RISP_wall_data, skiprows=1
                    )
                data = self._time_to_RISP_data[key]
        else:
            key = "wall_data"
            if key not in self._time_to_RISP_data.keys():
                self._time_to_RISP_data[key] = np.loadtxt(
                    self.path_to_RISP_wall_data, skiprows=1
                )
            data = self._time_to_RISP_data[key]

        return data[nb_mb - offset_mb, :]

    def heat(self, pulse_type: str, nb_mb: int, t_rel: float) -> float:
        """Returns the surface heat flux (W/m2) for a given pulse type

        Args:
            pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
            nb_mb: monoblock number
            t_rel: t_rel as an integer (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

        Raises:
            ValueError: if the pulse type is unknown

        Returns:
            the surface heat flux in W/m2
        """
        if pulse_type == "RISP":
            data = self.RISP_data(nb_mb, t_rel=t_rel)
        elif pulse_type in self.pulse_type_to_data.keys():
            data = self.pulse_type_to_data[pulse_type]
        else:
            raise ValueError(f"Invalid pulse type {pulse_type}")

        if pulse_type == "FP":
            heat_val = data[:, -2][nb_mb - 1]
            heat_ion = data[:, -1][nb_mb - 1]
        elif pulse_type == "RISP":
            if nb_mb in fw_bins:
                heat_val = data[-2]
                heat_ion = data[-1]
            else:
                heat_val = data[-1]
                heat_ion = 0.0
        else:
            heat_val = data[:, -1][nb_mb - 1]
            heat_ion = 0.0

        return heat_val, heat_ion
