import numpy as np
from numpy.typing import NDArray
import pandas as pd
from bin import SubBin, DivBin, FWBin

from typing import Dict


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
        self, pulse_type: str, bin: SubBin | DivBin, t_rel: float, ion=True
    ) -> float:
        """Returns the particle flux for a given pulse type

        Args:
            pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
            bin: SubBin or DivBin
            t_rel: t_rel as an integer (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds
            ion (bool, optional): _description_. Defaults to True.

        Returns:
            float: particle flux in part/m2/s
        """
        if isinstance(bin, SubBin):
            bin_index = bin.parent_bin_index
            wetted_frac = bin.wetted_frac
        elif isinstance(bin, DivBin):
            bin_index = bin.index
            wetted_frac = 1

        if ion:
            FP_index = 2
            other_index = 0
            flux_frac = wetted_frac
        if not ion:
            FP_index = 3
            other_index = 1
            flux_frac = 1

        if pulse_type == "FP":
            flux = self.pulse_type_to_data[pulse_type][:, FP_index][bin_index]
        elif pulse_type == "RISP":
            assert isinstance(
                t_rel, float
            ), f"t_rel should be a float, not {type(t_rel)}"
            flux = self.RISP_data(bin_index=bin_index, t_rel=t_rel)[other_index]
        elif pulse_type == "BAKE":
            flux = 0.0
        else:
            flux = self.pulse_type_to_data[pulse_type][:, other_index][bin_index]

        return flux * flux_frac

    def RISP_data(self, bin: SubBin | DivBin, t_rel: float | int) -> NDArray:
        """Returns the correct RISP data file for indicated bin

        Args:
            bin: Subbin or Divbin object
            t_rel: t_rel as an integer(in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

        Returns:
            data: data from correct file as a numpy array
        """
        if isinstance(bin, SubBin):
            bin_index = bin.parent_bin_index
            div = False
        elif isinstance(bin, DivBin):
            bin_index = bin.index
            div = True

        if div:
            if bin.inner_bin:
                folder = self.path_to_RISP_data
                offset_mb = 45
            elif bin.outer_bin:
                folder = self.path_to_ROSP_data
                offset_mb = 18
        else:
            offset_mb = 0

        t_rel = int(t_rel)
        # NOTE: what is the point of this test since it takes bin_index as an argument?
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

        return data[bin_index - offset_mb, :]

    def get_heat(self, pulse_type: str, bin: SubBin | DivBin, t_rel: float):
        """Returns the surface heat flux (W/m2) for a given pulse type

        Args:
            pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
            bin: SubBin or DivBin
            t_rel: t_rel as an integer (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

        Raises:
            ValueError: if the pulse type is unknown

        Returns:
            the surface heat flux in W/m2
        """
        if isinstance(bin, SubBin):
            bin_index = bin.parent_bin_index
        elif isinstance(bin, DivBin):
            bin_index = bin.index

        if pulse_type == "RISP":
            data = self.RISP_data(bin_index, t_rel=t_rel)
        elif pulse_type in self.pulse_type_to_data.keys():
            data = self.pulse_type_to_data[pulse_type]
        else:
            raise ValueError(f"Invalid pulse type {pulse_type}")

        if pulse_type == "FP":
            heat_total = data[:, -2][bin_index]
            heat_ion = data[:, -1][bin_index]
            if isinstance(bin, SubBin):
                heat_val = heat_total - heat_ion * (1 - bin.wetted_frac)
            else:
                heat_val = heat_total
        elif pulse_type == "RISP":
            if isinstance(bin, SubBin):
                heat_total = data[-2]
                heat_ion = data[-1]
                heat_val = heat_total - heat_ion * (1 - bin.wetted_frac)
            else:
                heat_val = data[-1]
        else:  # currently no heat for other pulse types
            heat_val = data[:, -1][bin_index]

        return heat_val
