import numpy as np
from numpy.typing import NDArray
from hisp.bin import SubBin, DivBin, FWBin
from hisp.helpers import periodic_step_function
from hisp.scenario import Pulse

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
        self, pulse: Pulse, bin: SubBin | DivBin, t_rel: float, ion=True
    ) -> float:
        """Returns the particle flux for a given pulse type

        Args:
            pulse: the pulse object
            bin: SubBin or DivBin
            t_rel: Relative time (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds
            ion (bool, optional): _description_. Defaults to True.

        Returns:
            float: particle flux in part/m2/s
        """
        if isinstance(bin, SubBin):
            bin_index = bin.parent_bin_index
            wetted_frac = bin.wetted_frac  # frac of wettedness for subbin
        elif isinstance(bin, DivBin):
            bin_index = bin.index
            wetted_frac = 1  # all div bins are wetted, so get full flux

        if ion:
            FP_index = 2
            other_index = 0
            flux_frac = (
                wetted_frac  # for an ion flux, apply the wetted frac for this bin
            )
        if not ion:
            FP_index = 3
            other_index = 1
            flux_frac = 1  # there is no wettedness for atom fluxes -- every subbin / bin gets all the atom flux

        if pulse.pulse_type == "FP":
            flux = self.pulse_type_to_data[pulse.pulse_type][:, FP_index][bin_index]
        elif pulse.pulse_type == "RISP":
            assert isinstance(
                t_rel, float
            ), f"t_rel should be a float, not {type(t_rel)}"

            t_rel_within_a_single_risp = t_rel % pulse.total_duration

            data = self.RISP_data(bin, t_rel=t_rel_within_a_single_risp)

            flux = data[other_index]
        elif pulse.pulse_type == "BAKE":
            flux = 0.0
        else:
            flux = self.pulse_type_to_data[pulse.pulse_type][:, other_index][bin_index]

        value = flux * flux_frac

        # add in the step function for the pulse
        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration

        return periodic_step_function(
            t_rel,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=value,
            value_off=0,
        )

    def RISP_data(self, bin: SubBin | DivBin, t_rel: float | int) -> NDArray:
        """Returns the correct RISP data file for indicated bin

        Args:
            bin: Subbin or Divbin object
            t_rel: relative time (in seconds).
                t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

        Returns:
            data: data from correct file as a numpy array
        """
        assert isinstance(
            bin, (SubBin, DivBin)
        ), f"bin should be a SubBin or DivBin, not {type(bin)}"

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
        else:
            offset_mb = 0

        t_rel = int(t_rel)
        # NOTE: what is the point of this test since it takes bin_index as an argument?
        if div and offset_mb != 0:
            if 0 <= t_rel <= 9:
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
            elif 261 <= t_rel <= 270:
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

    def get_heat(self, pulse: Pulse, bin: SubBin | DivBin, t_rel: float):
        """Returns the surface heat flux (W/m2) for a given pulse type

        Args:
            pulse: the pulse object
            bin: SubBin or DivBin
            t_rel: Relative time (in seconds).
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

        if pulse.pulse_type == "RISP":
            t_rel_within_a_single_risp = t_rel % pulse.total_duration
            data = self.RISP_data(bin, t_rel=t_rel_within_a_single_risp)
        elif pulse.pulse_type in self.pulse_type_to_data.keys():
            data = self.pulse_type_to_data[pulse.pulse_type]
        else:
            raise ValueError(f"Invalid pulse type {pulse.pulse_type}")

        if pulse.pulse_type == "FP":
            heat_total = data[:, -2][bin_index]
            heat_ion = data[:, -1][bin_index]
            if isinstance(bin, SubBin):
                heat_val = heat_total - heat_ion * (1 - bin.wetted_frac)
            else:
                heat_val = heat_total
        elif pulse.pulse_type == "RISP":
            if isinstance(bin, SubBin):
                heat_total = data[-2]
                heat_ion = data[-1]
                heat_val = heat_total - heat_ion * (1 - bin.wetted_frac)
            else:
                heat_val = data[-1]
        else:  # currently no heat for other pulse types
            heat_val = data[:, -1][bin_index]

        # add in the step function for the pulse
        total_time_on = pulse.duration_no_waiting
        total_time_pulse = pulse.total_duration

        return periodic_step_function(
            t_rel,
            period_on=total_time_on,
            period_total=total_time_pulse,
            value=heat_val,
            value_off=0,
        )
