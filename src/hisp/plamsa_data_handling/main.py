import numpy as np
from numpy.typing import NDArray

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
                data = np.loadtxt(f"{folder}/time0.dat", skiprows=1)
            elif 10 <= t_rel <= 98:
                data = np.loadtxt(f"{folder}/time10.dat", skiprows=1)
            elif 100 <= t_rel <= 260:
                data = np.loadtxt(f"{folder}/time{t_rel}.dat", skiprows=1)
            elif 261 <= t_rel <= 269:
                data = np.loadtxt(f"{folder}/time270.dat", skiprows=1)
            else:  # NOTE: so if time is too large a MB transforms into a FW element???
                data = np.loadtxt(self.path_to_RISP_wall_data, skiprows=1)
        else:
            data = np.loadtxt(self.path_to_RISP_wall_data, skiprows=1)

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
        elif pulse_type == "RISP":
            heat_val = data[-1]
        else:
            heat_val = data[:, -1][nb_mb - 1]

        return heat_val
