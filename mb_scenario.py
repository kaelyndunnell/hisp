# simple monoblock simulation in festim
import numpy as np
import matplotlib.pyplot as plt

from dolfinx.fem.function import Constant
import dolfinx
from numpy.typing import NDArray

from hisp.helpers import Scenario
from hisp import make_mb_model

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

NB_FP_PULSES_PER_DAY = 13
COOLANT_TEMP = 343  # 70 degree C cooling water

# tritium fraction = T/D
PULSE_TYPE_TO_TRITIUM_FRACTION = {
    "FP": 0.5,
    "ICWC": 0,
    "RISP": 0,
    "GDC": 0,
    "BAKE": 0,
}

# TODO change the dat file for other pulse types
pulse_type_to_DINA_data = {
    "FP": np.loadtxt("Binned_Flux_Data.dat", skiprows=1),
    "ICWC": np.loadtxt("ICWC_data.dat", skiprows=1),
    "GDC": np.loadtxt("GDC_data.dat", skiprows=1),
}


def RISP_data(nb_mb: int, t_rel: float | int) -> NDArray:
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
        label = "RISP"
        div = True
        offset_mb = 46
    elif nb_mb in outer_swept_bins:
        label = "ROSP"
        div = True
        offset_mb = 19
    else:
        div = False
        offset_mb = 0

    t_rel = int(t_rel)  # convert t_rel to an integer

    # NOTE: what is the point of this test since it takes nb_mb as an argument?
    if div:
        if 1 <= t_rel <= 9:
            data = np.loadtxt(f"{label}_data/time0.dat", skiprows=1)
        elif 10 <= t_rel <= 98:
            data = np.loadtxt(f"{label}_data/time10.dat", skiprows=1)
        elif 100 <= t_rel <= 260:
            data = np.loadtxt(f"{label}_data/time{t_rel}.dat", skiprows=1)
        elif 261 <= t_rel <= 269:
            data = np.loadtxt(f"{label}_data/time270.dat", skiprows=1)
        else:  # NOTE: so if time is too large a MB transforms into a FW element???
            data = np.loadtxt("RISP_Wall_data.dat", skiprows=1)
    else:
        data = np.loadtxt("RISP_Wall_data.dat", skiprows=1)

    return data[nb_mb - offset_mb, :]


def get_particle_flux(pulse_type: str, nb_mb: int, t_rel: float, ion=True) -> float:
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
        flux = pulse_type_to_DINA_data[pulse_type][:, FP_index][nb_mb - 1]
    elif pulse_type == "RISP":
        assert isinstance(t_rel, float), f"t_rel should be a float, not {type(t_rel)}"
        flux = RISP_data(nb_mb=nb_mb, t_rel=t_rel)[other_index]
    elif pulse_type == "BAKE":
        flux = 0.0
    else:
        flux = pulse_type_to_DINA_data[pulse_type][:, other_index][nb_mb - 1]

    return flux


def heat(nb_mb: int, pulse_type: str, t_rel: float) -> float:
    """Returns the surface heat flux (W/m2) for a given pulse type

    Args:
        nb_mb: monoblock number
        pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)
        t_rel: t_rel as an integer (in seconds).
            t_rel = t - t_pulse_start where t_pulse_start is the start of the pulse in seconds

    Raises:
        ValueError: if the pulse type is unknown

    Returns:
        the surface heat flux in W/m2
    """
    if pulse_type == "RISP":
        data = RISP_data(nb_mb, t_rel=t_rel)
    elif pulse_type in pulse_type_to_DINA_data.keys():
        data = pulse_type_to_DINA_data[pulse_type]
    else:
        raise ValueError(f"Invalid pulse type {pulse_type}")

    if pulse_type == "FP":
        heat_val = data[:, -2][nb_mb - 1]
    elif pulse_type == "RISP":
        heat_val = data[-1]
    else:
        heat_val = data[:, -1][nb_mb - 1]

    return heat_val


if __name__ == "__main__":
    my_scenario = Scenario("scenario_test.txt")

    nb_mb = 64
    L = 6e-3  # m

    def T_function(x: NDArray, t: Constant) -> float:
        """Monoblock temperature function.

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed monoblock temperature in K
        """
        resting_value = np.full_like(x[0], COOLANT_TEMP)
        pulse_row = my_scenario.get_row(float(t))
        pulse_type = my_scenario.get_pulse_type(float(t))
        t_rel = t - my_scenario.get_time_till_row(pulse_row)

        if pulse_type == "BAKE":
            T_bake = 483.15  # K
            flat_top_value = np.full_like(x[0], T_bake)
        else:
            heat_flux = heat(nb_mb, pulse_type, t_rel)
            T_surface = 1.1e-4 * heat_flux + COOLANT_TEMP
            T_rear = 2.2e-5 * heat_flux + COOLANT_TEMP
            a = (T_rear - T_surface) / L
            b = T_surface
            flat_top_value = a * x[0] + b

        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)

        relative_time = t - time_start_current_pulse
        return (
            flat_top_value
            if relative_time % total_time_pulse < total_time_on
            and relative_time % total_time_pulse != 0.0
            else resting_value
        )

    def deuterium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        ion_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * (1 - tritium_fraction)
        resting_value = 0

        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def tritium_ion_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        ion_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=True
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * tritium_fraction
        resting_value = 0.0

        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def deuterium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse
        atom_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )

        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * (1 - tritium_fraction)
        resting_value = 0.0
        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    def tritium_atom_flux(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"
        pulse_type = my_scenario.get_pulse_type(t)

        pulse_row = my_scenario.get_row(t)
        total_time_on = my_scenario.get_pulse_duration_no_waiting(pulse_row)
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        time_start_current_pulse = my_scenario.get_time_till_row(pulse_row)
        relative_time = t - time_start_current_pulse

        atom_flux = get_particle_flux(
            pulse_type=pulse_type, nb_mb=nb_mb, t_rel=relative_time, ion=False
        )
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * tritium_fraction
        resting_value = 0.0
        is_pulse_active = 0.0 < relative_time % total_time_pulse < total_time_on
        if is_pulse_active:
            return flat_top_value
        else:
            return resting_value

    my_model, quantities = make_mb_model(
        T_function=T_function,
        deuterium_ion_flux=deuterium_ion_flux,
        tritium_ion_flux=tritium_ion_flux,
        deuterium_atom_flux=deuterium_atom_flux,
        tritium_atom_flux=tritium_atom_flux,
        final_time=my_scenario.get_maximum_time(),
        L=6e-3,
        folder=f"mb{nb_mb}_results",
    )
    ############# Run Simu #############

    my_model.initialise()
    my_model.run()
    my_model.progress_bar.close()

    ############# Results Plotting #############

    for name, quantity in quantities.items():
        plt.plot(quantity.t, quantity.data, label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.yscale("log")

    plt.show()

    fig, ax = plt.subplots()

    ax.stackplot(
        quantity.t,
        [quantity.data for quantity in quantities.values()],
        labels=quantities.keys(),
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Total quantity (atoms/m2)")
    plt.legend()
    plt.show()
