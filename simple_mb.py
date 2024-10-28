# simple monoblock simulation in festim
import festim as F
import numpy as np
import matplotlib.pyplot as plt

import h_transport_materials as htm
import ufl
from dolfinx.fem.function import Constant
from scipy import constants
import dolfinx.fem as fem
import dolfinx

from helpers import PulsedSource, Scenario
from new_h_transport_class import CustomProblem

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

NB_FP_PULSES_PER_DAY = 13
COOLANT_TEMP = 343  # 70 degree C cooling water

############# CUSTOM CLASSES FOR PULSED FLUXES & RECOMBO BC #############

# TODO: ADJUST TO HANDLE ANY STRAIGHT W 6MM SIMU
mb = 50


# tritium fraction = T/D
PULSE_TYPE_TO_TRITIUM_FRACTION = {
    "FP": 0.5,
    "ICWC": 0,
    "RISP": 0,
    "GDC": 0,
    "BAKE": 0,
}


def gaussian_distribution(x, mod=ufl):
    depth = 3e-9
    width = 1e-9
    return mod.exp(-((x[0] - depth) ** 2) / (2 * width**2))


def make_mb_model(nb_mb):
    ############# Input Flux, Heat Data #############
    my_scenario = Scenario("scenario_test.txt")  # TODO make the filename a parameter

    my_model = CustomProblem()

    ############# Material Parameters #############

    L = 6e-3  # m
    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, 3e-6, num=300),
            np.linspace(3e-6, 30e-6, num=200),
            np.linspace(30e-6, L, num=200),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # W material parameters
    w_density = 6.3382e28  # atoms/m3
    w_diffusivity = htm.diffusivities.filter(material="tungsten").filter(isotope="h").filter(author="frauenfelder")
    w_diffusivity = w_diffusivity[0]
    D_0=w_diffusivity.pre_exp.magnitude
    E_D=w_diffusivity.act_energy.magnitude
    tungsten = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="tungsten",
    )

    # mb subdomains
    w_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [
        w_subdomain,
        inlet,
        outlet,
    ]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    trap3_D = F.Species("trap3_D", mobile=False)
    trap3_T = F.Species("trap3_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.338e24,  # 1e-4 at.fr.
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=6.338e24,
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    # density_func =
    empty_trap3 = F.ImplicitSpecies(  # fermi-dirac-like trap 3
        n=6.338e27,  # density_func # 1e-1 at.fr.
        others=[trap3_T, trap3_D],
        name="empty_trap3",
    )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
        trap2_D,
        trap2_T,
        trap3_D,
        trap3_T,
    ]

    # hydrogen reactions - 1 per trap per species
    my_model.reactions = [
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.5,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=D_0 / (1.1e-10**2 * 6 * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.5,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
    ]

    ############# Pulse Parameters (s) #############

    # TODO change the dat file for other pulse types
    pulse_type_to_DINA_data = {
        "FP": np.loadtxt("Binned_Flux_Data.dat", skiprows=1),
        "ICWC": np.loadtxt("ICWC_Data.dat", skiprows=1),
        "RISP": np.loadtxt("Binned_Flux_Data.dat", skiprows=1),
        "GDC": np.loadtxt("GDC_Data.dat", skiprows=1),
        "BAKE": np.loadtxt("Binned_Flux_Data.dat", skiprows=1),
    }

    # flat_top_duration = 50 * NB_FP_PULSES_PER_DAY
    # ramp_up_duration = 33 * NB_FP_PULSES_PER_DAY
    # ramp_down_duration = 35 * NB_FP_PULSES_PER_DAY
    # dwelling_time = 72000  # 20 hours

    # total_time_pulse = flat_top_duration + ramp_up_duration + ramp_down_duration
    total_time_cycle = my_scenario.get_maximum_time()

    ############# Temperature Parameters (K) #############

    def heat(pulse_type: str) -> float:
        """Returns the surface heat flux for a given pulse type

        Args:
            pulse_type: pulse type (eg. FP, ICWC, RISP, GDC, BAKE)

        Raises:
            ValueError: if the pulse type is unknown

        Returns:
            the surface heat flux in W/m2
        """
        if pulse_type not in ["FP", "ICWC", "RISP", "GDC", "BAKE"]:
            raise ValueError(f"Invalid pulse type {pulse_type}")
        data = pulse_type_to_DINA_data[pulse_type]
        return data[:, -2][nb_mb - 1]

    def T_surface(t: dolfinx.fem.Constant) -> float:
        """Monoblock surface temperature

        Args:
            t: time in seconds

        Returns:
            monoblock surface temperature in K
        """
        pulse_type = my_scenario.get_pulse_type(float(t))
        return 1.1e-4 * heat(pulse_type) + COOLANT_TEMP

    def T_rear(t: dolfinx.fem.Constant):
        """Monoblock surface temperature

        Args:
            t: time in seconds

        Returns:
            monoblock surface temperature in K
        """
        pulse_type = my_scenario.get_pulse_type(float(t))
        return 2.2e-5 * heat(pulse_type) + COOLANT_TEMP

    def T_function(x, t: Constant):
        """Monoblock temperature function

        Args:
            x: position along monoblock
            t: time in seconds

        Returns:
            pulsed monoblock temperature in K
        """
        a = (T_rear(t) - T_surface(t)) / L
        b = T_surface(t)
        flat_top_value = a * x[0] + b
        resting_value = COOLANT_TEMP
        pulse_row = my_scenario.get_row(float(t))
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        return (
            flat_top_value
            if float(t) % total_time_cycle < total_time_pulse
            else resting_value
        )

    # times = np.linspace(0, 10 * total_time_cycle, num=10000)

    # x = [0]
    # Ts = [T_function(x, t) for t in times]
    # import matplotlib.pyplot as plt

    # plt.plot(times, Ts, marker="o")
    # plt.show()
    # exit()

    my_model.temperature = T_function

    ############# Flux Parameters #############

    def deuterium_ion_flux(t: float):
        pulse_type = my_scenario.get_pulse_type(float(t))
        ion_flux = pulse_type_to_DINA_data[pulse_type][:, 2][nb_mb - 1]
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * (1 - tritium_fraction)
        resting_value = 0
        pulse_row = my_scenario.get_row(float(t))
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        return (
            flat_top_value
            if float(t) % total_time_cycle < total_time_pulse
            else resting_value
        )
    
    def tritium_ion_flux(t: float):
        pulse_type = my_scenario.get_pulse_type(float(t))
        ion_flux = pulse_type_to_DINA_data[pulse_type][:, 2][nb_mb - 1]
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = ion_flux * tritium_fraction
        resting_value = 0
        pulse_row = my_scenario.get_row(float(t))
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        return (
            flat_top_value
            if float(t) % total_time_cycle < total_time_pulse
            else resting_value
        )

    def deuterium_atom_flux(t: float):
        pulse_type = my_scenario.get_pulse_type(float(t))
        atom_flux = pulse_type_to_DINA_data[pulse_type][:, 3][nb_mb - 1]
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * (1 - tritium_fraction)
        resting_value = 0
        pulse_row = my_scenario.get_row(float(t))
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        return (
            flat_top_value
            if float(t) % total_time_cycle < total_time_pulse
            else resting_value
        )

    def tritium_atom_flux(t: float):
        pulse_type = my_scenario.get_pulse_type(float(t))
        atom_flux = pulse_type_to_DINA_data[pulse_type][:, 3][nb_mb - 1]
        tritium_fraction = PULSE_TYPE_TO_TRITIUM_FRACTION[pulse_type]
        flat_top_value = atom_flux * tritium_fraction
        resting_value = 0
        pulse_row = my_scenario.get_row(float(t))
        total_time_pulse = my_scenario.get_pulse_duration(pulse_row)
        return (
            flat_top_value
            if float(t) % total_time_cycle < total_time_pulse
            else resting_value
        )

    my_model.sources = [
        PulsedSource(
            flux=deuterium_ion_flux,
            distribution=gaussian_distribution,
            species=mobile_D,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            distribution=gaussian_distribution,
            species=mobile_T,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            distribution=gaussian_distribution,
            species=mobile_D,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            distribution=gaussian_distribution,
            species=mobile_T,
            volume=w_subdomain,
        ),
    ]

    ############# Boundary Conditions #############
    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_D],
        gas_pressure=0,
        k_r0=7.94e-17,
        E_kr=-2,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_tt = F.SurfaceReactionBC(
        reactant=[mobile_T, mobile_T],
        gas_pressure=0,
        k_r0=7.94e-17,
        E_kr=-2,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_dt = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_T],
        gas_pressure=0,
        k_r0=7.94e-17,
        E_kr=-2,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    my_model.boundary_conditions = [
        surface_reaction_dd,
        surface_reaction_dt,
        surface_reaction_tt,
    ]

    ############# Exports #############

    folder = f"mb{mb}_results"

    my_model.exports = [
        F.VTXExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
        F.VTXExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
        F.VTXExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
        F.VTXExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
        F.VTXExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
        F.VTXExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
        F.VTXExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
        F.VTXExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
    ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=w_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity

    ############# Settings #############

    my_model.settings = F.Settings(
        atol=1e-15,
        rtol=1e-15,
        max_iterations=1000,
        final_time=my_scenario.get_maximum_time(),
        # final_time=3000,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=20)

    return my_model, quantities


my_model, quantities = make_mb_model(nb_mb=mb)

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

# make the same but with a stack plot

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
