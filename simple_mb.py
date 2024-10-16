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

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

############# CUSTOM CLASSES FOR PULSED FLUXES & RECOMBO BC #############

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

# TODO: ADJUST TO HANDLE ANY STRAIGHT W 6MM SIMU
mb = 50

############# Input Flux, Heat Data #############
lines = np.genfromtxt('scenario.txt', dtype=str, comments='#')

DINA_data = np.loadtxt('Binned_Flux_Data.dat', skiprows=1)
ion_flux = DINA_data[:,2][mb-1]
atom_flux = DINA_data[:,3][mb-1]
heat = DINA_data[:,-2][mb-1]

my_model = F.HydrogenTransportProblem()

############# Material Parameters #############

L = 6e-3 # m
vertices = np.concatenate( # 1D mesh with extra refinement
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
tungsten = F.Material(
    D_0=w_diffusivity.pre_exp.magnitude,
    E_D=w_diffusivity.act_energy.magnitude,
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
    n=6.338e27, # density_func # 1e-1 at.fr.
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
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.20,
        p_0=1e13,
        E_p=0.85,
        volume=w_subdomain,
        reactant=[mobile_D, empty_trap1],
        product=trap1_D,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.2,
        p_0=1e13,
        E_p=0.85,
        volume=w_subdomain,
        reactant=[mobile_T, empty_trap1],
        product=trap1_T,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.2,
        p_0=1e13,
        E_p=1,
        volume=w_subdomain,
        reactant=[mobile_D, empty_trap2],
        product=trap2_D,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.2,
        p_0=1e13,
        E_p=1,
        volume=w_subdomain,
        reactant=[mobile_T, empty_trap2],
        product=trap2_T,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.2,
        p_0=1e13,
        E_p=1.5,
        volume=w_subdomain,
        reactant=[mobile_D, empty_trap3],
        product=trap3_D,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_density),
        E_k=0.2,
        p_0=1e13,
        E_p=1.5,
        volume=w_subdomain,
        reactant=[mobile_T, empty_trap3],
        product=trap3_T,
    ),
]

############# Pulse Parameters (s) #############
pulses_per_day = 13

flat_top_duration = 50*pulses_per_day
ramp_up_duration = 33*pulses_per_day 
ramp_down_duration = 35*pulses_per_day
dwelling_time = 72000 # 20 hours

total_time_pulse = flat_top_duration + ramp_up_duration + ramp_down_duration
total_time_cycle = total_time_pulse + dwelling_time

isotope_split = 0.5

############# Temperature Parameters (K) #############
T_coolant = 343 # 70 degree C cooling water

def T_surface(t): # plasma-facing side
    return 1.1e-4*heat+T_coolant

def T_rear(t): # coolant facing side
    return 2.2e-5*heat+T_coolant

def T_function(x, t: Constant):
    a = (T_rear(t) - T_surface(t)) / L
    b = T_surface(t)
    flat_top_value = a * x[0] + b
    resting_value = T_coolant
    return (
        flat_top_value
        if float(t) % total_time_cycle < total_time_pulse
        else resting_value
    )

my_model.temperature = T_function

############# Flux Parameters #############

def gaussian_distribution(x):
    depth = 3e-9
    width = 1e-9
    return ufl.exp(-((x[0] - depth) ** 2) / (2 * width**2))

def deuterium_ion_flux(t: Constant):
    flat_top_value = ion_flux*isotope_split 
    resting_value = 0
    return (
        flat_top_value
        if float(t) % total_time_cycle < total_time_pulse
        else resting_value
    )

def tritium_ion_flux(t: Constant):
    flat_top_value = ion_flux*isotope_split  
    resting_value = 0
    return (
        flat_top_value
        if float(t) % total_time_cycle < total_time_pulse
        else resting_value
    )

def deuterium_atom_flux(t: Constant):
    flat_top_value = atom_flux*isotope_split 
    resting_value = 0
    return (
        flat_top_value
        if float(t) % total_time_cycle < total_time_pulse
        else resting_value
    )

def tritium_atom_flux(t: Constant):
    flat_top_value = atom_flux*isotope_split  
    resting_value = 0
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
        volume=w_subdomain
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
        volume=w_subdomain
    )
    
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
    surface_reaction_tt
]

############# Exports #############

folder = f'mb{mb}_results'

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
    atol=1e-15, rtol=1e-15, max_iterations=1000, final_time=3000
)

my_model.settings.stepsize = F.Stepsize(initial_value=20)

############# Run Simu #############

my_model.initialise()
my_model.run()

############# Results Plotting #############

import matplotlib.pyplot as plt

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