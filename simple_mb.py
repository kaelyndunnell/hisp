# simple monoblock simulation in festim

import festim as F
import numpy as np
import h_transport_materials as htm


my_model = F.HydrogenTransportProblem()

# building 1D mesh, W mb

L = 6e-3  # m
vertices = np.concatenate(
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
tungsten_diff = htm.diffusivities.filter(material=htm.TUNGSTEN).mean()
tungsten = F.Material(
    D_0=tungsten_diff.pre_exp.magnitude,
    E_D=tungsten_diff.act_energy.magnitude,
    name="tungsten",
)

# subdomains
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

empty_trap3 = F.ImplicitSpecies(  # fermi-dirac-like trap 3
    n=6.338e27,  # 1e-1 at.fr.
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

# temperature
# my_model.temperature = 1000 - (1000-350)/L*x
my_model.temperature = 1000

# boundary conditions

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=inlet, value=1e20, species=mobile_T),
    F.DirichletBC(subdomain=inlet, value=1e19, species=mobile_D),
    F.DirichletBC(subdomain=outlet, value=0, species=mobile_T),
    F.DirichletBC(subdomain=outlet, value=0, species=mobile_D),
]

# exports

left_flux = F.SurfaceFlux(field=mobile_T, surface=inlet)
right_flux = F.SurfaceFlux(field=mobile_T, surface=outlet)

folder = "multi_isotope_trapping_example"

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

# settings

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, max_iterations=30, final_time=3000
)

my_model.settings.stepsize = F.Stepsize(initial_value=20)

# run simu

my_model.initialise()
my_model.run()


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
