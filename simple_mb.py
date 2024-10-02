# simple monoblock simulation in festim

import festim as F
import numpy as np
import h_transport_materials as htm 

my_model = F.HydrogenTransportProblem()

# building 1D mesh, W mb

L = 6e-3 # m
vertices = np.linspace(0,L,num=300)
my_model.mesh = F.Mesh1D(vertices)


# W material parameters
w_density = 6.3382e28 # atoms/m3
tungsten = F.Material(D_0=1.39e-6, E_D=0.69, name="tungsten") # compute mean curve from HTM database

# tungsten_diff = htm.diffusivities.filter(material=htm.TUNGSTEN).mean()

# subdomains
w_subdomain = F.VolumeSubdomain1D(id=1, borders=[0,L], material=tungsten)
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


# CHANGE THIS ################## do this for 3 
empty_trap = F.ImplicitSpecies(
    n=1e21,
    others=[trap1_T, trap1_D],
    name="empty_trap1",
)
###############################


my_model.species = [
    mobile_D,
    mobile_T,

]

# hydrogen reactions
my_model.reactions = [
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.2,
        reactant=[mobile_H, empty_trap],
        product=trapped_H1,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        reactant=[mobile_H, trapped_H1],
        product=trapped_H2,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.2,
        reactant=[mobile_D, empty_trap],
        product=trapped_D1,
    ),
]
################ DO THIS #############

# temperature 
# my_model.temperature = 
################ DO THIS #############

# boundary conditions

my_model.boundary_conditions = [
    # F.DirichletBC(subdomain=inlet, value=1e20, species=mobile_T), ####### CHANGE THIS ########
    # F.DirichletBC(subdomain=outlet, value=1e19, species=mobile_D),
    F.DirichletBC(subdomain=outlet, value=0, species=mobile_T),
    F.DirichletBC(subdomain=inlet, value=0, species=mobile_D),
]

# exports

left_flux = F.SurfaceFlux(field=mobile_T, surface=inlet)
right_flux = F.SurfaceFlux(field=mobile_T, surface=outlet)

folder = "multi_isotope_trapping_example"

my_model.exports = [
    F.XDMFExport(f"{folder}/mobile_concentration_t.xdmf", field=mobile_T),
    F.XDMFExport(f"{folder}/mobile_concentration_d.xdmf", field=mobile_D),
    F.XDMFExport(f"{folder}/trapped_concentration_t1.xdmf", field=trapped_T1),
    F.XDMFExport(f"{folder}/trapped_concentration_t2.xdmf", field=trapped_T2),
    F.XDMFExport(f"{folder}/trapped_concentration_d1.xdmf", field=trapped_D1),
    F.XDMFExport(f"{folder}/trapped_concentration_d2.xdmf", field=trapped_D2),
    F.XDMFExport(f"{folder}/trapped_concentration_dt.xdmf", field=trapped_DT),
]

# settings

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, max_iterations=30, final_time=3000
)

my_model.settings.stepsize = F.Stepsize(initial_value=20)

# run simu

my_model.initialise()

print(my_model.formulation)
my_model.run()