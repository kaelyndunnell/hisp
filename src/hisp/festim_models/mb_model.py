from hisp.h_transport_class import CustomProblem
from hisp.helpers import PulsedSource, gaussian_distribution

import numpy as np
import festim as F
import h_transport_materials as htm
import ufl

from typing import Callable, Tuple, Dict

# TODO this is hard coded and show depend on incident energy?
implantation_range = 3e-9  # m
width = 1e-9  # m


def make_W_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the W MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
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
    w_diffusivity = (
        htm.diffusivities.filter(material="tungsten")
        .filter(isotope="h")
        .filter(author="holzner")
    )
    w_diffusivity = w_diffusivity[0]
    D_0 = w_diffusivity.pre_exp.magnitude
    E_D = w_diffusivity.act_energy.magnitude
    tungsten = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="tungsten",
    )

    # mb subdomains
    w_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [w_subdomain, inlet, outlet]

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

    # TODO: make trap space dependent (existing in only first 10nm)
    # density_func = lambda x: ufl.conditional(ufl.gt(x[0],10), 6.338e27, 0.0) #  small damanged zone in first 10nm, 1e-1 at.fr.
    empty_trap3 = F.ImplicitSpecies(
        n=6.338e27,
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

    interstitial_distance = 1.117e-10  # m
    interstitial_sites_per_atom = 6

    # hydrogen reactions - 1 per trap per species
    my_model.reactions = [
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.5,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=D_0 / (interstitial_distance * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.5,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    my_model.sources = [
        PulsedSource(
            flux=deuterium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=w_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
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

    my_model.exports = [
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
    ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=w_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity

    ############# Settings #############
    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        max_iterations=1000000,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    return my_model, quantities


def make_B_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the B MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############

    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, L, num=200),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # B material parameters
    b_density = 1.34e29  # atoms/m3
    D_0 = 1.07e-6  # m^2/s
    E_D = 0.3  # eV
    boron = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="boron",
    )

    # mb subdomains
    b_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=boron)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [b_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    trap3_D = F.Species("trap3_D", mobile=False)
    trap3_T = F.Species("trap3_T", mobile=False)
    trap4_D = F.Species("trap3_T", mobile=False)
    trap4_T = F.Species("trap3_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.867e-1 * b_density,
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=5.214e-1 * b_density,
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    empty_trap3 = F.ImplicitSpecies(
        n=2.466e-1 * b_density,
        others=[trap3_T, trap3_D],
        name="empty_trap3",
    )

    empty_trap4 = F.ImplicitSpecies(
        n=1.280e-1 * b_density,
        others=[trap4_T, trap4_D],
        name="empty_trap4",
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
        trap4_D,
        trap4_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 8e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap4],
            product=trap4_D,
        ),
        F.Reaction(
            k_0=1e13/b_density,#D_0 / (interstitial_distance * interstitial_sites_per_atom * b_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap4],
            product=trap4_T,
        ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    my_model.sources = [
        PulsedSource(
            flux=deuterium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=b_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=b_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=b_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=b_subdomain,
        ),
    ]

    ############# Boundary Conditions #############
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
        F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
    ]

    ############# Exports #############

    my_model.exports = [
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d4.bp", field=trap4_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t4.bp", field=trap4_T),
    ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=b_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity

    ############# Settings #############
    my_model.settings = F.Settings(
        atol=1e-15,
        rtol=1e-15,
        max_iterations=1000,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    return my_model, quantities


def make_DFW_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the DFW MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############

    L = L  # m
    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, 3e-6, num=300),
            np.linspace(3e-6, 30e-6, num=200),
            np.linspace(30e-6, L, num=200),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # DFW material parameters
    dfw_density = 8.45e28  # atoms/m3
    D_0 = 1.45e-6  # m^2/s
    E_D = 0.59  # eV
    dfw = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="dfw",
    )

    # mb subdomains
    dfw_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=dfw)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [dfw_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=8e-2 * dfw_density,
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 2.545e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * dfw_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.7,
            volume=dfw_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * dfw_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.7,
            volume=dfw_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    my_model.sources = [
        PulsedSource(
            flux=deuterium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=dfw_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=dfw_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=dfw_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=dfw_subdomain,
        ),
    ]

    ############# Boundary Conditions #############
    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_D],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_tt = F.SurfaceReactionBC(
        reactant=[mobile_T, mobile_T],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_dt = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_T],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
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

    my_model.exports = [
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
        F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
        F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
    ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=dfw_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity

    ############# Settings #############
    my_model.settings = F.Settings(
        atol=1e-15,
        rtol=1e-15,
        max_iterations=1000,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    return my_model, quantities
