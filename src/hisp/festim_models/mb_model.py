from hisp.h_transport_class import CustomProblem
from hisp.helpers import PulsedSource, gaussian_distribution
from hisp.scenario import Scenario
from hisp.plamsa_data_handling import PlasmaDataHandling
import hisp.bin

import numpy as np
import festim as F
import h_transport_materials as htm

from typing import Callable, Tuple, Dict
from numpy.typing import NDArray

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
    exports=False,
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

    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=300),
            np.linspace(30e-9, 3e-6, num=400),
            np.linspace(3e-6, 30e-6, num=400),
            np.linspace(30e-6, 1e-4, num=400),
            np.linspace(1e-4, L, num=300),
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
    if exports:
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
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1e-3)

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
    exports=False,
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
            np.linspace(0, 30e-9, num=500),
            np.linspace(30e-9, 1e-7, num=500),
            np.linspace(1e-7, L, num=500),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # B material parameters from Etienne Hodilles's unpublished TDS study for boron
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
    trap4_D = F.Species("trap4_D", mobile=False)
    trap4_T = F.Species("trap4_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.867e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=5.214e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    empty_trap3 = F.ImplicitSpecies(
        n=2.466e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap3_T, trap3_D],
        name="empty_trap3",
    )

    empty_trap4 = F.ImplicitSpecies(
        n=1.280e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
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
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap4],
            product=trap4_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
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
    if exports:
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
        atol=1e8,
        rtol=1e-10,
        max_iterations=30,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1e-4)

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
    exports=False,
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

    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, 3e-6, num=300),
            np.linspace(3e-6, 30e-6, num=300),
            np.linspace(30e-6, 1e-4, num=300),
            np.linspace(1e-4, L, num=200),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # TODO: pull DFW material parameters from HTM?

    # from ITER mean value parameters (FIXME: add DOI)
    ss_density = 8.45e28  # atoms/m3
    D_0 = 1.45e-6  # m^2/s
    E_D = 0.59  # eV
    ss = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="ss",
    )

    # mb subdomains
    ss_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=ss)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [ss_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=8e-2 * ss_density,  # from Guillermain D 2016 ITER report T2YEND
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
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
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
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_D,
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            distribution=lambda x: gaussian_distribution(x, implantation_range, width),
            species=mobile_T,
            volume=ss_subdomain,
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
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=ss_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity

    ############# Settings #############
    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1e-3)

    return my_model, quantities


# calculate how the rear temperature of the W layer evolves with the surface temperature
# data from E.A. Hodille et al 2021 Nucl. Fusion 61 126003 10.1088/1741-4326/ac2abc (Table I)
heat_fluxes_hodille = [10e6, 5e6, 1e6]  # W/m2
T_rears_hodille = [552, 436, 347]  # K

import scipy.stats

slope_T_rear, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    heat_fluxes_hodille, T_rears_hodille
)


def calculate_temperature_W(
    x: float | NDArray, heat_flux: float, coolant_temp: float, thickness: float
) -> float | NDArray:
    """Calculates the temperature in the W layer based on coolant temperature and heat flux

    Reference:
    - Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020) 10.1038/s41598-020-74844-w
    - E.A. Hodille et al 2021 Nucl. Fusion 61 126003 10.1088/1741-4326/ac2abc

    Args:
        x: position in m
        heat_flux: heat flux in W/m2
        coolant_temp: coolant temperature in K
        thickness: thickness of the W layer in m

    Returns:
        temperature in K
    """
    # the evolution of T surface is taken from Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020).
    # https://doi.org/10.1038/s41598-020-74844-w
    T_surface = 1.1e-4 * heat_flux + coolant_temp

    T_rear = slope_T_rear * heat_flux + coolant_temp
    a = (T_rear - T_surface) / thickness
    b = T_surface
    return a * x + b


def calculate_temperature_B(heat_flux: float, coolant_temp: float) -> float:
    """
    Calculates the temperature in the boron layer based on coolant temperature and heat flux.
    The temperature is assumed to be homogeneous in the B layer and is calculated based on the
    surface temperature of the W layer.

    T_B = R_c * q + T_surface_W

    where
    - R_c is the thermal contact resistance of the layer in m2 K/W
    - q is the heat flux in W/m2
    - T_surface_W is the surface temperature of the W layer in K

    References:
    - Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020) 10.1038/s41598-020-74844-w
    - Jae-Sun Park et al 2023 Nucl. Fusion 63 076027 10.1088/1741-4326/acd9d9

    Args:
        heat_flux: heat flux in W/m2
        coolant_temp: coolant temperature in K

    Returns:
        temperature in K
    """
    # the evolution of T surface is taken from Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020).
    # https://doi.org/10.1038/s41598-020-74844-w
    T_surf_tungsten = 1.1e-4 * heat_flux + coolant_temp
    R_c_jet = 5e-4  # m2 K/W  calculated from JET-ILW (JPN#98297)
    return R_c_jet * heat_flux + T_surf_tungsten


def make_temperature_function(
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    bin: hisp.bin.SubBin | hisp.bin.DivBin,
    coolant_temp: float,
) -> Callable[[NDArray, float], NDArray]:
    """Returns a function that calculates the temperature of the bin based on time and position.

    Args:
        scenario: the Scenario object containing the pulses
        plasma_data_handling: the object containing the plasma data
        bin: the bin/subbin to get the temperature function for
        coolant_temp: the coolant temperature in K

    Returns:
        a callable of x, t returning the temperature in K
    """

    def T_function(x: NDArray, t: float) -> NDArray:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"

        # get the pulse and time relative to the start of the pulse
        pulse = scenario.get_pulse(t)
        t_rel = t - scenario.get_time_start_current_pulse(t)

        if pulse.pulse_type == "BAKE":
            T_bake = 483.15  # K
            value = np.full_like(x[0], T_bake)
        else:
            heat_flux = plasma_data_handling.get_heat(pulse, bin, t_rel)
            if (
                bin.material == "W" or bin.material == "SS"
            ):  # FIXME: update ss temp when gven data:
                value = calculate_temperature_W(
                    x[0], heat_flux, coolant_temp, bin.thickness
                )
            elif bin.material == "B":
                T_value = calculate_temperature_B(heat_flux, coolant_temp)
                value = np.full_like(x[0], T_value)
            else:
                raise ValueError(f"Unsupported material: {bin.material}")

        return value

    return T_function


def make_particle_flux_function(
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    bin: hisp.bin.SubBin | hisp.bin.DivBin,
    ion: bool,
    tritium: bool,
) -> Callable[[float], float]:
    """Returns a function that calculates the particle flux based on time.

    Args:
        scenario: the Scenario object containing the pulses
        plasma_data_handling: the object containing the plasma data
        bin: the bin/subbin to get the temperature function for
        ion: whether to get the ion flux
        tritium: whether to get the tritium flux

    Returns:
        a callable of t returning the **incident** particle flux in m^-2 s^-1
    """

    def particle_flux_function(t: float) -> float:
        assert isinstance(t, float), f"t should be a float, not {type(t)}"

        # get the pulse and time relative to the start of the pulse
        pulse = scenario.get_pulse(t)
        relative_time = t - scenario.get_time_start_current_pulse(t)

        # get the incident particle flux
        incident_hydrogen_particle_flux = plasma_data_handling.get_particle_flux(
            pulse=pulse,
            bin=bin,
            t_rel=relative_time,
            ion=ion,
        )

        # if tritium is requested, multiply by tritium fraction
        if tritium:
            value = incident_hydrogen_particle_flux * pulse.tritium_fraction
        else:
            value = incident_hydrogen_particle_flux * (1 - pulse.tritium_fraction)

        return value

    return particle_flux_function
