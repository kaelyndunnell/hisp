from hisp.h_transport_class import CustomProblem
from hisp.helpers import PulsedSource, gaussian_distribution

import numpy as np
import festim as F
import h_transport_materials as htm

from typing import Callable, Tuple, Dict

# TODO this is hard coded and show depend on incident energy?
implantation_range = 3e-9  # m
width = 1e-9  # m

def make_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float = 6e-3,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the MB scenario.

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
        .filter(author="frauenfelder")
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

    # density_func =
    empty_trap3 = F.ImplicitSpecies(  # not implicit, but can simplify trap creation model to small damanged zone in first 10nm
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
        final_time=final_time,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    return my_model, quantities
