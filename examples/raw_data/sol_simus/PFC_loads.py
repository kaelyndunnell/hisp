import numpy as np
from sys import exit
import imas
import logging
logging.basicConfig(level=logging.DEBUG, filename='./output.log',filemode='w')
import time
import re

 
def Find_grid_subset(grid_ggd,subset_name):
    # Searches for the subset with given name return subset nubmber in the subset array or -1 if search has failed
    ind = -1
    try:
        nr_subset = len(grid_ggd[0].grid_subset.array)
    except:
        logging.error('supplied object has no attribute grid_subset')
    else:
        for sub_id in range(nr_subset):
            sub_name = grid_ggd[0].grid_subset[sub_id].identifier.name
            if ( sub_name.lower() == subset_name.lower()):
                ind = sub_id

    return ind;

def Find_grid_subset_index(grid_ggd,subset_name):
    # Searches for the subset with given name return subset index or -1 if search has failed
    ind = -1
    try:
        nr_subset = len(grid_ggd[0].grid_subset.array)
    except:
        logging.error('supplied object has no attribute grid_subset')
    else:
        for sub_id in range(nr_subset):
            sub_name = grid_ggd[0].grid_subset[sub_id].identifier.name
            if ( sub_name.lower() == subset_name.lower()):
                ind = grid_ggd[0].grid_subset[sub_id].identifier.index

    return ind;

def Find_grid_subset_index_number(grid_ggd,subset_index):
    # Searches for the subset with given name return subset index or -1 if search has failed
    ind = -1
    try:
        nr_subset = len(grid_ggd[0].grid_subset.array)
    except:
        logging.error('supplied object has no attribute grid_subset')
    else:
        for sub_id in range(nr_subset):
            sub_index = grid_ggd[0].grid_subset[sub_id].identifier.index
            if ( sub_index == subset_index):
                ind = sub_id

    return ind;

def Find_subset_number(grid_ggd_mass,subset_index):
    # Searches for the subset with given index return subset number or -1 if search has failed
    ind = -1
    try:
        nr_subset = len(grid_ggd_mass.array)
    except:
        logging.error('supplied object has no attribute grid_subset')
    else:
        for sub_id in range(nr_subset):
            sub_index = grid_ggd_mass[sub_id].grid_subset_index
            if ( sub_index == subset_index):
                ind = sub_id

    return ind;

def Find_ion_specie(ggd, nucleus_charge):
    # Searches for the index of ion with given nuclei charge
    # Igonres molecular ions and other complex chemical stuff
    # Returns the list of isotopes associated with the given nucleus charge
    # If none found returns an empty list 
    isotope_list = []
    try:
        nr_ion = len(ggd.ion.array)
    except:
        logging.error('supplied object has no attribute ion')
    else:
        for ion_id in range(nr_ion):
            if ( (len(ggd.ion[ion_id].element.array) == 1) and (ggd.ion[ion_id].element[0].z_n == nucleus_charge) and (ggd.ion[ion_id].element[0].atoms_n == 1. ) ):
                isotope_list.append(ion_id)

    return isotope_list;


def Find_neut_specie(ggd, nucleus_charge):
    # Searches for the index of the ions with given nuclei charge
    # Includes simple molecules, complex ones are ignored
    # Returns the list of isotopes associated with the given nucleus charge
    # and second array with number of neucleus per particle
    # If none found returns an empty list 
    isotope_list = []
    isotope_count = []
    try:
        nr_neut = len(ggd.neutral.array)
    except:
        logging.error('supplied object has no attribute neutral')
    else:
        for neut_id in range(nr_neut):
            if ( (len(ggd.neutral[neut_id].element.array) == 1) and (ggd.neutral[neut_id].element[0].z_n == nucleus_charge) ):
                isotope_list.append(neut_id)
                isotope_count.append(ggd.neutral[neut_id].element[0].atoms_n)

    return isotope_list, isotope_count;

def SOLPS_target_loads_read(profiles, transport, slice_info):

    # Information about the time slice
    i_time       = slice_info["i_time"]
    shot         = slice_info["shot"  ]
    run          = slice_info["run"]

    # Constants
    H1_pot  = 1.35984340e+01
    He1_pot = 2.45873876e+01
    He2_pot = 5.44177630e+01
    eV2J    = 1.60217663e-19
    amu2kg  = 1.66054e-27


    # Output variables, _i - inner target, _o - outer
    ne_i = []   # electron density              [m^-3]
    flxi_i = [] # fuel ions flux                [m^-2s^-1]
    flxn_i = [] # fuel neutral flux             [m^-2s^-1]
    prm_i  = [] # fuel molecule pressure        [Pa]
    te_i = []   # electorn temperature          [eV]
    ti_i = []   # ion temperature               [eV]
    tn_i = []   # neutral temperature           [eV]
    pwre_i = [] # power loads with electorns    [W/m^2] 
    pwri_i = [] # power loads with ions         [W/m^2] 
    pwrn_i = [] # power loads with neutrals     [W/m^2] 
    ne_o = []   # electron density              [m^-3]
    flxi_o = [] # fuel ions flux                [m^-2s^-1]
    flxn_o = [] # fuel neutral flux             [m^-2s^-1]
    prm_o  = [] # fuel molecule pressure        [Pa]
    te_o = []   # electorn temperature          [eV]
    ti_o = []   # ion temperature               [eV]
    tn_o = []   # neutral temperature           [eV]
    pwre_o = [] # power loads with electorns    [W/m^2] 
    pwri_o = [] # power loads with ions         [W/m^2] 
    pwrn_o = [] # power loads with neutrals     [W/m^2] 
    # values are given at [r,z] - centers of cell faces that compose the divertor targets
    r_i = []  # r coordinate                    [m]
    z_i = []  # z coordinate                    [m]
    r_o = []  # r coordinate                    [m]
    z_o = []  # z coordinate                    [m]
    r1_i = [] # r beginning of the element      [m]
    z1_i = [] # z beginning of the element      [m]
    r2_i = [] # r end of the element            [m]
    z2_i = [] # z end of the element            [m]
    r1_o = [] # r beginning of the element      [m]
    z1_o = [] # z beginning of the element      [m]
    r2_o = [] # r end of the element            [m]
    z2_o = [] # z end of the element            [m]
    # values of x are given along the target, from PFR to SOL edge, 0 corresponds to strike point
    x_i = []  # x coordinate                    [m]
    x_o = []  # x coordinate                    [m]

    # Keep all the geometry reading just in case
    nr_nodes = len(profiles.grid_ggd[0].space[0].objects_per_dimension[0].object.array)
    nr_edges = len(profiles.grid_ggd[0].space[0].objects_per_dimension[1].object.array)
    nr_faces = len(profiles.grid_ggd[0].space[0].objects_per_dimension[2].object.array)

    # 0d
    pts = []
    for node_id in range(nr_nodes):
        pts.append(profiles.grid_ggd[0].space[0].objects_per_dimension[0].object[node_id].geometry)
    # 1d
    edges = []
    for edge_id in range(nr_edges):
        edges.append(profiles.grid_ggd[0].space[0].objects_per_dimension[1].object[edge_id].nodes)
    # 2d
    faces = []
    for face_id in range(nr_faces):
        faces.append(profiles.grid_ggd[0].space[0].objects_per_dimension[2].object[face_id].nodes)
    #3d
    volumes = []
    for face_id in range(nr_faces):
        volumes.append(profiles.grid_ggd[0].space[0].objects_per_dimension[2].object[face_id].geometry)

    # Find inner and outer target subsets
    inner_ind = Find_grid_subset(profiles.grid_ggd,'inner_target')
    outer_ind = Find_grid_subset(profiles.grid_ggd,'outer_target')
    sep_ind   = Find_grid_subset(profiles.grid_ggd,'Separatrix')
    if ( (inner_ind < 0) or (outer_ind < 0) or (sep_ind < 0) ):
        logging.critical('Could not find inner, outer divertor or separatrix cannot procceed. inner_subset:%s/outer_subset:%s:separatrix_subset:%s', inner_ind, outer_ind, sep_ind)
        sys.exit()

    # Find hydrogenic and helium species in transport ids to account for the recombination loads
    # Impurities other than helium are ignored because thier density is always low
    H_isotope = Find_ion_specie(transport.model[0].ggd[0],1.0)
    He_isotope = Find_ion_specie(transport.model[0].ggd[0],2.0)

    #Find fuel neutrals
    Hn_isotope, Hn_count = Find_neut_specie(transport.model[0].ggd[0],1.0)

    # Check for the data availability in the ids
    nr_inner = len(profiles.grid_ggd[0].grid_subset[inner_ind].element.array)
    dummy = np.zeros(nr_inner, dtype=np.float64)
    # te
    try:
        te = profiles.ggd[0].electrons.temperature[inner_ind].values[0]
    except:
        logging.error('no data for the electron temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        te_avail = False
        te_i = dummy
    else:
        te_avail = True
    # ti
    try:
        ti = profiles.ggd[0].ion[0].temperature[inner_ind].values[0]
        ti = 1./ti
    except:
        logging.error('no data for the ion temperature found in edge_profiles ids, trying average ion temperature')
        try:
            ti = profiles.ggd[0].t_i_average[inner_ind].values[0]
        except:
            logging.error('no data for the ion temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
            ti_avail = False
            ti_i = dummy
        else:
            ti_avail = True
            ti_avg = True
    else:
        ti_avail = True
        ti_avg = False
    # tn
    try:
        tn = profiles.ggd[0].neutral[0].state[0].temperature[inner_ind].values[0]
    except:
        logging.error('no data for the neutral temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        tn_avail = False
        tn_i = dummy
    else:
        tn_avail = True
    # ne
    try:
        ne = profiles.ggd[0].electrons.density[inner_ind].values[0]
    except:
        logging.error('no data for the electron density found in edge_profiles ids, corresponding massive will be filled with zeros')
        ne_avail = False
        ne_i = dummy
    else:
        ne_avail = True
    # prm
    try:
        prm = profiles.ggd[0].neutral[0].state[0].pressure[inner_ind].values[0]
    except:
        logging.error('no data for the neutral pressure found in edge_profiles ids, corresponding massive will be filled with zeros')
        prm_avail = False
        prm_i = dummy
    else:
        prm_avail = True
    # pwre
    try:
        pwre = transport.model[0].ggd[0].electrons.energy.flux[inner_ind].values[0]
    except:
        logging.error('no data for the electron heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
        pwre_avail = False
        pwre_i = dummy
    else:
        pwre_avail = True
    # pwri
    try:
        pwri = transport.model[0].ggd[0].total_ion_energy.flux[inner_ind].values[0]
    except:
        logging.error('no data for the ion heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
        pwri_avail = False
        pwri_i = dummy
    else:
        pwri_avail = True
    # ion recombination component of pwri
    try:
        tri = transport.model[0].ggd[0].ion[0].particles.flux[inner_ind].values[0]
    except:
        logging.error('no data for the ion particle flux found in edge_transport ids, the data for recombination heat loads to the target wont be added to the ion heat flux')
        tri_avail = False
    else:
        tri_avail = True
    # pwrn
    try:
        pwrn = transport.model[0].ggd[0].neutral[0].energy.flux[inner_ind].values[0]
        hlp = transport.model[0].ggd[0].neutral[0].energy.flux[inner_ind].values[0:nr_inner]
        if ( np.amax(hlp) <= 0. ):
            pwrn = transport.model[0].ggd[0].neutral[0].energy.flux[inner_ind].values[999999]
    except:
        logging.error('no data for the neutral heat flux found in edge_transport ids, trying crude estimate: Pneut = 1/2*k*n*T*u, where u=sqrt(8kT/pi/M)')
        try:
            pwrn = profiles.ggd[0].neutral[0].state[0].density[inner_ind].values[0]
            pwrn = profiles.ggd[0].neutral[0].state[0].temperature[inner_ind].values[0]
        except:
            logging.error('no data for the neutral heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
            pwrn_avail = False
            pwrn_i = dummy
        else:
            pwrn_avail = True
            pwrn_calc = True
    else:
        pwrn_avail = True
        pwrn_calc = False
    if ( pwrn_avail == True ):
        if ( pwrn_calc == True ):
            nr_neut = len(profiles.ggd[0].neutral.array)
        else:
            nr_neut = len(transport.model[0].ggd[0].neutral.array)
    # pwrr
    logging.warning('SOLPS ids do not contain radiation power loads, corresponding massive will be filled with zeros')
    pwrr_i = dummy

    # Define x, r and z coordinates of the inner target data and extract corresponding data
    edge_index = profiles.grid_ggd[0].grid_subset[sep_ind].element[0].object[0].index
    sep_1 = pts[edges[edge_index-1][0]][0]
    sep_2 = pts[edges[edge_index-1][0]][1]
    dsp = 0.
    x_0 = 0.
    x_current = 0.
    for elem_id in range(nr_inner):
        edge_index = profiles.grid_ggd[0].grid_subset[inner_ind].element[elem_id].object[0].index
        node_1 = edges[edge_index-1][0]
        node_2 = edges[edge_index-1][1]
        r_current = 0.5 * (pts[node_1-1][0] + pts[node_2-1][0])
        z_current = 0.5 * (pts[node_1-1][1] + pts[node_2-1][1])
        r_i.append(r_current)
        z_i.append(z_current)
        r1_i.append(pts[node_1-1][0])
        r2_i.append(pts[node_2-1][0])
        z1_i.append(pts[node_1-1][1])
        z2_i.append(pts[node_2-1][1])
        x_1 = 0.5*np.sqrt((pts[node_2-1][0] - pts[node_1-1][0])**2. + (pts[node_2-1][1] - pts[node_1-1][1])**2.) 
        x_i.append(x_current)
        x_current = x_current + x_0 + x_1
        x_0 = x_1
        if ((sep_1 == pts[node_2-1][0]) and (sep_2 == pts[node_2-1][1])):
            dsp = x_current - x_0
        if ( te_avail == True ):
            te = profiles.ggd[0].electrons.temperature[inner_ind].values[elem_id]
            te_i.append(te)
        if ( ti_avail == True ):
            if ( ti_avg == True ):
                ti = profiles.ggd[0].t_i_average[inner_ind].values[elem_id]
            else:
                ti = profiles.ggd[0].ion[H_isptope[0]].temperature[inner_ind].values[elem_id]
            ti_i.append(ti)
        if ( tn_avail == True ):
            for p in range(len(Hn_isotope)):
                if ( Hn_count[p] == 1. ):
                    tn = profiles.ggd[0].neutral[Hn_isotope[p]].state[0].temperature[inner_ind].values[elem_id]
            tn_i.append(tn)
        if ( ne_avail == True ):
            ne = profiles.ggd[0].electrons.density[inner_ind].values[elem_id]
            ne_i.append(ne)
        if ( prm_avail == True ):
            for p in range(np.size(Hn_isotope)):
                if ( Hn_count[p] == 2. ):
                    prm = profiles.ggd[0].neutral[Hn_isotope[p]].state[0].pressure[inner_ind].values[elem_id]
            prm_i.append(prm)
        if ( pwre_avail == True ):
            pwre = -transport.model[0].ggd[0].electrons.energy.flux[inner_ind].values[elem_id]
            pwre_i.append(pwre)
        if ( pwri_avail == True ):
            pwri = -transport.model[0].ggd[0].total_ion_energy.flux[inner_ind].values[elem_id]
            if ( tri_avail == True ):
                for H_isotope_ind in range(0,len(H_isotope)):
                    ion_ind = H_isotope[H_isotope_ind]
                    flxi = -transport.model[0].ggd[0].ion[ion_ind].particles.flux[inner_ind].values[elem_id]
                    pwri = pwri - transport.model[0].ggd[0].ion[ion_ind].particles.flux[inner_ind].values[elem_id]*H1_pot*eV2J
                for He_isotope_ind in range(0,len(He_isotope)):
                    ion_ind = He_isotope[He_isotope_ind]
                    if (transport.model[0].ggd[0].ion[ion_ind].z_ion == 1.0):
                        pwri = pwri - transport.model[0].ggd[0].ion[ion_ind].state[0].particles.flux[inner_ind].values[elem_id]*He1_pot*eV2J
                    elif (transport.model[0].ggd[0].ion[ion_ind].z_ion == 2.0): 
                        pwri = pwri - transport.model[0].ggd[0].ion[ion_ind].state[0].particles.flux[inner_ind].values[elem_id]*He2_pot*eV2J
                    else:
                        logging.warning("He ion with Z neither 1, nor 2 found.. skipping")
            pwri_i.append(pwri)
            flxi_i.append(flxi)
        if ( pwrn_avail == True ):
            pwrn = 0.
            flxn = 0.
            for neut_id in range(nr_neut):
                if ( pwrn_calc == True ):
                    n_neut = profiles.ggd[0].neutral[neut_id].state[0].density[inner_ind].values[elem_id]
                    t_neut = profiles.ggd[0].neutral[neut_id].state[0].temperature[inner_ind].values[elem_id]*eV2J
                    m_neut = profiles.ggd[0].neutral[neut_id].element[0].a*amu2kg
                    v_neut = np.sqrt(8.*t_neut/np.pi/m_neut)
                    pwrn = pwrn + 0.5*n_neut*t_neut*v_neut 
                    for p in range(0,np.size(Hn_isotope)):
                        if ( (neut_id == Hn_isotope[p]) and (Hn_count[p] == 1.) ):
                            flxn = flxn + 0.25*n_neut*v_neut*Hn_count[p]
                else:
                    pwrn = pwrn - transport.model[0].ggd[0].neutral[neut_id].energy.flux[inner_ind].values[elem_id]
                    for p in range(np.size(Hn_isotope)):
                        if ( (neut_id == Hn_isotope[p]) and (Hn_count[p] == 1.) ):
                            flxn = flxn - transport.model[0].ggd[0].neutral[neut_id].state[0].particles.flux[inner_ind].values[elem_id]
            pwrn_i.append(pwrn)
            flxn_i.append(flxn)
    x_i = x_i - dsp

    # Repeat for the outer divertor
    # In current SOLPS version the some checks are unnecessary because nr_outer and nr_outer are always the same
    # things can change in wide_grids version comming 2023 though
    nr_outer = len(profiles.grid_ggd[0].grid_subset[outer_ind].element.array)
    dummy = np.zeros(nr_outer, dtype=np.float64)
    # te
    try:
        te = profiles.ggd[0].electrons.temperature[outer_ind].values[0]
    except:
        logging.error('no data for the electron temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        te_avail = False
        te_o = dummy
    else:
        te_avail = True
    # ti
    try:
        ti = profiles.ggd[0].ion[0].temperature[outer_ind].values[0]
    except:
        logging.error('no data for the ion temperature found in edge_profiles ids, trying average ion temperature')
        try:
            ti = profiles.ggd[0].t_i_average[outer_ind].values[0]
            ti = 1./ti
        except:
            logging.error('no data for the ion temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
            ti_avail = False
            ti_o = dummy
        else:
            ti_avail = True
            ti_avg = True
    else:
        ti_avail = True
        ti_avg = False
    # tn
    try:
        tn = profiles.ggd[0].neutral[0].state[0].temperature[outer_ind].values[0]
    except:
        logging.error('no data for the neutral temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        tn_avail = False
        tn_o = dummy
    else:
        tn_avail = True
    # ne
    try:
        ne = profiles.ggd[0].electrons.density[outer_ind].values[0]
    except:
        logging.error('no data for the electron density found in edge_profiles ids, corresponding massive will be filled with zeros')
        ne_avail = False
        ne_o = dummy
    else:
        ne_avail = True
    # prm
    try:
        prm = profiles.ggd[0].neutral[0].state[0].pressure[outer_ind].values[0]
    except:
        logging.error('no data for the neutral pressure found in edge_profiles ids, corresponding massive will be filled with zeros')
        prm_avail = False
        prm_o = dummy
    else:
        prm_avail = True
    # pwre
    try:
        pwre = transport.model[0].ggd[0].electrons.energy.flux[outer_ind].values[0]
    except:
        logging.error('no data for the electron heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
        pwre_avail = False
        pwre_o = dummy
    else:
        pwre_avail = True
    # pwri
    try:
        pwri = transport.model[0].ggd[0].total_ion_energy.flux[outer_ind].values[0]
    except:
        logging.error('no data for the ion heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
        pwri_avail = False
        pwri_o = dummy
    else:
        pwri_avail = True
    # ion recombination component of pwri
    try:
        tri = transport.model[0].ggd[0].ion[0].particles.flux[outer_ind].values[0]
    except:
        logging.error('no data for the ion particle flux found in edge_transport ids, the data for recombination heat loads to the target wont be added to the ion heat flux')
        tri_avail = False
    else:
        tri_avail = True
    # pwrn
    try:
        pwrn = transport.model[0].ggd[0].neutral[0].energy.flux[outer_ind].values[0]
        hlp = transport.model[0].ggd[0].neutral[0].energy.flux[outer_ind].values[0:nr_inner]
        if ( np.amax(hlp) <= 0. ):
            pwrn = transport.model[0].ggd[0].neutral[0].energy.flux[outer_ind].values[999999]
    except:
        logging.error('no data for the neutral heat flux found in edge_transport ids, trying crude estimate: Pneut = 1/2*k*n*T*u, where u=sqrt(8kT/pi/M)')
        try:
            pwrn = profiles.ggd[0].neutral[0].state[0].density[outer_ind].values[0]
            pwrn = profiles.ggd[0].neutral[0].state[0].temperature[outer_ind].values[0]
        except:
            logging.error('no data for the neutral heat flux found in edge_transport ids, corresponding massive will be filled with zeros')
            pwrn_avail = False
            pwrn_i = dummy
        else:
            pwrn_avail = True
            pwrn_calc = True
    else:
        pwrn_avail = True
        pwrn_calc = False
    if ( pwrn_avail == True ):
        if ( pwrn_calc == True ):
            nr_neut = len(profiles.ggd[0].neutral.array)
        else:
            nr_neut = len(transport.model[0].ggd[0].neutral.array)
    # pwrr
    logging.warning('SOLPS ids do not contain radiation power loads, corresponding massive will be filled with zeros')
    pwrr_o = dummy

    # Define x, r and z coordinates of the outer target data and extract corresponding data
    nr_sep = len(profiles.grid_ggd[0].grid_subset[sep_ind].element.array)
    edge_index = profiles.grid_ggd[0].grid_subset[sep_ind].element[nr_sep-1].object[0].index
    sep_1 = pts[edges[edge_index-1][1]][0]
    sep_2 = pts[edges[edge_index-1][1]][1]
    dsp = 0.
    x_0 = 0.
    x_current = 0.
    for elem_id in range(nr_outer):
        edge_index = profiles.grid_ggd[0].grid_subset[outer_ind].element[elem_id].object[0].index
        node_1 = edges[edge_index-1][0]
        node_2 = edges[edge_index-1][1]
        r_current = 0.5 * (pts[node_1-1][0] + pts[node_2-1][0])
        z_current = 0.5 * (pts[node_1-1][1] + pts[node_2-1][1])
        r_o.append(r_current)
        z_o.append(z_current)
        r1_o.append(pts[node_1-1][0])
        r2_o.append(pts[node_2-1][0])
        z1_o.append(pts[node_1-1][1])
        z2_o.append(pts[node_2-1][1])
        x_1 = 0.5*np.sqrt((pts[node_2-1][0] - pts[node_1-1][0])**2. + (pts[node_2-1][1] - pts[node_1-1][1])**2.) 
        x_o.append(x_current)
        x_current = x_current + x_0 + x_1
        x_0 = x_1
        if ((sep_1 == pts[node_2-1][0]) and (sep_2 == pts[node_2-1][1])):
            dsp = x_current - x_0
        if ( te_avail == True ):
            te = profiles.ggd[0].electrons.temperature[outer_ind].values[elem_id]
            te_o.append(te)
        if ( ti_avail == True ):
            if ( ti_avg == True ):
                ti = profiles.ggd[0].t_i_average[outer_ind].values[elem_id]
            else:
                ti = profiles.ggd[0].ion[H_isotope[0]].temperature[outer_ind].values[elem_id]
            ti_o.append(ti)
        if ( tn_avail == True ):
            for p in range(len(Hn_isotope)):
                if ( Hn_count[p] == 1. ):
                    tn = profiles.ggd[0].neutral[Hn_isotope[p]].state[0].temperature[outer_ind].values[elem_id]
            tn_o.append(tn)
        if ( ne_avail == True ):
            ne = profiles.ggd[0].electrons.density[outer_ind].values[elem_id]
            ne_o.append(ne)
        if ( prm_avail == True ):
            for p in range(np.size(Hn_isotope)):
                if ( Hn_count[p] == 2. ):
                    prm = profiles.ggd[0].neutral[Hn_isotope[p]].state[0].pressure[outer_ind].values[elem_id]
            prm_o.append(prm)
        if ( pwre_avail == True ):
            pwre = transport.model[0].ggd[0].electrons.energy.flux[outer_ind].values[elem_id]
            pwre_o.append(pwre)
        if ( pwri_avail == True ):
            pwri = transport.model[0].ggd[0].total_ion_energy.flux[outer_ind].values[elem_id]
            if ( tri_avail == True ):
                for H_isotope_ind in range(0,len(H_isotope)):
                    ion_ind = H_isotope[H_isotope_ind]
                    flxi = transport.model[0].ggd[0].ion[ion_ind].particles.flux[outer_ind].values[elem_id] 
                    pwri = pwri + transport.model[0].ggd[0].ion[ion_ind].particles.flux[outer_ind].values[elem_id]*H1_pot*eV2J
                for He_isotope_ind in range(0,len(He_isotope)):
                    ion_ind = He_isotope[He_isotope_ind]
                    if (transport.model[0].ggd[0].ion[ion_ind].z_ion == 1.0):
                        pwri = pwri + transport.model[0].ggd[0].ion[ion_ind].state[0].particles.flux[outer_ind].values[elem_id]*He1_pot*eV2J
                    elif (transport.model[0].ggd[0].ion[ion_ind].z_ion == 2.0): 
                        pwri = pwri + transport.model[0].ggd[0].ion[ion_ind].state[0].particles.flux[outer_ind].values[elem_id]*He2_pot*eV2J
                    else:
                        logging.warning("He ion with Z neither 1, nor 2 found.. skipping")
            pwri_o.append(pwri)
            flxi_o.append(flxi)
        if ( pwrn_avail == True ):
            pwrn = 0.
            flxn = 0.
            for neut_id in range(nr_neut):
                if ( pwrn_calc == True ):
                    n_neut = profiles.ggd[0].neutral[neut_id].state[0].density[outer_ind].values[elem_id]
                    t_neut = profiles.ggd[0].neutral[neut_id].state[0].temperature[outer_ind].values[elem_id]*eV2J
                    m_neut = profiles.ggd[0].neutral[neut_id].element[0].a*amu2kg
                    v_neut = np.sqrt(8.*t_neut/np.pi/m_neut)
                    pwrn = pwrn + 0.5*n_neut*t_neut*v_neut 
                    for p in range(0,np.size(Hn_isotope)):
                        if ( (neut_id == Hn_isotope[p]) and (Hn_count[p] == 1.) ):
                            flxn = flxn + 0.25*n_neut*v_neut*Hn_count[p]
                else:
                    pwrn = pwrn + transport.model[0].ggd[0].neutral[neut_id].energy.flux[outer_ind].values[elem_id]
                    for p in range(0,np.size(Hn_isotope)):
                        if ( (neut_id == Hn_isotope[p]) and (Hn_count[p] == 2.) ):
                            flxn = flxn + transport.model[0].ggd[0].neutral[neut_id].state[0].particles.flux[outer_ind].values[elem_id]
            pwrn_o.append(pwrn)
            flxn_o.append(flxn)
    x_o = x_o - dsp

    inner_coords = np.array([[r_i[i], z_i[i]] for i in range(len(r_i))])
    outer_coords = np.array([[r_o[i], z_o[i]] for i in range(len(r_o))])

###########
#    # Sample output for the demonstration purpose, to be removed by user
#    print('##########################################################')
#    print('Shot : run : i_time', shot,run,i_time) 
#    print('##########################################################')
#    print('=================== Inner target =========================') 
#    for i in range(nr_inner):
#        print('Segment : [r, z]    || ', i+1,inner_coords[i])
#        print('  te : ti : ne      || ', te_i[i], ti_i[i], ne_i[i])
#        print(' pwre : pwri : pwrn || ', pwre_i[i], pwri_i[i], pwrn_i[i])
#        print('  flxi : flxn       || ', flxi_i[i], flxn_i[i])
#        print('------------------------------------------------------')
#    print('=================== Outer target =========================') 
#    for i in range(nr_outer):
#        print('Segment : [r, z]    || ', i+1,outer_coords[i])
#        print('  te : ti : ne      || ', te_o[i], ti_o[i], ne_o[i])
#        print(' pwre : pwri : pwrn || ', pwre_o[i], pwri_o[i], pwrn_o[i])
#        print('  flxi : flxn       || ', flxi_o[i], flxn_o[i])
#        print('------------------------------------------------------')
#############

    # Writting output to files
    names = ['./inner_target','./outer_target']
    files = []
    for i in range(0,np.size(names)):
        ffile = '{0}{1}{2}{3}{4}{5}'.format(names[i],'.shot',str(shot),'.run',str(run),'.dat')
        files.append(ffile)
    for f in range(0,np.size(files)):

        ffile = files[f]

        with open(ffile,'w') as ff:

            header = ' # r1,r2 - radial coordinates of the segment    [m]        \n' + \
                     ' # z1,z2 - vertical coordinates of the segemnt  [m]        \n' + \
                     ' # rc,zc - coordinates of the segment center    [m]        \n' + \
                     ' # xc    - coordinate along the target (0 at SP)[m]        \n' + \
                     ' # ne    - electron density                     [m^-3]     \n' + \
                     ' # Te    - electron temperature                 [eV]       \n' + \
                     ' # Ti    - ion temperature                      [eV]       \n' + \
                     ' # Tn    - neurtal temperature                  [eV]       \n' + \
                     ' # flxi  - ion flux (only fuel ions)            [m^-2*s^-1]\n' + \
                     ' # flxn  - neutral flux (only fuel atoms)       [m^-2*s^-1]\n' + \
                     ' # prm   - molecule pressure (only fuel mol.)   [Pa]       \n' + \
                     ' # pwre  - electron power flux                  [W/m^2]    \n' + \
                     ' # pwri  - ion power flux (including recomb.)   [W/m^2]    \n' + \
                     ' # pwrn  - neutral power flux                   [W/m^2]    \n' + \
                     ' # pwrr  - radiation power flux                 [W/m^2]    \n' 
            ff.write(header)
            
            str_len = 14
            ttls = ['r1','z1','r2','z2','rc','zc','xc','ne','Te','Ti','Tn','flxi','flxn','prm','pwre','pwri','pwrn','pwrr']
            
            prin = ' #'
            for p in range(0,np.size(ttls)):
                ttl_len = len(ttls[p])
                step1 = ''
                step2 = ''
                if ( ttl_len > str_len ):
                    logging.warning('title length is greater than limit and will be truncated, title:limit:%s:%s',ttls[p],str_len)
                    ttl=ttls[p][0:str_len]
                else:
                    step1 = int((str_len - ttl_len)/2.)
                    step2 = str_len - step1 - ttl_len
                    ttl = ttls[p]
                prin = prin + ' '*step1 + ttl + ' '*step2
            prin = prin + '\n'
            ff.write(prin)

            if ( f == 0 ):
                data = (r1_i,z1_i,r2_i,z2_i,r_i,z_i,x_i,ne_i,te_i,ti_i,tn_i,flxi_i,flxn_i,prm_i,pwre_i,pwri_i,pwrn_i,pwrr_i)
            elif ( f == 1 ):
                data = (r1_o,z1_o,r2_o,z2_o,r_o,z_o,x_o,ne_o,te_o,ti_o,tn_o,flxi_o,flxn_o,prm_o,pwre_o,pwri_o,pwrn_o,pwrr_o)
            for l in range(0,np.size(data[0])):
                prin = '  '
                for i in range(0,np.size(ttls)):
                    prin = '{0}{1}'.format(prin,' % 7.6E' % data[i][l])
                prin = prin + '\n'
                ff.write(prin)

    return {'inner_target_coords':inner_coords,'inner_target_te:':te_i,'inner_target_ti':ti_i, \
            'inner_target_tn':tn_i,'inner_target_ne':ne_i,'inner_target_prm':prm_i, \
            'inner_target_pwre':pwre_i,'inner_target_pwri':pwri_i, \
            'inner_target_pwrn':pwrn_i,'inner_target_pwrr':pwrr_i, \
            'outer_target_coords':outer_coords,'outer_target_te:':te_o, 'outer_target_ti':ti_o, \
            'outer_target_tn':tn_o,'outer_target_ne':ne_o,'outer_target_prm':prm_o, \
            'outer_target_pwre':pwre_o,'outer_target_pwri':pwri_o, \
            'outer_target_pwrn':pwrn_o,'outer_target_pwrr':pwrr_o }

def SOLEDGE_full_wall_loads_read(profiles, wall, slice_info):

    # Information about the time slice
    i_time       = slice_info["i_time"]
    shot         = slice_info["shot"  ]
    run          = slice_info["run"]

    # Constants
    H1_pot  = 1.35984340e+01
    He1_pot = 2.45873876e+01
    He2_pot = 5.44177630e+01
    eV2J    = 1.60217663e-19
    amu2kg  = 1.66054e-27


    # Output variables,
    ne_w   = []   # electron density            [m^-3]
    flxi_w = []   # fuel ions flux              [m^-2s^-1]
    flxn_w = []   # fuel neutral flux (atoms)   [m^-2s^-1]
    prm_w  = []   # fuel molecule pressure      [Pa]
    te_w   = []   # electorn temperature        [eV]
    ti_w   = []   # ion temperature             [eV]
    tn_w   = []   # ion temperature             [eV]
    pwre_w = []   # power loads with electorns  [W/m^2] 
    pwri_w = []   # power loads with ions       [W/m^2] 
    pwrn_w = []   # power loads with neutrals   [W/m^2] 
    pwrr_w = []   # power loads with radiation  [W/m^2] 
    # values are given at [r,z] - centers of cell faces that compose the divertor targets
    rc_w = []  # r coordinate                  [m]
    zc_w = []  # z coordinate                  [m]
    r1_w = []  # r beginning of the element    [m]
    z1_w = []  # z beginning of the element    [m]
    r2_w = []  # r end of the element          [m]
    z2_w = []  # z end of the element          [m]
    # values of x are given along the wall from the first node to the last
    xc_w = []  # x coordinate                    [m]

    # Keep all the geometry reading just in case
    nr_nodes = len(wall.grid_ggd[0].space[0].objects_per_dimension[0].object.array)
    nr_edges = len(wall.grid_ggd[0].space[0].objects_per_dimension[1].object.array)
    nr_faces = len(wall.grid_ggd[0].space[0].objects_per_dimension[2].object.array)
    nt_nodes = len(profiles.grid_ggd[0].space[0].objects_per_dimension[0].object.array)
    if ( nr_nodes != nt_nodes):
        logging.critircal('Grids from wall and edge_profiles ids differ, cannot proceed further...')
        sys.exit()

    # 0d
    pts = []
    for node_id in range(nr_nodes):
        pts.append(wall.grid_ggd[0].space[0].objects_per_dimension[0].object[node_id].geometry)
        ptst = profiles.grid_ggd[0].space[0].objects_per_dimension[0].object[node_id].geometry
        if ( (pts[node_id][0] != ptst[0]) or (pts[node_id][1] != ptst[1]) ):
            logging.critircal('Point %s from wall and edge_profiles grid differ (pts_wall:pts_profiles:%s:%s) cannot proceed further...',node_id,pts[node_id],ptst)
            sys.exit()
    # 1d
    edges = []
    for edge_id in range(nr_edges):
        edges.append(wall.grid_ggd[0].space[0].objects_per_dimension[1].object[edge_id].nodes)
    # 2d
    faces = []
    for face_id in range(nr_faces):
        faces.append(wall.grid_ggd[0].space[0].objects_per_dimension[2].object[face_id].nodes)
    #3d
    volumes = []
    for face_id in range(nr_faces):
        volumes.append(wall.grid_ggd[0].space[0].objects_per_dimension[2].object[face_id].geometry)

    # Find inner and outer target subsets
    fwall_ind = Find_grid_subset_index(wall.grid_ggd,'full_wall')
    fprof_ind = Find_grid_subset_index(profiles.grid_ggd,'full_wall')
    if ( (fwall_ind < 0) or (fprof_ind < 0) ):
        logging.critical('Could not find "full wall" subset cannot proceed..')
        sys.exit()

    # Find hydrogenic and helium species in transport ids to account for the recombination loads
    # Impurities other than helium are ignored because thier density is always low
    H_isotope = Find_ion_specie(wall.ggd[0].energy_fluxes.kinetic,1.0)

    #Find fuel neutrals
    Hn_isotope, Hn_count = Find_neut_specie(wall.ggd[0].energy_fluxes.kinetic,1.0)

    # Check for the data availability
    ind = Find_grid_subset(wall.grid_ggd,'full_wall')
    nr_wall = len(wall.grid_ggd[0].grid_subset[ind].element.array)
    dummy = np.zeros(nr_wall, dtype=np.float64)
    # te
    try:
        ind = Find_subset_number(profiles.ggd[0].electrons.temperature,fprof_ind)
        te = profiles.ggd[0].electrons.temperature[ind].values[0]
    except:
        logging.error('no data for the electron temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        te_avail = False
        te_w = dummy
    else:
        te_avail = True
    # ti
    try:
        ind = Find_subset_number(profiles.ggd[0].ion[0].temperature,fprof_ind)
        ti = profiles.ggd[0].ion[0].temperature[ind].values[0]
        ti = 1./ti
    except:
        logging.error('no data for the ion temperature found in edge_profiles ids, trying average ion temperature')
        try:
            ind = Find_subset_number(profiles.ggd[0].t_i_average,fprof_ind)
            ti = profiles.ggd[0].t_i_average[ind].values[0]
        except:
            logging.error('no data for the ion temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
            ti_avail = False
            ti_w = dummy
        else:
            ti_avail = True
            ti_avg = True
    else:
        ti_avail = True
        ti_avg = False
    # tn
    try:
        ind = Find_subset_number(profiles.ggd[0].neutral[0].temperature,fprof_ind)
        tn = profiles.ggd[0].neutral[0].temperature[ind].values[0]
    except:
        logging.error('no data for the neutral temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        tn_avail = False
        tn_w = dummy
    else:
        tn_avail = True
    # ne
    try:
        ind = Find_subset_number(profiles.ggd[0].electrons.density,fprof_ind)
        ne = profiles.ggd[0].electrons.density[ind].values[0]
    except:
        logging.error('no data for the electron density found in edge_profiles ids, corresponding massive will be filled with zeros')
        ne_avail = False
        ne_w = dummy
    else:
        ne_avail = True
    # prm
    try:
        ind = Find_subset_number(profiles.ggd[0].neutral[0].pressure,fprof_ind)
        prm = profiles.ggd[0].neutral[0].pressure[ind].values[0]
    except:
        logging.error('no data for the neutral temperature found in edge_profiles ids, corresponding massive will be filled with zeros')
        prm_avail = False
        prm_w = dummy
    else:
        prm_avail = True
    # pwre
    try:
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.electrons.incident,fwall_ind)
        pwre = wall.ggd[0].energy_fluxes.kinetic.electrons.incident[ind].values[0]
    except:
        logging.error('no data for the electron heat flux found in wall ids, corresponding massive will be filled with zeros')
        pwre_avail = False
        pwre_w = dummy
    else:
        pwre_avail = True
    # pwri
    try:
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.ion[0].incident,fwall_ind)
        pwri = wall.ggd[0].energy_fluxes.kinetic.ion[0].incident[ind].values[0]
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.recombination.ion[0].incident,fwall_ind)
        pwri = wall.ggd[0].energy_fluxes.kinetic.ion[0].incident[ind].values[0]
    except:
        logging.error('no data for the ion heat flux and/or ion recombination power flux found in wall ids, corresponding massive will be filled with zeros')
        pwri_avail = False
        pwri_w = dummy
    else:
        pwri_avail = True
    # pwrn
    try:
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.neutral[0].incident,fwall_ind)
        pwrn = wall.ggd[0].energy_fluxes.kinetic.neutral[0].incident[ind].values[0]
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.neutral[0].emitted,fwall_ind)
        pwrn = wall.ggd[0].energy_fluxes.kinetic.neutral[0].emitted[ind].values[0]
    except:
        logging.error('no data for the neutral heat flux on/from the wall found in wall ids, corresponding massive will be filled with zeros')
        pwrn_avail = False
        pwrn_w = dummy
    else:
        pwrn_avail = True
    # pwrr
    try:
        ind = Find_subset_number(wall.ggd[0].energy_fluxes.radiation.incident,fwall_ind)
        pwrr = wall.ggd[0].energy_fluxes.radiation.incident[ind].values[0]
    except:
        logging.error('no data for the radiation heat flux found in wall ids, corresponding massive will be filled with zeros')
        pwrr_avail = False
        pwrr_w = dummy
    else:
        pwrr_avail = True
    # flxi
    try:
        ind = Find_subset_number(wall.ggd[0].particle_fluxes.ion[0].incident,fwall_ind)
        flxi = wall.ggd[0].particle_fluxes.ion[0].incident[ind].values[0]
    except:
        logging.error('no data for the ion particle flux found in wall ids, the data for corresponding massive will be filled with zeros')
        flxi_avail = False
        flxi_w = dummy
    else:
        flxi_avail = True
    # flxn
    try:
        ind = Find_subset_number(wall.ggd[0].particle_fluxes.neutral[0].incident,fwall_ind)
        flxn = wall.ggd[0].particle_fluxes.neutral[0].incident[ind].values[0]
    except:
        logging.error('no data for the neutral particle flux found in wall ids, the data for corresponding massive will be filled with zeros')
        flxn_avail = False
        flxn_w = dummy
    else:
        flxn_avail = True

    # Define x, r and z coordinates of the inner target data and extract corresponding data
    x_0 = 0.
    x_current = 0.
    ind_w = Find_grid_subset(profiles.grid_ggd,'full_wall')
    for elem_id in range(nr_wall):
        edge_index = profiles.grid_ggd[0].grid_subset[ind_w].element[elem_id].object[0].index
        node_1 = edges[edge_index-1][0]
        node_2 = edges[edge_index-1][1]
        r_current = 0.5 * (pts[node_1-1][0] + pts[node_2-1][0])
        z_current = 0.5 * (pts[node_1-1][1] + pts[node_2-1][1])
        rc_w.append(r_current)
        zc_w.append(z_current)
        r1_w.append(pts[node_1-1][0])
        r2_w.append(pts[node_2-1][0])
        z1_w.append(pts[node_1-1][1])
        z2_w.append(pts[node_2-1][1])
        x_1 = 0.5*np.sqrt((pts[node_2-1][0] - pts[node_1-1][0])**2. + (pts[node_2-1][1] - pts[node_1-1][1])**2.) 
        xc_w.append(x_current)
        x_current = x_current + x_0 + x_1
        x_0 = x_1
        if ( te_avail == True ):
            ind = Find_subset_number(profiles.ggd[0].electrons.temperature,fprof_ind)
            te = profiles.ggd[0].electrons.temperature[ind].values[elem_id]
            te_w.append(te)
        if ( ti_avail == True ):
            if ( ti_avg == True ):
                ind = Find_subset_number(profiles.ggd[0].t_i_average,fprof_ind)
                ti = profiles.ggd[0].t_i_average[ind].values[elem_id]
            else:
                ind = Find_subset_number(profiles.ggd[0].ion[0].temperature,fprof_ind)
                ti = profiles.ggd[0].ion[0].temperature[ind].values[elem_id]
            ti_w.append(ti)
        if ( tn_avail == True ):
            for p in range(np.size(Hn_isotope)):
                if ( Hn_count[p] == 1. ):
                    ind = Find_subset_number(profiles.ggd[0].neutral[H_isotope[p]].temperature,fprof_ind)
                    tn = profiles.ggd[0].neutral[H_isotope[p]].temperature[ind].values[elem_id]
            tn_w.append(tn)
        if ( ne_avail == True ):
            ind = Find_subset_number(profiles.ggd[0].electrons.density,fprof_ind)
            ne = profiles.ggd[0].electrons.density[ind].values[elem_id]
            ne_w.append(ne)
        if ( prm_avail == True ):
            for p in range(np.size(Hn_isotope)):
                if ( Hn_count[p] == 2. ):
                    ind = Find_subset_number(profiles.ggd[0].neutral[Hn_isotope[p]].pressure,fprof_ind)
                    prm = profiles.ggd[0].neutral[Hn_isotope[p]].pressure[ind].values[elem_id]
            prm_w.append(prm)
        if ( pwre_avail == True ):
            ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.electrons.incident,fwall_ind)
            pwre = wall.ggd[0].energy_fluxes.kinetic.electrons.incident[ind].values[elem_id]
            pwre_w.append(pwre)
        if ( pwri_avail == True ):
            pwri = 0.
            nr_ions = len(wall.ggd[0].energy_fluxes.kinetic.ion.array)
            for ion_id in range(nr_ions):
                ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.ion[ion_id].incident,fwall_ind)
                pwri = pwri + wall.ggd[0].energy_fluxes.kinetic.ion[ion_id].incident[ind].values[elem_id]
                ind = Find_subset_number(wall.ggd[0].energy_fluxes.recombination.ion[ion_id].incident,fwall_ind)
                pwri = pwri + wall.ggd[0].energy_fluxes.recombination.ion[ion_id].incident[ind].values[elem_id]
            pwri_w.append(pwri)
        if ( pwrn_avail == True ):
            pwrn = 0.
            nr_neut = len(wall.ggd[0].energy_fluxes.kinetic.neutral.array)
            for neut_id in range(nr_neut):
                ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.neutral[neut_id].incident,fwall_ind)
                pwrn = pwrn + wall.ggd[0].energy_fluxes.kinetic.neutral[neut_id].incident[ind].values[elem_id]
                ind = Find_subset_number(wall.ggd[0].energy_fluxes.kinetic.neutral[neut_id].emitted,fwall_ind)
                pwrn = pwrn - wall.ggd[0].energy_fluxes.kinetic.neutral[neut_id].emitted[ind].values[elem_id]
            pwrn_w.append(pwrn)
        if ( pwrr_avail == True ):
            ind = Find_subset_number(wall.ggd[0].energy_fluxes.radiation.incident,fwall_ind)
            pwrr = wall.ggd[0].energy_fluxes.radiation.incident[ind].values[elem_id]
            pwrr_w.append(pwrr)
        if ( flxi_avail == True ):
            flxi = 0.
            for ion_id in H_isotope:
                ind = Find_subset_number(wall.ggd[0].particle_fluxes.ion[ion_id].incident,fwall_ind)
                flxi = flxi + wall.ggd[0].particle_fluxes.ion[ion_id].incident[ind].values[elem_id]
            flxi_w.append(flxi)
        if ( flxn_avail == True ):
            flxn = 0.
            for p in range(np.size(Hn_isotope)):
                if ( Hn_count[p] == 1. ):
                    ind = Find_subset_number(wall.ggd[0].particle_fluxes.neutral[Hn_isotope[p]].incident,fwall_ind)
                    flxn = flxn + Hn_count[p]*wall.ggd[0].particle_fluxes.neutral[Hn_isotope[p]].incident[ind].values[elem_id]
            flxn_w.append(flxn)

    wall_coords = np.array([[rc_w[i], zc_w[i]] for i in range(len(rc_w))])

###########
#    # Sample output for the demonstration purpose, to be removed by user
#    print('##########################################################')
#    print('Shot : run : i_time', shot,run,i_time) 
#    print('##########################################################')
#    print('=================== Inner target =========================') 
#    for i in range(nr_wall):
#        print('     Segment : [r, z]      || ', i+1,wall_coords[i])
#        print('      te : ti : ne         || ', te_w[i], ti_w[i], ne_w[i])
#        print(' pwre : pwri : pwrn : pwrr || ', pwre_w[i], pwri_w[i], pwrn_w[i], pwrr_w[i])
#        print('      flxi : flxn          || ', flxi_w[i], flxn_w[i])
#        print('------------------------------------------------------')
#############

    # Writting output to files
    name = './wall'
    ffile = '{0}{1}{2}{3}{4}{5}'.format(name,'.shot',str(shot),'.run',str(run),'.dat')

    with open(ffile,'w') as ff:

        header = ' # r1,r2 - radial coordinates of the segment    [m]        \n' + \
                 ' # z1,z2 - vertical coordinates of the segemnt  [m]        \n' + \
                 ' # rc,zc - coordinates of the segment center    [m]        \n' + \
                 ' # ne    - electron density                     [m^-3]     \n' + \
                 ' # Te    - electron temperature                 [eV]       \n' + \
                 ' # Ti    - ion temperature                      [eV]       \n' + \
                 ' # Tn    - neutral temperature                  [eV]       \n' + \
                 ' # flxi  - ion flux (only fuel ions)            [m^-2*s^-1]\n' + \
                 ' # flxn  - neutral flux (only fuel atoms)       [m^-2*s^-1]\n' + \
                 ' # prm   - molectule pressure (only fuel mol.)  [Pa]       \n' + \
                 ' # pwre  - electron power flux                  [W/m^2]    \n' + \
                 ' # pwri  - ion power flux (including recomb.)   [W/m^2]    \n' + \
                 ' # pwrn  - neutral power flux                   [W/m^2]    \n' + \
                 ' # pwrr  - radiation power flux                 [W/m^2]    \n' 
        ff.write(header)
            
        str_len = 14
        ttls = ['r1','z1','r2','z2','rc','zc','ne','Te','Ti','Tn','flxi','flxn','prm','pwre','pwri','pwrn','pwrr']
            
        prin = ' #'
        for p in range(0,np.size(ttls)):
            ttl_len = len(ttls[p])
            step1 = ''
            step2 = ''
            if ( ttl_len > str_len ):
                logging.warning('title length is greater than limit and will be truncated, title:limit:%s:%s',ttls[p],str_len)
                ttl=ttls[p][0:str_len]
            else:
                step1 = int((str_len - ttl_len)/2.)
                step2 = str_len - step1 - ttl_len
                ttl = ttls[p]
            prin = prin + ' '*step1 + ttl + ' '*step2
        prin = prin + '\n'
        ff.write(prin)

        data = (r1_w,z1_w,r2_w,z2_w,rc_w,zc_w,ne_w,te_w,ti_w,tn_w,flxi_w,flxn_w,prm_w,pwre_w,pwri_w,pwrn_w,pwrr_w)
         
        for l in range(0,np.size(data[0])):
            prin = '  '
            for i in range(0,np.size(ttls)):
                prin = '{0}{1}'.format(prin,' % 7.6E' % data[i][l])
            prin = prin + '\n'
            ff.write(prin)

    return {'wall_coords':wall_coords,'wall_te:':te_w,'wall_ti':ti_w, 'wall_tn':tn_w, \
            'wall_ne':ne_w,'wall_pwre':pwre_w,'wall_pwri':pwri_w,'wall_pwrn':pwrn_w, \
            'wall_pwrr':pwrr_w, 'wall_flxi':flxi_w,'wall_flxn':flxn_w,'wall_prm':prm_w }

# Main program
def main():

    # Global input parameters
    username          = "public"
    device            = "iter"
#    shot_list    = [122481,122258]
    shot_list         = [123102,   106000  ]
    run_list          = [  1   ,     1     ]
    code_origin_list  = ['SOLPS', 'SOLEDGE']          # Code from which the radiation IDS was computed

    # shot_list         = [122258 ]
    # run_list          = [  1    ]
    # code_origin_list  = ['SOLPS']          # Code from which the radiation IDS was computed
    # shot_list         = [  106000  ]
    # run_list          = [     1     ]
    # code_origin_list  = [ 'SOLEDGE']          # Code from which the radiation IDS was computed

    for record in range(len(shot_list)):

        shot = shot_list[record]
        run  = run_list[record]
        code_origin = code_origin_list[record]

        logging.info("Reading shot = %s, run = %s from database = %s of user = %s "%(shot,run,device,username))
        x = imas.ids(shot, run)
        x.open_env(username, device, "3")
        x.edge_profiles.get()
        x.edge_transport.get()
        x.wall.get()

        # Time loop: Go over time slices in the IDS
        N_times = len(x.edge_profiles.time)
        logging.info( "Number of time slices = " + str(N_times))

        for i_time in range(0, N_times):

            time_now = x.edge_profiles.time[i_time]
            logging.info('Time  = ' + str(time_now) + ' s ')

            slice_info = { 'shot' : shot, 'run' : run, 'i_time' : i_time }

            if ( code_origin == 'SOLPS' ):
                trgloadsdata = SOLPS_target_loads_read(x.edge_profiles, x.edge_transport,  slice_info)
            elif ( code_origin == 'SOLEDGE' ):
                fwlloadsdata = SOLEDGE_full_wall_loads_read(x.edge_profiles, x.wall.description_ggd[0], slice_info)


if __name__ == "__main__":
    main()
