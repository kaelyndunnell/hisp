# bins the background fluxes as determined by bg_fluxes_plotter.py 
# determines average of flux in each bin and plots 
# outputs fluxes to a .dat file with corresponding wall index

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from bg_fluxes_avgofcases import plot_data, read_flux_file, average_values

# need a separate .dat file to read the background flux data files produced by bg_fluxes_avgofcases
# this is with the old data that required us to take averages
def read_dat_file(filename):

    # open and read .dat file
    with open(filename) as file: 
        lines = [line for line in file if not line.startswith('#')]

    # extract data from .dat file
    data = np.loadtxt(lines, skiprows=1)

    return data

# function to read output divertor data from solps
def read_div_solps(particle_fluxes_inner,
    particle_fluxes_outer,
    power_loads_inner, 
    power_loads_outer,
    inner_target_file_script, 
    outer_target_file_script):
    """
    reads output solps file directly, associating values from solps output 
    to the output produced by script described below: 

    reads the output solps file from Andrei Pshenov's divertor_target_loads_test.py script

    parameters:
        from solps directly: 
            outer and inner target power loads output file
            outer and inner target particle fluxes output file
        from Andrei Pshenov's script: 
            inner divertor target output solps file
            outer divertor target output solps file

    returns: 
        data_div: numpy array with divertor data for both targets
    """

    # open and read inner data file from Andrei
    with open(inner_target_file_script) as file: 
        inner_lines_script = np.loadtxt([line for line in file if not line.startswith('#')])

    # open and read outer data file from Andrei
    with open(outer_target_file_script) as file: 
        outer_lines_script = np.loadtxt([line for line in file if not line.startswith('#')])

    # read and open inner/outer particle fluxes data from solps
    with open(particle_fluxes_inner) as file:
        inner_particle_flux = np.loadtxt([line for line in file if not line.startswith('#')])

    with open(particle_fluxes_outer) as file:
        outer_particle_flux = np.loadtxt([line for line in file if not line.startswith('#')])

    # read and open inner/out power loads data from solps
    with open(power_loads_inner) as file:
        inner_power_loads = np.loadtxt([line for line in file if not line.startswith('#')])
    
    with open(power_loads_outer) as file:
        outer_power_loads = np.loadtxt([line for line in file if not line.startswith('#')])

    # start with the skeleton of the script output

    # THIS IS THE INNER TARGET
    # adjust geometry coordinates to match raw solps data
    # first, get rid of first line / point since we don't have it in solps data 
    inner_lines_script = inner_lines_script[1:]
    # the last four lines are power fluxes, but we can take this as a total
    # because that's what we want anywho
    inner_lines_script = np.delete(inner_lines_script,[13,14,16],axis=1)

    # first replace all data in script with data from raw (except the coordinates)
    inner_lines_script[:,7] = inner_power_loads[:-1,1] # ne
    inner_lines_script[:,8] = inner_power_loads[:-1,2] # Te
    inner_lines_script[:,9] = inner_power_loads[:-1,3]# Ti
    inner_lines_script[:,10] = inner_power_loads[:-1,3]# Tn -- equal to Ti for now
    inner_lines_script[:,11] = inner_particle_flux[:-1,6] # flxi
    inner_lines_script[:,12] = inner_particle_flux[:-1,5]# flxn
    inner_lines_script[:,13] = inner_power_loads[:-1,4] # Wtot
    inner_lines_script[:,14] = inner_power_loads[:-1,7] # Wion

    # now, add last line of solps data to inner_lines_script
    # to do this, have to calculate the last r2 and z2 point for 
    # the line
    r_diff = inner_lines_script[2][2] - inner_lines_script[1][2]
    z_diff = abs(inner_lines_script[2][1]) - abs(inner_lines_script[1][1])
    r2_sec_to_last = inner_lines_script[2][-1]
    z2_sec_to_last = inner_lines_script[3][-1]

    # row to add to inner_lines_script, filling with zeros to start
    columns = np.shape(inner_lines_script)[1] # number of columns in inner_lines_script
    last_row = np.zeros((1,columns))
    inner_lines_script = np.vstack((inner_lines_script, last_row))
    
    # THIS IS ALL FOR THE LAST ROW
    # now replace values with those in solps output
    # starting with geometry first 
    inner_lines_script[-1][0] = inner_lines_script[-2][2] # r1
    inner_lines_script[-1][1] = inner_lines_script[-2][3] # z1
    inner_lines_script[-1][2] = r_diff + inner_lines_script[-1][0] # r2
    inner_lines_script[-1][3] = z_diff + inner_lines_script[-1][1] # z2
    # next two columns in inner_lines_script are rc and zc
    # which we can find by using the midpoint 
    inner_lines_script[-1][4] = np.mean([inner_lines_script[-1][0],inner_lines_script[-1][2]]) # rc
    inner_lines_script[-1][5] = np.mean([inner_lines_script[-1][1],inner_lines_script[-1][3]]) # zc
    inner_lines_script[-1][6] = inner_particle_flux[-1][0] # xc
    inner_lines_script[-1][7] = inner_power_loads[-1][1] # ne
    inner_lines_script[-1][8] = inner_power_loads[-1][2] # Te
    inner_lines_script[-1][9] = inner_power_loads[-1][3] # Ti
    inner_lines_script[-1][10] = inner_power_loads[-1][3] # Tn, setting them roughly equal for now
    inner_lines_script[-1][11] = inner_particle_flux[-1][6] # ion flux flxi
    inner_lines_script[-1][12] = inner_particle_flux[-1][7] # atom flux flxn
    # inner_lines_script[-1][13] = inner_particle_flux[-1][-2] # fuel molecule pressure
    # now add in the last power fluxes here 
    inner_lines_script[-1][13] = inner_power_loads[-1][4] # Wtot
    inner_lines_script[-1][14] = inner_power_loads[-1][7] # Wion

    # do the same for the outer data 
    outer_lines_script = outer_lines_script[1:]
    outer_lines_script = np.delete(outer_lines_script,[13,14,16],axis=1)

    outer_lines_script[:,7] = outer_power_loads[:-1,1] # ne
    outer_lines_script[:,8] = outer_power_loads[:-1,2] # Te
    outer_lines_script[:,9] = outer_power_loads[:-1,3]# Ti
    outer_lines_script[:,10] = outer_power_loads[:-1,3]# Tn -- equal to Ti for now
    outer_lines_script[:,11] = outer_particle_flux[:-1,6] # flxi
    outer_lines_script[:,12] = outer_particle_flux[:-1,5]# flxn
    outer_lines_script[:,13] = outer_power_loads[:-1,4] # Wtot
    outer_lines_script[:,14] = outer_power_loads[:-1,7] # Wion

    r_diff = outer_lines_script[2][2] - outer_lines_script[1][2]
    z_diff = abs(outer_lines_script[2][1]) - abs(outer_lines_script[1][1])
    r2_sec_to_last = outer_lines_script[2][-1]
    z2_sec_to_last = outer_lines_script[3][-1]

    columns = np.shape(outer_lines_script)[1] # number of columns in outer_lines_script
    last_row = np.zeros((1,columns))
    outer_lines_script = np.vstack((outer_lines_script, last_row))
    # THIS IS ALL FOR THE LAST ROW
    outer_lines_script[-1][0] = outer_lines_script[-2][2] # r1
    outer_lines_script[-1][1] = outer_lines_script[-2][3] # z1
    outer_lines_script[-1][2] = r_diff + outer_lines_script[-1][0] # r2
    outer_lines_script[-1][3] = z_diff + outer_lines_script[-1][1] # z2
    outer_lines_script[-1][4] = np.mean([outer_lines_script[-1][0],outer_lines_script[-1][2]]) # rc
    outer_lines_script[-1][5] = np.mean([outer_lines_script[-1][1],outer_lines_script[-1][3]]) # zc
    outer_lines_script[-1][6] = outer_particle_flux[-1][0] # xc
    outer_lines_script[-1][7] = outer_power_loads[-1][1] # ne
    outer_lines_script[-1][8] = outer_power_loads[-1][2] # Te
    outer_lines_script[-1][9] = outer_power_loads[-1][3] # Ti
    outer_lines_script[-1][10] = outer_power_loads[-1][3] # Tn, setting them roughly equal for now
    outer_lines_script[-1][11] = outer_particle_flux[-1][6] # ion flux flxi
    outer_lines_script[-1][12] = outer_particle_flux[-1][7] # atom flux flxn
    # outer_lines_script[-1][13] = outer_particle_flux[-1][-2] # fuel molecule pressure
    # now add in the last power fluxes here 
    outer_lines_script[-1][13] = outer_power_loads[-1][4] # Wtot
    outer_lines_script[-1][14] = outer_power_loads[-1][7] # Wion

    # finally, combine these values into the final script 
    data_div = np.concatenate((inner_lines_script, outer_lines_script))

    return inner_lines_script, outer_lines_script, data_div

# function to read output wall data from soledge & Andrei's script
def read_wall_soledge(wall_sol_file):
    """
    reads wall soledge output data from script provided by 
    Andrei Pshenov

    parameters:
    wall sol file that is created by Andrei's script 

    returns: 
    data_wall: numpy array with wall plasma data from soledge
    """

    # open and read .dat file
    with open(wall_sol_file) as file: 
        lines = [line for line in file if not line.startswith('#')]

    # extract data from .dat file
    data_wall = np.loadtxt(lines)

    return data_wall

# function to load tokamak geometry for bins (z and r coordinates on pre-defined map, input as .txt file)
def load_geometry(filename, wall=True):
    """
    loads .txt file and assigns data columns accordingly 
    """

    # open and read input .txt file 
    with open(filename) as file: 
        lines = [line for line in file if not line.startswith('#')] 

    # load file in numpy array 
    geometry = np.loadtxt(lines)

    # extract z and r coordinates, which will serve as our bin edges
    # z-coordinates are the first column in data, convert to list
    if wall:
        z_coord = geometry[:,0]*1e-3 # convert mm to m

        # r-coordinates are the second column in data, convert to list 
        r_coord = geometry[:,1]*1e-3 # convert mm to m

    else:
        z_coord = geometry[:,0]
        r_coord = geometry[:,1]

    return z_coord, r_coord

# remove lines that hold divertor info from input SOLEDGE data which we don't want 
def remove_structure_points_soledge(data_wall):
    """
    removes data that represents the divertor structure and not actual 
    divertor surfaces from our data lists 

    this works for SOLEDGE so we are going to leave it be
    
    parameters: numpy array of data
    """
    # for SOLEDGE data
    r1 = data_wall[:,0].tolist()
    z1 = data_wall[:,1].tolist()

    removed_idx = []

    data_wall_clean = []

    # left-side. bins (z1,r1) = -3760,4640 and -3910,4500
    # maybe instead of removing, do it by building. that way idxs won't keep changing
    for idx in range(len((r1))):
        # we only want to manipulate the divertor points, so add all points automatically
        if z1[idx] >= -3.051:
            data_wall_clean.append((data_wall[idx].tolist()))
        else:

            # first we'll keep what is above -3.910 and to the left of 4.500
            if z1[idx] > -3.910 and r1[idx] < 4.640:
                    data_wall_clean.append((data_wall[idx].tolist()))
                    removed_idx.append(idx)

            # second, we want to keep everything above -3.760 and to the right of 4.500
            elif z1[idx] >= -3.790 and r1[idx] <= 5.000: # adjusted based on data i see in trial case
                data_wall_clean.append((data_wall[idx].tolist()))
                removed_idx.append(idx)

    # right-side. bins (z1, r1) = -4.260, 5.265 and -3.990, 5.230. also -3.900, 5.180 right above top point.
    for idx in range(len((r1))):
        # first, want to keep what is above -4.560 and to the right of 5.265
        if z1[idx] >= -4.560 and r1[idx] >= 5.265:
            data_wall_clean.append((data_wall[idx].tolist()))
            removed_idx.append(idx)

        # now we want to get that last part of the dome
        # this is above -3.990 and left of 5.265
        elif z1[idx] > -3.990 and r1[idx] < 5.265:
            if r1[idx] > 5 and r1[idx] > 5.06:
                data_wall_clean.append((data_wall[idx].tolist()))
                removed_idx.append(idx)

    return np.array(data_wall_clean), removed_idx

# remove indices for divertor data from all the cases for plotting purposes
def remove_indices(case, removed_idx):
    """
    remove indices that correspond to divertor infrastructure
    """
    
    wall_index = case[:,0]
    wall_lst = []
    
    for n in wall_index:
        wall_lst.append(int(n))


    case = case.tolist()

    for i in range(len(case)):
        if wall_lst[i] in removed_idx:
            del case[i]

    return np.array(case)

# now need to use these coordinates to create bins 
def create_bins(z_coord, r_coord):
    """
    uses tokamak geometry to create bins that will later be used to sort fluxes 

    parameters: input z and r coordinate numpy arrays of tokamak geometry, created by
    load_txt function
    """

    # initialize empty bin list
    bins = []

    # need to iterate directly through the z and r coordinate arrays to create bins 
    for coord in range(len(z_coord) - 1):
        z_start = z_coord[coord]
        z_end = z_coord[coord+1]
        r_start = r_coord[coord]
        r_end = r_coord[coord+1]

        # add these ranges to bins as a set of tuples
        bins.append(((z_start, z_end),(r_start, r_end)))

    return bins # bins is now a list of tuples, where each tuple has two tuples -- one for z_coords and one for r_coords

# now we want to bin our fluxes into these geometry-based bins 
def bin_fluxes(data, data_wall, data_div, bins, div_bins):
    """
    bin input flux data into bins as determined by tokamak geometry created in create_bins function 

    returns numpy arrays:
        binned ion and atom fluxes
        binned ion flux without divertor values
        binned atom flux without divertor values
        binned divertor ion flux
        binned divertor atom flux

    """

    # assign values to each column in data (from wdn)
    r1 = data[:,1].tolist()
    r2 = data[:,2].tolist()
    z1 = data[:,3].tolist()
    z2 = data[:,4].tolist()
    ion_flux = data[:,5].tolist()
    atom_flux = data[:,6].tolist()
    E_ion = data[:,7].tolist()
    E_atom = data[:,8].tolist()
    alpha_ion = data[:,9].tolist()
    alpha_atom = data[:,10].tolist()
    heat_ion_data = data[:,11]
    heat_atom = data[:,12]
    heat = heat_ion_data+heat_atom

    # do the same for wall data from soledge
    r1w = data_wall[:,0].tolist()
    r2w = data_wall[:,2].tolist()
    z1w = data_wall[:,1].tolist()
    z2w = data_wall[:,3].tolist()
    E_ionw = data_wall[:,8].tolist()
    E_atomw = data_wall[:,9].tolist()
    ion_fluxw = data_wall[:,-7].tolist()
    atom_fluxw = data_wall[:,-6].tolist()
    alpha_ionw = np.full(len(ion_fluxw),6.0e+01).tolist() # approximating as 40 degrees for all points
    alpha_atomw = np.full(len(atom_fluxw),4.5e+01).tolist() # approximating as perpendicular for all points
    # wall data has all of these values separated, so add all of them
    heat_ionw = data_wall[:,-3].tolist()
    # heat_atomw = data_wall[:,-2]+data_wall[:,-1]
    heat_wall = data_wall[:,-3]+data_wall[:,-2]+data_wall[:,-1]
    heat_wall = heat_wall.tolist()

    # and the same for divertor data from solps
    r1d = data_div[:,0].tolist()
    r2d = data_div[:,2].tolist()
    z1d = data_div[:,1].tolist()
    z2d = data_div[:,3].tolist()
    E_iond = data_div[:,8].tolist()
    E_atomd = data_div[:,9].tolist()
    ion_fluxd = data_div[:,11].tolist()
    atom_fluxd = data_div[:,12].tolist()
    alpha_iond = np.full(len(ion_fluxd),6.0e+01).tolist() # approximating as 40 degrees for all points
    alpha_atomd = np.full(len(atom_fluxw),4.5e+01).tolist() # approximating as perpendicular for all points
    heat_div = data_div[:,-1].tolist()
    heat_iond = data_div[:,-3].tolist()
    # heat_atomd = data_div[:,-2]+data_div[:,-1]
    # heat_atomd = heat_atomd.tolist()


    # r1 = r1w+r1d
    # r2 = r2w+r2d
    # z1 = z1w+z1d
    # z2 = z2w+z2d
    # E_ion = E_ionw+E_iond
    # E_atom = E_atomw+E_atomd
    # ion_flux = ion_fluxw + ion_fluxd
    # atom_flux = atom_fluxw + atom_fluxd
    # alpha_ion = alpha_ionw + alpha_iond
    # alpha_atom = alpha_atomw + alpha_atomd
    # heat = heat_wall + heat_div

    # need to create bins for the divertor, and have those be separate 
    # create dummy lists for binned divertor flux values
    # set up empty lists for binned divertor fluxes
    binned_div_ion = [[] for _ in range(len(div_bins))]
    binned_div_atom = [[] for _ in range(len(div_bins))]
    div_E_ion = [[] for _ in range(len(div_bins))]
    div_E_atom = [[] for _ in range(len(div_bins))]
    div_alpha_ion = [[] for _ in range(len(div_bins))]
    div_alpha_atom = [[] for _ in range(len(div_bins))]
    div_heat = [[] for _ in range(len(div_bins))]
    div_heat_ion = [[] for _ in range(len(div_bins))]
    
    # create dummy list for divertor indices
    div_indices = []

    # min z-value for the main chamber bins 
    min_z_start = min(z_start for z_start, _ in bins)[0]
    min_z_end = min(z_end for ((z1, z_end),(r1,r2)) in bins)

    # add indices to divertor indices if they are below the lowest bin coordinates 
    for idx, (z1_val, r1_val, z2_val) in enumerate(zip(z1, r1, z2)):
        if r1_val < 5:
            if z1_val < min_z_start:
                div_indices.append(idx)
        else:
            if z1_val < min_z_end:
                div_indices.append(idx)

    # setting up empty lists for the binned ion and atom wall fluxes
    binned_ion_flux = [[] for _ in range(len(bins+div_bins))]
    binned_atom_flux = [[] for _ in range(len(bins+div_bins))]
    wall_E_ion = [[] for _ in range(len(bins+div_bins))]
    wall_E_atom = [[] for _ in range(len(bins+div_bins))]
    wall_alpha_ion = [[] for _ in range(len(bins+div_bins))]
    wall_alpha_atom = [[] for _ in range(len(bins+div_bins))]
    wall_heat = [[] for _ in range(len(bins+div_bins))]
    wall_heat_ion = [[] for _ in range(len(bins+div_bins))]

    # need to iterate through each bin to set up the z and r bins and their center points
    # to later assign which fluxes have coordinates closest to each bin 
    bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in bins+div_bins]

    # do the same for divertor
    div_bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in div_bins]

    # keep track of which indices are going where 
    indices_covered_by_bin = [[] for _ in range(len(bins)+len(div_bins))]

    # create arrays that hold all data for data saving purposes 
    ion_flux_total = [[] for _ in range(len(ion_flux))]
    atom_flux_total = [[] for _ in range(len(atom_flux))]
    E_ion_total = [[] for _ in range(len(E_ion))]
    E_atom_total = [[] for _ in range(len(E_atom))]
    alpha_ion_total = [[] for _ in range(len(alpha_ion))]
    alpha_atom_total = [[] for _ in range(len(alpha_atom))]
    heat_total = [[] for _ in range(len(heat))]
    heat_ion = [[] for _ in range(len(heat))]

    # # # count for divertor bins 
    # count = 1

    # binning divertor fluxes first
    # iterate through each flux and calculate its distance from the nearest bin  
    for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_val, heat_ion) in enumerate(zip(z1d, z2d, r1d, r2d, ion_fluxd, atom_fluxd, E_iond, E_atomd, alpha_iond, alpha_atomd, heat_div, heat_iond)):
    # for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_ion_val, heat_atom_val) in enumerate(zip(z1d, z2d, r1d, r2d, ion_fluxd, atom_fluxd, E_iond, E_atomd, alpha_iond, alpha_atomd, heat_iond, heat_atomd)):
        distances = []
        mid_z = (z1_val + z2_val)/2
        mid_r = (r1_val + r2_val)/2

    # find the distance of this bin to each bin center
        for z_center, r_center in div_bin_centers:
            dist = np.sqrt((mid_z - z_center)**2 + (mid_r - r_center)**2)
            distances.append(dist)

        # take the minimum to determine which bin is closest
        closest_bin_index = np.argmin(distances)
        
        # if idx in div_indices:

        binned_div_ion[closest_bin_index].append(ion_val)
        binned_div_atom[closest_bin_index].append(atom_val)
        div_E_ion[closest_bin_index].append(E_ion_val) 
        div_E_atom[closest_bin_index].append(E_atom_val) 
        div_alpha_ion[closest_bin_index].append(alpha_ion_val)
        div_alpha_atom[closest_bin_index].append(alpha_atom_val)
        div_heat[closest_bin_index].append(heat_val)
        div_heat_ion[closest_bin_index].append(heat_ion)


        # to set up similar process to last time of adding all the values into one list
        closest_bin_index_new = len(bins) + closest_bin_index - 1

        indices_covered_by_bin[closest_bin_index_new].append(idx)

    # binning wall fluxes second 
    for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_val, heat_ion) in enumerate(zip(z1, z2, r1, r2, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat, heat_ion_data)):
    # for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_ion_val, heat_atom_val) in enumerate(zip(z1w, z2w, r1w, r2w, ion_fluxw, atom_fluxw, E_ionw, E_atomw, alpha_ionw, alpha_atomw, heat_ionw, heat_atomw)):
        distances = []
        mid_z = (z1_val + z2_val)/2
        mid_r = (r1_val + r2_val)/2


        # find the distance of this bin to each bin center
        for z_center, r_center in bin_centers:
            dist = np.sqrt((mid_z - z_center)**2 + (mid_r - r_center)**2)
            distances.append(dist)

        # take the minimum to determine which bin is closest
        closest_bin_index = np.argmin(distances)

        # if idx not in div_indices:

        binned_ion_flux[closest_bin_index].append(ion_val)
        binned_atom_flux[closest_bin_index].append(atom_val)
        wall_E_ion[closest_bin_index].append(E_ion_val)
        wall_E_atom[closest_bin_index].append(E_atom_val)
        wall_alpha_ion[closest_bin_index].append(alpha_ion_val)
        wall_alpha_atom[closest_bin_index].append(alpha_atom_val)
        wall_heat[closest_bin_index].append(heat_val)
        wall_heat_ion[closest_bin_index].append(heat_ion)

        # add index being covered to list
        indices_covered_by_bin[closest_bin_index].append(idx)

    total_indices_covered = sum(len(bin_indices) for bin_indices in indices_covered_by_bin)
    print(f"Total number of data points: {total_indices_covered}")

    # for divertor bins, need to combine wdn and solps data
    # wall_add = np.array(binned_ion_flux[18:])
    # solps_add = np.array(binned_div_ion)
    
    # div = wall_add+solps_add
    # div = div.tolist()
    # new = binned_ion_flux[:18]+div
    # print(len(new))


    # find mean flux for fluxes in each bin 
    binned_ion_flux = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_ion_flux]
    binned_atom_flux = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_atom_flux]
    wall_E_ion = [np.mean(energies) if energies else 0 for energies in wall_E_ion] 
    wall_E_atom = [np.mean(energies) if energies else 0 for energies in wall_E_atom]
    wall_alpha_ion = [np.mean(alphas) if alphas else 0 for alphas in wall_alpha_ion]
    wall_alpha_atom = [np.mean(alphas) if alphas else 0 for alphas in wall_alpha_atom]
    wall_heat = [np.mean(heat) if heat else 0 for heat in wall_heat]
    wall_heat_ion = [np.mean(heat) if heat else 0 for heat in wall_heat_ion]

    binned_div_ion = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_div_ion]
    binned_div_atom = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_div_atom]
    div_E_ion = [np.mean(energies) if energies else 0 for energies in div_E_ion] 
    div_E_atom = [np.mean(energies) if energies else 0 for energies in div_E_atom]
    div_alpha_ion = [np.mean(alphas) if alphas else 0 for alphas in div_alpha_ion]
    div_alpha_atom = [np.mean(alphas) if alphas else 0 for alphas in div_alpha_atom]
    div_heat = [np.mean(heat) if heat else 0 for heat in div_heat]
    div_heat_ion = [np.mean(heat) if heat else 0 for heat in div_heat_ion]

    ion_flux_total = binned_ion_flux + binned_div_ion
    atom_flux_total = binned_atom_flux + binned_div_atom
    E_ion_total = wall_E_ion + div_E_ion
    E_atom_total = wall_E_atom + div_E_atom 
    alpha_ion_total = wall_alpha_ion + div_alpha_ion
    alpha_atom_total = wall_alpha_atom + div_alpha_atom
    heat_total = wall_heat + div_heat
    heat_ion = wall_heat_ion + div_heat_ion

    
    # return all binned fluxes 
    return indices_covered_by_bin, div_indices, ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, binned_ion_flux, binned_atom_flux, binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat, wall_E_ion, wall_E_atom, wall_alpha_ion, wall_alpha_atom, wall_heat, heat_ion

# now we need a plotting function!
def binned_flux_plot(avg_ion_flux, avg_neutral_flux, indices_covered_by_bin, 
    div_indices, 
    bins, 
    wall_index, 
    data,
    data_div,
    data_wall,
    ion_flux_total, 
    atom_flux_total, 
    ion_fluxes, 
    neutral_fluxes,
    heat_total, 
    case_labels):
    """
    plots binned, averaged fluxes from input binned&averaged, data 

    parameters: 
        wall indexes for plotting 
        binned fluxes, including those for the divertor
        fluxes for all cases and their labels (see bg_fluxes_avgofcases.py for more information about these)
    """
    
    # want to plot on an x-axis that is representative of the wall indexes
    # create array to store binned values 

    r1w = data_wall[:,0].tolist()
    z1w = data_wall[:,1].tolist()
    ion_fluxw = data_wall[:,-7].tolist()
    atom_fluxw = data_wall[:,-6].tolist()
    heat_wall = data_wall[:,-3]+data_wall[:,-2]+data_wall[:,-1]
    heat_wall = heat_wall.tolist()

    # and the same for divertor data from solps
    # print(data_div)
    r1d = data_div[:,0].tolist()
    z1d = data_div[:,1].tolist()
    ion_fluxd = data_div[:,11].tolist()
    atom_fluxd = data_div[:,12].tolist()
    heat_div = data_div[:,-1].tolist()

    # r1 = r1w+r1d
    # z1 = z1w+z1d
    # ion_flux_raw = ion_fluxw + ion_fluxd
    # atom_flux_raw = atom_fluxw + atom_fluxd
    # heat_raw = heat_wall + heat_div

    r1 = data[:,1]
    z1 = data[:,3]
    ion_flux_raw = data[:,5]
    atom_flux_raw = data[:,6]
    heat_ion = data[:,11]
    heat_atom = data[:,12]
    heat_raw = heat_ion+heat_atom

    expanded_ion = np.zeros(len(r1))
    expanded_atom = np.zeros(len(r1))

    # extending with wall indices first, which are the first indices in indices_covered_by_bin
    for bin_indices, ion_flux, atom_flux in zip(indices_covered_by_bin[:len(bins)], ion_flux_total, atom_flux_total):
        for idx in bin_indices:
            expanded_ion[idx] = ion_flux  # all values in the bin are set to the mean flux for that bin
            expanded_atom[idx] = atom_flux

    # divertor indices expansion second, which are the indices after the wall bins 
    for bin_indices, ion_flux, atom_flux in zip(indices_covered_by_bin[len(bins):], ion_flux_total[len(bins):], atom_flux_total[len(bins):]):
        for idx in bin_indices:
            expanded_ion[idx] = ion_flux  # all values in the bin are set to the mean flux for that bin
            expanded_atom[idx] = atom_flux

    # FLUXES VS. WALL INDEX PLOTTING
    # can comment this out when integrate with IMAS
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # binned ion flux plotting 
    # for flux, label in zip(ion_fluxes, case_labels):
    #     ax1.plot(wall_index, flux, label=label)

    # ax1.plot(wall_index,expanded_ion, label="Estimated Flux")
    # ax1.set_yscale('log')
    # ax1.set_xlabel("Wall Index")
    # ax1.set_ylabel("Ion Fluxes")
    # ax1.set_title("Binned, Averaged Ion Fluxes")
    # ax1.legend()
    # ax1.grid(True)
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # ax1.yaxis.get_offset_text().set_fontsize(10)

    # # binned atom flux plotting 
    # for flux, label in zip(neutral_fluxes, case_labels):
    #     ax2.plot(wall_index, flux, label=label)

    # ax2.plot(wall_index, expanded_atom, label="Estimated Flux")
    # ax2.set_yscale('log')
    # ax2.set_xlabel("Wall Index")
    # ax2.set_ylabel("Neutral Fluxes")
    # ax2.set_title("Binned, Averaged Neutral Fluxes")
    # ax2.legend()
    # ax2.grid(True)
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # ax2.yaxis.get_offset_text().set_fontsize(10)

    # plt.tight_layout()
    # plt.savefig('plots/Binned_Fluxes.png')

    # FLUXES VS. DISTANCE (R1) PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # binned ion flux plotting
    ax1.plot(r1, avg_ion_flux, label="Average Flux")

    # with new IMAS data we do not care about plotting avg_fluxes because these are calculated from the cases
    # instead we want to plot the raw data from the shot 

    # ax1.plot(r1,ion_flux_raw, label="Raw Flux")

    ax1.plot(r1,expanded_ion, label="Estimated Flux")
    ax1.plot(r1d,ion_fluxd, label="SOLPS Flux")
    # ax1.plot(r1w,ion_fluxw, label='SOLEDGE Flux')
    ax1.set_yscale('log')
    ax1.set_xlabel("Radius (m)")
    ax1.set_ylabel("Ion Fluxes")
    ax1.set_title("Binned, Averaged Ion Fluxes Shot 122481")
    ax1.grid(True)
    ax1.legend()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax1.yaxis.get_offset_text().set_fontsize(10)

    # binned atom flux plotting 
    ax2.plot(r1, avg_neutral_flux, label="Average Flux")

    # ax2.plot(r1,atom_flux_raw, label="Raw Flux")
    ax2.plot(r1, expanded_atom, label="Estimated Flux")
    ax2.plot(r1d,atom_fluxd, label="SOLPS Flux")
    # ax2.plot(r1w,atom_fluxw, label='SOLEDGE Flux')
    ax2.set_yscale('log')
    ax2.set_xlabel("Radius (m)")
    ax2.set_ylabel("Neutral Fluxes")
    ax2.set_title("Binned, Averaged Neutral Fluxes Shot 122481")
    ax2.grid(True)
    ax2.legend()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax2.yaxis.get_offset_text().set_fontsize(10)

    plt.tight_layout()
    plt.savefig("plots/Binned_Fluxes_v._Distance.png")

    # binned heat plotting 
    expanded_heat = np.zeros(len(r1))

    # extending with wall indices first, which are the first indices in indices_covered_by_bin
    for bin_indices, heat in zip(indices_covered_by_bin[:len(bins)], heat_total):
        for idx in bin_indices:
            expanded_heat[idx] = heat  # all values in the bin are set to the mean flux for that bin

    # divertor indices expansion second, which are the indices after the wall bins 
    for bin_indices, heat in zip(indices_covered_by_bin[len(bins):], heat_total[len(bins):]):
        for idx in bin_indices:
            expanded_heat[idx] = heat  # all values in the bin are set to the mean flux for that bin

    plt.figure(figsize=(10,8))
    plt.plot(r1, expanded_heat, label="Estimated Heat")
    plt.plot(r1, heat_raw, label="Raw Heat")
    plt.yscale('log')
    plt.xlabel("Radius (m)")
    plt.ylabel("Heat W/m^2")
    plt.title("Binned, Averaged Total Heat Shot 122481")
    plt.grid(True)
    plt.legend()

    # # binned atom heat plotting 
    # ax2.plot(r1, expanded_heat_atom)
    # ax2.set_yscale('log')
    # ax2.set_xlabel("Radius (m)")
    # ax2.set_ylabel("Neutral Heat")
    # ax2.set_title("Binned, Averaged Neutral Heat")
    # ax2.grid(True)
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # ax2.yaxis.get_offset_text().set_fontsize(10)

    # plt.tight_layout()
    plt.savefig("plots/Binned_Heats_v._Distance.png")

# function to plot the bins we are using!
def bin_2D_plots(bins, div_bins, data_div, data_wall, div_indices):
    """
    creates 3 2D plots to visualize: 
        wall bins
        divertor bins
        all bins together

    parameters: 
        bins -- list of tuples of tuples where each tuple is (z-coords), (r-coords)
        div_bins -- same as above but for divertor
        data -- actual data points, used to test divertor bins (do not need) 
        div_indices -- divertor indices 
    """

    # main chamber plotting first 

    # initialize z and r coordinates
    wall_z = []
    wall_r = []
    end_r = []
    end_z = []

    # each point will be represented by the first z and r coordinates in each tuple set
    for (z_start, z_end), (r_start, r_end) in bins:
        wall_z.append(z_start)  # first z coordinate
        wall_r.append(r_start)  # first r coordinate
        end_r.append(r_end) # to pull out very last point
        end_z.append(z_end) # to pull out very last point

    wall_z.append(end_z[len(end_z)-1])
    wall_r.append(end_r[len(end_r)-1])

    # plotting
    plt.figure(figsize=(8, 14))
    plt.plot(wall_r, wall_z, color='green')
    plt.scatter(wall_r, wall_z, marker='.', s=200, color='green')

    # add bin labels to plot
    for i in range(1, len(wall_r)):
        mid_r = (wall_r[i-1] + wall_r[i]) / 2 +0.2
        mid_z = (wall_z[i-1] + wall_z[i]) / 2
        plt.text(mid_r, mid_z, str(i), fontsize=25, ha='center', va='center', color='green')

    plt.xlabel('Radius: R-Axis (m)', fontsize=25)
    plt.ylabel('Height: Z-Axis (m)', fontsize=25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.title('Main Chamber Bins', fontsize=25)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/Main_Chamber_Bins")

    # divertor plotting second 
    # assign values to each column in data
    r1d = data_div[:,0]
    z1d = data_div[:,1]

    # r1w = data_wall[:,0]
    # z1w = data_wall[:,1]

    r1w = data_wall[:,1]
    z1w = data_wall[:,3]

    div_z_lst = []
    div_r_lst = []

    for item in range(len(r1)):
        if item in div_indices:
            div_z_lst.append(z1[item])
            div_r_lst.append(r1[item])

    # when plotting wdn data
    # actual_bins = create_bins(div_z_lst, div_r_lst)
    # actual_z = []
    # actual_r = []
    # end_ac_z = []
    # end_ac_r = []
    # each point will be represented by the first z and r coordinates in each tuple set
    # for (z_start, z_end), (r_start, r_end) in actual_bins:
    #     actual_z.append(z_start)  # first z coordinate
    #     actual_r.append(r_start)  # first r coordinate
    #     end_ac_r.append(r_end) # to pull out very last point
    #     end_ac_z.append(z_end) # to pull out very last point
    # actual_z.append(end_ac_z[len(end_ac_z)-1])
    # actual_r.append(end_ac_r[len(end_ac_r)-1])

    
    div_z = []
    div_r = []
    end_div_z = []
    end_div_r = []

    # each point will be represented by the first z and r coordinates in each tuple set
    for (z_start, z_end), (r_start, r_end) in div_bins:
        div_z.append(z_start)  # first z coordinate
        div_r.append(r_start)  # first r coordinate
        end_div_r.append(r_end) # to pull out very last point
        end_div_z.append(z_end) # to pull out very last point

    div_z.append(end_div_z[len(end_div_z)-1])
    div_r.append(end_div_r[len(end_div_r)-1])

    
    # plotting
    plt.figure(figsize=(20, 20))
    plt.plot(div_r, div_z, color='green')
    plt.scatter(div_r, div_z, marker='+', s=200, color='black')
    plt.xlabel('Radius: R-Axis (m)', fontsize=35)
    plt.ylabel('Height: Z-Axis (m)', fontsize=35)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.title('Divertor Bins', fontsize = 35)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/Divertor_Bins")

    # plotting it all together 
    plt.figure(figsize=(15, 30))
    plt.scatter(div_r, div_z, marker='+', s=200, color='green', label='Main Chamber')
    plt.scatter(wall_r, wall_z, marker='+', s=200, color='black', label='Divertor')
    plt.plot(wall_r+div_r, wall_z+div_z, color='green')
    # plt.scatter(r1d, z1d, marker='+', s=200, color='blue', label='Actual Divertor Points')
    # plt.scatter(r1w, z1w, marker='+', s=200, color='blue', label='Actual Wall Points')
    plt.xlabel('Radius: R-Axis (m)', fontsize=35)
    plt.ylabel('Height: Z-Axis (m)', fontsize=35)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.title('Bins', fontsize=35)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/All_Bins")
    

if __name__ == "__main__":

    # reading average flux data from .dat file
    data = read_dat_file('Background_Flux_Data')

    inner_data, outer_data, data_div = read_div_solps('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/fp_tg_i.2481.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/fp_tg_o.2481.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/ld_tg_i.2481.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/ld_tg_o.2481.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/inner_target.shot122481.run1.dat', '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/outer_target.shot122481.run1.dat')
    data_wall1 = read_wall_soledge('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/wall.shot106000.run1.dat')
    data_wall, removed_idx = remove_structure_points_soledge(data_wall1)

    # create bins for fluxes
    z_coord, r_coord = load_geometry("FWpanelcorners.txt")
    div_z, div_r = load_geometry('RISPDivpanelscorners.txt', wall=False)
    bins = create_bins(z_coord, r_coord)
    div_bins = create_bins(div_z, div_r)
    print(f'There are {len(bins)+len(div_bins)} bins.')

    # get average binned fluxes
    indices, div_indices, ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, binned_ion_flux, binned_atom_flux, binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat, wall_E_ion, wall_E_atom, wall_alpha_ion, wall_alpha_atom, wall_heat, heat_ion = bin_fluxes(data, data_wall, data_div, bins, div_bins)

    # save wall fluxes to .dat file
    header = "Z_Coord             \tR_Coord                      \tFlux_Ion                      \tFlux_Atom                          \tE_ion                          \tE_atom              \talpha_ion           \talpha_atom         \theat_total"
    data_to_save = np.array([
        [z_coord, r_coord, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat]
        for ((z_coord, _), (r_coord, _)), ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat in zip(bins, binned_ion_flux, binned_atom_flux, wall_E_ion, wall_E_atom, wall_alpha_ion, wall_alpha_atom, wall_heat)
    ])
    np.savetxt("Wall_Flux_Data.dat", data_to_save, delimiter='\t', header=header, comments='')

    # save divertor fluxes to .dat file
    header = "Z_Coord             \tR_Coord                      \tFlux_Ion                      \tFlux_Atom                          \tE_ion                          \tE_atom              \talpha_ion           \talpha_atom         \theat_total"
    data_to_save = np.array([
        [z_coord, r_coord, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat]
        for ((z_coord, _), (r_coord, _)), ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat in zip(div_bins, binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat)
    ])
    np.savetxt("Divertor_Flux_Data.dat", data_to_save, delimiter='\t', header=header, comments='')

    print(list(range(len(bins)+len(div_bins))))
    # save total fluxes to .dat file
    header = "Bin_Index,Flux_Ion,Flux_Atom,E_ion,E_atom,alpha_ion,alpha_atom,heat_total,heat_ion"
    data_to_save = np.array([
        [bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat, heat_ion_val]
        for bin_index, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat, heat_ion_val in zip(list(range(len(bins)+len(div_bins))), ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, heat_ion)
    ])
    np.savetxt("Binned_Flux_Data.dat", data_to_save, delimiter=',', header=header, comments='', fmt=['%d']  + ['%.18e'] * (data_to_save.shape[1] - 1)) #, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])


    # plotting other cases on top of the ones we have calculated
    # reading in all 8 scenarios for analysis 
    case_files = [
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-00d-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-00g-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-00k-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-00m-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-01d-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-01g-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-01k-Ne.mrt.wall_flux',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wdn-data/i-wdn-0003-2481-01m-Ne.mrt.wall_flux'
    ]
    
    cases = [read_flux_file(filename) for filename in case_files]

    # plot and save the data
    if all(case is not None for case in cases):
        # same wall index for all cases
        wall_index = cases[0][:, 0]  

        # reading coordinates for each case and storing in master list
        r1 = cases[0][:,1]
        r2 = cases[0][:,3]
        z1 = cases[0][:,2]
        z2 = cases[0][:,4]

        # reading coordinates for plotting 
        r1w = data_wall[:,0]
        r1d = data_div[:,0]
        z1d = data_div[:,1]
        z1w = data_wall[:,1]

        # reading ion flux for each case and storing in master list
        ion_fluxes = [case[:, 6] for case in cases]

        # same as ion fluxes but with neutrals here 
        neutral_fluxes = [case[:, 9] for case in cases]

        # labeling each case for legend on graphs 
        case_labels = [f"Case {i+1}" for i in range(len(cases))]

        # plotting
        avg_ion_flux, avg_neutral_flux = plot_data(wall_index, r1, z1, ion_fluxes, neutral_fluxes, case_labels)
        binned_flux_plot(avg_ion_flux, avg_neutral_flux, indices, div_indices, bins, wall_index, data, data_div, data_wall, ion_flux_total, atom_flux_total, ion_fluxes, neutral_fluxes, heat_total, case_labels)


    else:
        print("Error: Failed to read one or more data files.")

    bin_2D_plots(bins, div_bins, data_div, data, div_indices)