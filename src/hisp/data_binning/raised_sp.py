import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import sys
from scipy.interpolate import interp1d
import subprocess
import shutil
import matplotlib.patches as mpatches

from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from scipy.interpolate import UnivariateSpline
from bg_fluxes_binned import remove_structure_points_soledge, read_wall_soledge, read_dat_file, load_geometry, create_bins

# produces average flux files for raised SP scenario

def read_DINA(DINA_data):
    """
    reads DINA excel data
    
    parameters:
    DINA data in .txt file from excel 

    returns: 
    DINA data in numpy array 
    """

    with open(DINA_data) as file:
            lines = [line for line in file]
        
    DINA = np.loadtxt(lines, skiprows=1)

    return DINA

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
    inner_lines_script[:,10] = inner_particle_flux[-1][2]# Tn -- equal to Ti for now
    inner_lines_script[:,11] = inner_particle_flux[:-1,4] # flxi
    inner_lines_script[:,12] = inner_particle_flux[:-1,1]# flxn
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
    inner_lines_script[-1][10] = inner_particle_flux[-1][2] # Tn, setting them roughly equal for now
    inner_lines_script[-1][11] = inner_particle_flux[-1][4] # ion flux flxi
    inner_lines_script[-1][12] = inner_particle_flux[-1][1] # atom flux flxn
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
    outer_lines_script[:,10] = outer_particle_flux[:-1,2]# Tn -- equal to Ti for now
    outer_lines_script[:,11] = outer_particle_flux[:-1,4] # flxi
    outer_lines_script[:,12] = outer_particle_flux[:-1,1]# flxn
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
    outer_lines_script[-1][10] = outer_power_loads[-1][2]# Tn, setting them roughly equal for now
    outer_lines_script[-1][11] = outer_particle_flux[-1][4] # ion flux flxi
    outer_lines_script[-1][12] = outer_particle_flux[-1][1] # atom flux flxn
    # outer_lines_script[-1][13] = outer_particle_flux[-1][-2] # fuel molecule pressure
    # now add in the last power fluxes here 
    outer_lines_script[-1][13] = outer_power_loads[-1][4] # Wtot
    outer_lines_script[-1][14] = outer_power_loads[-1][7] # Wion

    # finally, combine these values into the final script 
    data_div = np.concatenate((inner_lines_script, outer_lines_script))

    return inner_lines_script, outer_lines_script, data_div

# need to see which bins are covered by sweep (can be used for both inner and outer)
def find_xc(data, div_bins, inner_flag=True):
    """
    determines which bins are covered by innersp points 
    """

    # instead of keeping track of the fluxes, we are keeping track of the xc coordinates
    # sort those by index instead of fluxes 
    r1 = data[:,0].tolist()
    r2 = data[:,2].tolist()
    z1 = data[:,1].tolist()
    z2 = data[:,3].tolist()
    heat = data[:,-2]
    xc_solps = data[:,6]

    # print(xc_solps)
    xc_bins = [[] for _ in range(len(div_bins))]

    # INNER SP:
    if inner_flag: 

        # div bin centers
        div_bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in div_bins[27:]]

        z_points = []
        r_points = []

        for (z_start, z_end), (r_start, r_end) in div_bins[27:]:
            z_points.append(z_start)
            r_points.append(r_start)

        x_ref = 0

        # find the xc of each bin using distance between bin edges and bin centers
        for idx, (z_val, r_val, (z_center, r_center)) in enumerate(zip(z_points, r_points, div_bin_centers)):
            dist = np.sqrt((z_val - z_center)**2 + (r_val - r_center)**2)
            xc_bins[idx+27].append(x_ref+dist)
            x_ref+= 2*dist


    else:
        # div bin centers
        div_bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in div_bins[:15]]
        
        z_points = []
        r_points = []

        for (z_start, z_end), (r_start, r_end) in div_bins[:15]:
            z_points.append(z_start)
            r_points.append(r_start)

        x_ref = 0

        # find the xc of each bin using distance between bin edges and bin centers
        for idx, (z_val, r_val, (z_center, r_center)) in enumerate(zip(z_points, r_points, div_bin_centers)):
            dist = np.sqrt((z_val - z_center)**2 + (r_val - r_center)**2)
            xc_bins[14-idx].append(x_ref+dist) # our new bins!
            x_ref+= 2*dist
    
    xc_bins = [np.mean(xc) if xc else 0 for xc in xc_bins]

    return heat, xc_bins

# find the change in SP position with time
def plot_sp_time(time, sp, name):
    """
    plots the SP x coordinate against time as given by 
    DINA file

    returns: slope of line (rate of change of SP point)
    """

    plt.figure()
    plt.plot(time, sp, label="DINA Data")
    plt.xlabel('Time (s)')
    plt.ylabel(name+'Location (m)')
    plt.title('Change in'+name+'Location in Time')

    # line evens out at some point 
    slope, lowest_point = np.polyfit(time[6685:],sp[6685:],1)
    estimated_y = time*slope+lowest_point

    # removing these evened out points from the polyfit so we get an accurate line
    straight_line = []

    for x,y in zip(time, sp):
        if y < estimated_y[-1]:
            straight_line.append(x)

    time = time.tolist()

    last_pt = time.index(straight_line[0])

    time = time[:last_pt]
    sp = sp[:last_pt]
    lowest_sp = sp[last_pt-1]

    # added_time = np.arange(300,315.16,)
    time = np.array(time)

    # now that we have removed the horizontal part, we can 
    # fit the line 
    slope, intercept = np.polyfit(time, sp, 1)
    # print(time)
    plt.plot(time, time*slope+intercept, label="Fitted Data")
    plt.legend()
   
    plt.savefig('Dina_Raised_'+name+'_on_Time')

    total_time = time[-1] - time[0]

    return slope, total_time, lowest_sp

# create time dependent heat profile 
def time_dep(lowest_sp, heat_profile, data, slope, ramp_up, constant_in_time, steady_state, ramp_down, xc_bins, angle_file_list, inner=True):
    """
    creates time dependent lists for use in mhims simulation 
    """

    # fluxes and energies
    ion_flux = data[:,11]
    atom_flux = data[:,12]
    ion_E = data[:,9]
    atom_E = data[:,10]
    xc_solps = data[:,6]
    r1 = data[:,0]
    r2 = data[:,2]
    z1 = data[:,1]
    z2 = data[:,3]

    # creating time profile for our sweep 
    time = list(range(ramp_up+constant_in_time+steady_state+1))
    x_shifted = [[] for _ in range(len(time))]
    master_list = [[] for _ in range(len(time))]
    sweep_time = steady_state-constant_in_time
    
    # xc_solps=xc_solps.tolist()
    slopeslow = -6.131e-03
    slopefast = -8.434e-03
    sweep_time_half = sweep_time/2
    max_sweep_x = -slopefast*sweep_time_half-slopeslow*sweep_time_half+lowest_sp # max position of the sweep

    # creating heat profile for each timestep
    for idx, t in enumerate(time): 
        xc_long_solps = []
        if t > ramp_up + constant_in_time and t < ramp_up+steady_state:

            if inner:
                for x in xc_solps:
                    if x > xc_bins[-7]: # slow slope
                        shift = max_sweep_x+slopeslow*(t-(ramp_up+constant_in_time+1))+x
                        xc_long_solps.append(shift)
                    else: # high slope
                        shift = max_sweep_x+slopefast*(t-(ramp_up+constant_in_time+1))+x
                        xc_long_solps.append(shift)

            else: # for inner = False
                for x in xc_solps:
                    shift = max_sweep_x+slope*(t-(ramp_up+constant_in_time+1))+x
                    # shift = lowest_sp-slope*sweep_time+slope*(t-(ramp_up+constant_in_time+1))+x
                    xc_long_solps.append(shift)

            x_shifted[idx].append(xc_long_solps)

        else:
            shift = lowest_sp+xc_solps
            x_shifted[idx].append(shift.tolist())

    # adjusting fluxes and heats for angle dependency on inner SP

    for idx, xarray in enumerate(x_shifted):

        for xarray2 in xarray:
           
            master_list[idx].append(xarray2)
            master_list[idx].append(ion_flux.tolist())
            master_list[idx].append(atom_flux.tolist())
            master_list[idx].append(ion_E.tolist())
            master_list[idx].append(atom_E.tolist())
            master_list[idx].append(heat_profile.tolist())

    plt.figure()

    for idx, array in enumerate(x_shifted):
        for array2 in array:
            plt.plot(array2, heat_profile, color='red')
            plt.plot(array2, ion_flux, color='blue')
            plt.plot(array2, atom_flux, color='green')

    red_patch = mpatches.Patch(color='red', label='Heat Profile')
    blue_patch = mpatches.Patch(color='blue', label='Ion Fluxes')
    green_patch = mpatches.Patch(color='green', label='Atom Fluxes')

    plt.legend(handles=[red_patch, blue_patch, green_patch])

    plt.xlabel('X Coordinate (m)')

    if inner:
        plt.title('SOLPS Inner Target Heat Load Profile')
        plt.savefig('plots/Profile_Inner')
    else:
        plt.title('SOLPS Outer Target Heat Load Profile')
        plt.savefig('plots/Profile_Outer')

    # fluxes
    plt.figure()

    for idx, array in enumerate(master_list):
        plt.plot(master_list[idx][0], ion_flux, label="SOLPS Ion Flux Profile at "+str(idx)+'s')

    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Ion Flux (m^-2)')

    if inner:
        plt.title('SOLPS Inner Target Ion Flux Profile')
        plt.savefig('plots/Ion_Flux_Inner')
    else:
        plt.title('SOLPS Outer Target Ion Flux Profile')
        plt.savefig('plots/Ion_Flux_Outer')

    plt.figure()

    for idx, array in enumerate(x_shifted):
        for array2 in array:
            plt.plot(array2, atom_flux, label="SOLPS Ion Flux Profile at "+str(idx)+'s')

    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Atom Flux (m^-2)')

    if inner:
        plt.title('SOLPS Inner Target Atom Flux Profile')
        plt.savefig('plots/Atom_Flux_Inner')
    else:
        plt.title('SOLPS Outer Target Atom Flux Profile')
        plt.savefig('plots/Atom_Flux_Outer')

    # energies
    plt.figure()

    for idx, array in enumerate(x_shifted):
        for array2 in array:
            plt.plot(array2, ion_E, label="SOLPS Ion Energy Profile at "+str(idx)+'s')

    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Ion Energy')

    if inner: 
        plt.title('SOLPS Inner Target Ion Energy Profile')
        plt.savefig('plots/Ion_Energy_Inner')
    else:
        plt.title('SOLPS Outer Target Ion Energy Profile')
        plt.savefig('plots/Ion_Energy_Outer')

    plt.figure()

    for idx, array in enumerate(x_shifted):
        for array2 in array:
            plt.plot(array2, atom_E, label="SOLPS Ion Energy Profile at "+str(idx)+'s')

    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Atom Energy')

    if inner: 
        plt.title('SOLPS Inner Target Atom Energy Profile')
        plt.savefig('plots/Atom_Energy_Inner')
    else:
        plt.title('SOLPS Outer Target Atom Energy Profile')
        plt.savefig('plots/Atom_Energy_Outer')


    # keeping this for reference later, code done with Tom on 9/7/24
    # time=10 and time=100
    # plt.plot(0.144+xc_for_sort, heat_profile, color='red', label="SOLPS Heat Load Profile")
    # # time=101
    # plt.plot(0.144-slope_in*160+xc_for_sort, heat_profile, color='red', label="SOLPS Heat Load Profile")
    # # time=101+4
    # plt.plot(0.144-slope_in*(160-4)+xc_for_sort, heat_profile, color='red', label="SOLPS Heat Load Profile")
    # # time=101+8
    # plt.plot(0.144-slope_in*(160-8)+xc_for_sort, heat_profile, color='red', label="SOLPS Heat Load Profile")

    return x_shifted, master_list

# taking inspiration from binned fluxes file, need to bin using the SOLEDGE
# low power data
def bin_SOLEDGE_fluxes(data_wall, data_div, bins, div_bins):
    """
    bin SOLEDGE data only -- reminder that this data is frozen in time

    returns numpy arrays:
        binned ion and atom fluxes
        binned ion flux without divertor values
        binned atom flux without divertor values
        binned divertor ion flux
        binned divertor atom flux

    """

    # extract data values for wall data from SOLEDGE
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
    heat_ion_wall = data_wall[:,-3].tolist()
    heat_wall = heat_ion_wall+data_wall[:,-2]+data_wall[:,-1]
    heat_wall = heat_wall.tolist()

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
    for idx, (z1_val, r1_val, z2_val) in enumerate(zip(z1w, r1w, z2w)):
        if r1_val < 5:
            if z1_val < min_z_start:
                div_indices.append(idx)
        else:
            if z1_val < min_z_end:
                div_indices.append(idx)

    # setting up empty lists for the binned ion and atom wall fluxes
    binned_ion_flux = [[] for _ in range(len(bins))]
    binned_atom_flux = [[] for _ in range(len(bins))]
    wall_E_ion = [[] for _ in range(len(bins))]
    wall_E_atom = [[] for _ in range(len(bins))]
    wall_alpha_ion = [[] for _ in range(len(bins))]
    wall_alpha_atom = [[] for _ in range(len(bins))]
    wall_heat = [[] for _ in range(len(bins))]
    wall_heat_ion = [[] for _ in range(len(bins))]

    # need to iterate through each bin to set up the z and r bins and their center points
    # to later assign which fluxes have coordinates closest to each bin 
    bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in bins]

    # do the same for divertor
    div_bin_centers = [(np.mean(z_bin), np.mean(r_bin)) for z_bin, r_bin in div_bins]

    # keep track of which indices are going where 
    indices_covered_by_bin = [[] for _ in range(len(bins)+len(div_bins))]

    # create arrays that hold all data for data saving purposes 
    ion_flux_total = [[] for _ in range(len(ion_fluxw))]
    atom_flux_total = [[] for _ in range(len(atom_fluxw))]
    E_ion_total = [[] for _ in range(len(E_ionw))]
    E_atom_total = [[] for _ in range(len(E_atomw))]
    alpha_ion_total = [[] for _ in range(len(alpha_ionw))]
    alpha_atom_total = [[] for _ in range(len(alpha_atomw))]
    heat_total = [[] for _ in range(len(heat_wall))]
    heat_ion_total = [[] for _ in range(len(heat_wall))]

    # binning divertor SOLEDGE fluxes first
    # iterate through each flux and calculate its distance from the nearest bin  
    for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_val, heat_ion_val) in enumerate(zip(z1w, z2w, r1w, r2w, ion_fluxw, atom_fluxw, E_ionw, E_atomw, alpha_ionw, alpha_atomw, heat_wall, heat_ion_wall)):
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

        if idx in div_indices:
            binned_div_ion[closest_bin_index].append(ion_val)
            binned_div_atom[closest_bin_index].append(atom_val)
            div_E_ion[closest_bin_index].append(E_ion_val) 
            div_E_atom[closest_bin_index].append(E_atom_val) 
            div_alpha_ion[closest_bin_index].append(alpha_ion_val)
            div_alpha_atom[closest_bin_index].append(alpha_atom_val)
            div_heat[closest_bin_index].append(heat_val)
            div_heat_ion[closest_bin_index].append(heat_ion_val)

            # to set up similar process to last time of adding all the values into one list
            closest_bin_index_new = len(bins) + closest_bin_index - 1

            indices_covered_by_bin[closest_bin_index_new].append(idx)

    # binning wall fluxes second 
    for idx, (z1_val, z2_val, r1_val, r2_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_val, heat_ion_val) in enumerate(zip(z1w, z2w, r1w, r2w, ion_fluxw, atom_fluxw, E_ionw, E_atomw, alpha_ionw, alpha_atomw, heat_wall, heat_ion_wall)):
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

        if idx not in div_indices:
            binned_ion_flux[closest_bin_index].append(ion_val)
            binned_atom_flux[closest_bin_index].append(atom_val)
            wall_E_ion[closest_bin_index].append(E_ion_val)
            wall_E_atom[closest_bin_index].append(E_atom_val)
            wall_alpha_ion[closest_bin_index].append(alpha_ion_val)
            wall_alpha_atom[closest_bin_index].append(alpha_atom_val)
            wall_heat[closest_bin_index].append(heat_val)
            wall_heat_ion[closest_bin_index].append(heat_ion_val)

            # add index being covered to list
            indices_covered_by_bin[closest_bin_index].append(idx)

    total_indices_covered = sum(len(bin_indices) for bin_indices in indices_covered_by_bin)

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
    heat_ion_total = wall_heat_ion + div_heat_ion

    # return all binned fluxes 
    return indices_covered_by_bin, div_indices, ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, binned_ion_flux, binned_atom_flux, binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat, wall_E_ion, wall_E_atom, wall_alpha_ion, wall_alpha_atom, wall_heat, heat_ion_total

def bin_SOLPS_fluxes(div_bins, master_list, xc_bins, t, inner=True):
    """
    bin SOLPS data only -- time dependent data, the only bins we care about are the inner and outer strike point bins 
    need to run this separately for inner and outer SPs
    here, div_bins is inner or outer SP bins as a list of bin #

    returns numpy arrays:
        binned ion and atom fluxes
        binned ion flux without divertor values
        binned atom flux without divertor values
        binned divertor ion flux
        binned divertor atom flux

    """

    # extract values from master_list solps data, all already are lists
    x_d = master_list[0]
    # print(x_d)
    E_iond = master_list[3]
    E_atomd = master_list[4]
    ion_fluxd = master_list[1]
    atom_fluxd = master_list[2]
    alpha_iond = np.full(len(ion_fluxd),6.0e+01).tolist() # approximating as 40 degrees for all points
    alpha_atomd = np.full(len(atom_fluxd),4.5e+01).tolist() # approximating as perpendicular for all points
    heat_div = master_list[-1]

    # need to create bins for the divertor, and have those be separate 
    # create dummy lists for binned divertor flux values
    # set up empty lists for binned divertor fluxes
    binned_div_ion = [[] for _ in range(len(xc_bins))]
    binned_div_atom = [[] for _ in range(len(xc_bins))]
    div_E_ion = [[] for _ in range(len(xc_bins))]
    div_E_atom = [[] for _ in range(len(xc_bins))]
    div_alpha_ion = [[] for _ in range(len(xc_bins))]
    div_alpha_atom = [[] for _ in range(len(xc_bins))]
    div_heat = [[] for _ in range(len(xc_bins))]

    # we have the xc coordinate for each bin
    for idx, (x_val, ion_val, atom_val, E_ion_val, E_atom_val, alpha_ion_val, alpha_atom_val, heat_val) in enumerate(zip(x_d, ion_fluxd, atom_fluxd, E_iond, E_atomd, alpha_iond, alpha_atomd, heat_div)):

        distances = []
        

        # find the distance of this bin to each bin center
        for x_cent in xc_bins:
            # print(x_cent)
            # print(x_val)
            dist = abs(abs(x_cent) - abs(x_val))
            # print(dist)
            distances.append(dist)

        # take the minimum to determine which bin is closest
        closest_bin_index = np.argmin(distances)
        # print(closest_bin_index)

        binned_div_ion[closest_bin_index].append(ion_val)
        binned_div_atom[closest_bin_index].append(atom_val)
        div_E_ion[closest_bin_index].append(E_ion_val) 
        div_E_atom[closest_bin_index].append(E_atom_val) 
        div_alpha_ion[closest_bin_index].append(alpha_ion_val)
        div_alpha_atom[closest_bin_index].append(alpha_atom_val)
        div_heat[closest_bin_index].append(heat_val)

    binned_div_ion = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_div_ion]
    binned_div_atom = [np.mean(fluxes) if fluxes else 0 for fluxes in binned_div_atom]
    div_E_ion = [np.mean(energies) if energies else 0 for energies in div_E_ion] 
    div_E_atom = [np.mean(energies) if energies else 0 for energies in div_E_atom]
    div_alpha_ion = [np.mean(alphas) if alphas else 0 for alphas in div_alpha_ion]
    div_alpha_atom = [np.mean(alphas) if alphas else 0 for alphas in div_alpha_atom]
    div_heat = [np.mean(heat) if heat else 0 for heat in div_heat]

    if inner:
        if t >= ramp_up+constant_in_time+1 and t < ramp_up+steady_state+1:
        # opening correct angle coeff profile based on timestep
        # need to get correct idx relationships
            filename = angle_coeffs_matlab[t-(ramp_up+constant_in_time+1)]
            with open(filename) as file: # indexing matches at t=101 (means t=1 for angle coeffs)
                lines = [line for line in file]
        
            coeffs = np.loadtxt(lines)

            # multiplying heats and fluxes by profile
            for q in range(len(div_heat)):
                div_heat[q] = div_heat[q]*coeffs[q]
                binned_div_ion[q] = binned_div_ion[q]*coeffs[q]

    # return all binned fluxes 
    return binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat 



if __name__ == '__main__':

    DINA_inner = read_DINA('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/RISP_excel.txt')
    DINA_outer = read_DINA('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/ROSP_excel.txt')

    time = DINA_inner[:,0]
    innersp = DINA_inner[:,1]
    outersp = DINA_outer[:,1]

    os.system('rm -r RISP_data')
    os.system('rm -r ROSP_data')
    os.mkdir('RISP_data')
    os.mkdir('ROSP_data')

    inner_data, outer_data, data_div = read_div_solps('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/raised_SP_solps/fp_tg_i.123102.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/raised_SP_solps/fo_tg_o.123102.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/raised_SP_solps/ld_tg_i.123102.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/solps_data/raised_SP_solps/ld_tg_o.123102.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/inner_target.shot123102.run1.dat','/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/outer_target.shot123102.run1.dat')
    # np.savetxt('inner.txt', inner_data)

    # create bins for fluxes
    div_z, div_r = load_geometry('RISPDivpanelscorners.txt', wall=False)
    z_coord, r_coord = load_geometry("FWpanelcorners.txt")
    bins = create_bins(z_coord, r_coord)
    div_bins = create_bins(div_z, div_r)

    # find center xc values for our bins 
    inner_heat, inner_xc_bins = find_xc(inner_data,div_bins)
    outer_heat, outer_xc_bins = find_xc(outer_data,div_bins, inner_flag=False)

    np.savetxt('inner_heat', inner_heat)

    inner_xc_bins=inner_xc_bins[27:]
    outer_xc_bins=outer_xc_bins[:15]

    # setting swept SOLPS bins as all bins on inner and outer strike points
    
    inner_bins = list(range(46,len(div_bins+bins)))
    outer_bins = list(range(18,34))

    # find rate of change of inner and outer sp coordinates
    slope_in, total_time, lowest_in = plot_sp_time(time, innersp, 'Inner_SP')
    slope_out, total_time, lowest_out = plot_sp_time(time, outersp, 'Outer_SP')

    # time values for SOLPS simulation from paper 
    ramp_up = 10
    constant_in_time = 90
    steady_state= constant_in_time+160
    ramp_down=10

    angle_coeffs_matlab = [
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t1.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t2.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t3.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t4.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t5.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t6.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t7.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t8.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t9.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t10.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t11.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t12.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t13.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t14.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t15.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t16.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t17.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t18.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t19.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t20.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t21.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t22.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t23.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t24.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t25.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t26.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t27.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t28.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t29.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t30.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t31.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t32.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t33.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t34.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t35.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t36.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t37.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t38.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t39.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t40.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t41.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t42.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t43.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t44.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t45.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t46.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t47.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t48.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t49.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t50.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t51.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t52.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t53.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t54.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t55.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t56.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t57.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t58.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t59.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t60.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t61.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t62.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t63.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t64.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t65.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t66.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t67.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t68.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t69.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t70.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t71.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t72.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t73.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t74.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t75.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t76.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t77.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t78.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t79.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t80.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t81.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t82.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t83.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t84.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t85.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t86.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t87.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t88.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t89.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t90.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t91.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t92.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t93.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t94.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t95.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t96.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t97.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t98.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t99.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t100.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t101.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t102.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t103.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t104.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t105.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t106.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t107.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t108.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t109.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t110.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t111.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t112.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t113.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t114.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t115.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t116.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t117.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t118.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t119.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t120.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t121.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t122.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t123.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t124.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t125.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t126.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t127.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t128.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t129.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t130.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t131.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t132.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t133.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t134.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t135.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t136.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t137.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t138.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t139.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t140.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t141.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t142.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t143.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t144.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t145.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t146.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t147.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t148.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t149.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t150.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t151.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t152.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t153.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t154.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t155.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t156.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t157.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t158.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t159.txt',
        '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/angles_on_div/angle_coeffs_t160.txt'
    ]

    # inner heat profile
    x_inner, master_list_inner = time_dep(lowest_in, inner_heat, inner_data, slope_in,ramp_up, constant_in_time, steady_state, ramp_down, inner_xc_bins, angle_coeffs_matlab)
    np.savetxt('inner_master.txt', master_list_inner[200][-1])
    
    # outer heat profile
    x_outer, master_list_outer = time_dep(lowest_out, outer_heat, outer_data, slope_out,ramp_up, constant_in_time, steady_state, ramp_down, outer_xc_bins, angle_coeffs_matlab, inner=False)

    # xc_solps = data[:,6]+0.144
    max_x = []
    for x in master_list_inner:
        max_x.append(max(x[0]))


    # plt.figure()
    # plt.plot(master_list_inner[149][0], master_list_inner[149][1], label='Ion flux (m^-2)')
    # plt.plot(master_list_inner[149][0], master_list_inner[149][2], label='Atom flux (m^-2)')
    # plt.plot(master_list_inner[149][0], master_list_inner[149][-1], label='Heat (W/m^2)')
    # plt.yscale('log')
    # plt.title('Profiles vs. Time at t=150s')
    # plt.xlabel('X Coord (m)')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('Profiles_Time')

    print(f'Inner SP Rate of Change: {slope_in}')
    print(f'Outer SP Rate of Change: {slope_out}')

    # we need to produce .dat files and do bin_fluxes for every timestep
    # which is the equivalent of the index in master_list
    # here, we want t = 0, 10, 100 ...164, 174
    time = [0,ramp_up]
    time.extend(list(range(ramp_up+constant_in_time,ramp_up+steady_state+1)))
    time.append(ramp_up+steady_state+ramp_down)

    data_wall1 = read_wall_soledge('/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/sol_simus/wall.shot106000.run1.dat')
    data_wall, removed_idx = remove_structure_points_soledge(data_wall1)

    # SOLEDGE binning for time independent case for main chamber values and dome values
    indices, div_indices, ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, binned_ion_flux, binned_atom_flux, binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat, wall_E_ion, wall_E_atom, wall_alpha_ion, wall_alpha_atom, wall_heat, heat_ion_total = bin_SOLEDGE_fluxes(data_wall, data_div, bins, div_bins)

    header = "Bin_Index,Flux_Ion,Flux_Atom,E_ion,E_atom,alpha_ion,alpha_atom,heat_total,heat_ion"
    data_to_save = np.array([
        [bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat, heat_ion]
        for bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat, heat_ion in zip(list(range(len(bins)+len(div_bins)+1)),ion_flux_total, atom_flux_total, E_ion_total, E_atom_total, alpha_ion_total, alpha_atom_total, heat_total, heat_ion_total)
    ])
    np.savetxt("RISP_Wall_data", data_to_save, delimiter=',', header=header, comments='', fmt=['%d']  + ['%.18e'] * (data_to_save.shape[1] - 1)) #, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])

    inner_bin_ids = list(range(45, 64))
    outer_bin_ids = list(range(18, 32)) 

    # now, we do SOLPS binning
    for t in time: # t is equivalent to index in master_list!

        # producing the .dat file that create_scenario takes as an input
        # the 'frozen in time' values which we are relating to a time axis

        # INNER 
        binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat = bin_SOLPS_fluxes(inner_bins, master_list_inner[t],inner_xc_bins, t, inner=True)

        header = "Bin_Index,Flux_Ion,Flux_Atom,E_ion,E_atom,alpha_ion,alpha_atom,heat_total"
        data_to_save = np.array([
            [bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat]
            for bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat in zip(inner_bin_ids,binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat)
        ])
        np.savetxt("RISP_data/time"+str(t)+".dat", data_to_save, delimiter=',', header=header, comments='', fmt=['%d']  + ['%.18e'] * (data_to_save.shape[1] - 1)) #, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])

        # OUTER
        binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat = bin_SOLPS_fluxes(outer_bins, master_list_outer[t],outer_xc_bins, t, inner=False)

        header = "Bin_Index,Flux_Ion,Flux_Atom,E_ion,E_atom,alpha_ion,alpha_atom,heat_total"
        data_to_save = np.array([
            [bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat]
            for bin_index,ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat in zip(outer_bin_ids,binned_div_ion, binned_div_atom, div_E_ion, div_E_atom, div_alpha_ion, div_alpha_atom, div_heat)
        ])
        np.savetxt("ROSP_data/time"+str(t)+".dat", data_to_save, delimiter=',', header=header, comments='', fmt=['%d']  + ['%.18e'] * (data_to_save.shape[1] - 1)) #, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])



