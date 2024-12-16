# plots fluxes for each of the 8 background cases 
# outputs a .dat file with average ion and atom fluxes corresponding to each wall index 

# each plot is wall index as the x-axis and flux as the y-axis, two sets of plots -- one for 
# ion fluxes and one for neutral fluxes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

# to import all data at once from each case, want to set up an empty list to put the data in
cases = []

# function to read the data file 
def read_flux_file(filename):
    """
    reads .dat file and extracts data for plotting

    parameters: 
        path to .dat file in a str

    returns: 
        numpy array of the data

    """
    try:

        data = []

        # open and read data file, skipping first 26 lines (in our files, these lines hold information in strings)

        with open(filename, 'r') as file:
            for _ in range(26): 
                next(file) # skipping first 26 lines

        # only want to read lines 26-349 (inclusive), because data after that is not our concern for right now
        # until 303 after removing divertor infrastructure points
            for line_num in range(26, 303):
                line = file.readline().strip()

                # data is split by whitespace, so need to assign as deliminater to differentiate values
                # we only want the first 11 columns from our file (up until energy distribution)
                if line:
                    line_data = [float(val) for val in line.split()[:11]] # each value is a string in file, so must convert to float before appending
                    data.append(line_data)

        # convert newly extracted data into a numpy array 
        data = np.array(data)

        return data

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def plot_data(wall_index, r1, z1, ion_fluxes, neutral_fluxes, case_labels):
    """
    plots data using matplotlib

    parameters: 
        wall index data, ion and neutral flux data as lists, labels of each case  
    
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
  
    # first plot of ion fluxes 
    for flux, label in zip(ion_fluxes, case_labels):
        ax1.plot(wall_index, flux, label=label)

    # calculating average ion flux
    avg_ion_flux = np.mean(ion_fluxes, axis=0)
    ax1.plot(wall_index, avg_ion_flux, label='Average', color='black', linestyle='--', linewidth=2)
    ax1.set_yscale('log')

    ax1.set_xlabel("Wall Index")
    ax1.set_ylabel("Ion Fluxes")
    ax1.set_title("Ion Fluxes")
    ax1.legend()
    ax1.grid(True)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.get_offset_text().set_fontsize(10)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # second plot of neutral fluxes
    for flux, label in zip(neutral_fluxes, case_labels):
        ax2.plot(wall_index, flux, label=label)

    # calculating average neutral flux
    avg_neutral_flux = np.mean(neutral_fluxes, axis=0)
    ax2.plot(wall_index, avg_neutral_flux, label='Average', color='black', linestyle='--', linewidth=2)
    ax2.set_yscale('log')

    ax2.set_xlabel("Wall Index")
    ax2.set_ylabel("Neutral Fluxes")
    ax2.set_title("Neutral Fluxes")
    ax2.legend()
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.get_offset_text().set_fontsize(10)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig("plots/Case_Fluxes.png")

    return avg_ion_flux, avg_neutral_flux

def average_values(temp_ion, temp_atom, temp_e, ion_fluxes, neutral_fluxes, avg_ion_flux, avg_neutral_flux):
    """
    calculates averages of values needed for mhims simulation: 
    E_ion
    E_atom
    heat_ion
    heat_atom
    """

    # finding average energies first 
    # must first multiply each energy by flux, and then divide those values by 
    # average flux value at the poin to get average energy 
    energy_ion = []
    q = 1

    for case in range(len(ion_fluxes)):
        intermediate_eng = 3*temp_e[case]*q + 2*temp_ion[case]
        energy_ion.append(intermediate_eng*ion_fluxes[case]/avg_ion_flux)
    energy_ion = np.mean(energy_ion, axis=0)

    # replacing nan values with 0
    energy_ion[np.isnan(energy_ion)] = 0

    # same for neutrals 
    energy_atom = []
    for case in range(len(neutral_fluxes)):
        intermediate = temp_atom[case]*neutral_fluxes[case]
        energy_atom.append(intermediate/avg_neutral_flux)
    energy_atom = np.mean(energy_atom, axis=0)

    # replacing nan values with 0
    energy_atom[np.isnan(energy_atom)] = 0

    # now we want to find ion heat and atom heat
    temp_ion = np.mean(temp_ion)
    temp_atom = np.mean(temp_atom)
    
    coulomb_cst = 1.6022e-19 
    heat_ion = temp_ion*avg_ion_flux*coulomb_cst # units conversion, heat in [W/m^2]

    radiation_heat = 0.35e6  # W/m2
    heat_atom = temp_atom*avg_neutral_flux*1.6022e-19 +radiation_heat # adding radiation


    return energy_ion, energy_atom, heat_ion, heat_atom



if __name__ == "__main__":
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

        # reading ion flux for each case and storing in master list
        ion_fluxes = [case[:, 6] for case in cases]

        # same as ion fluxes but with neutrals here 
        neutral_fluxes = [case[:, 9] for case in cases]

        # more data for extracting 
        temp_ion = cases[0][:,8]
        temp_atom = cases[0][:,10]
        temp_e = cases[0][:,7]

        # labeling each case for legend on graphs 
        case_labels = [f"Case {i+1}" for i in range(len(cases))]

        # plotting
        avg_ion_flux, avg_neutral_flux = plot_data(wall_index, r1, z1, ion_fluxes, neutral_fluxes, case_labels)
        
        # other average values
        ion_energy, atom_energy, heat_ion, heat_atom = average_values(temp_ion, temp_atom, temp_e, ion_fluxes, neutral_fluxes, avg_ion_flux, avg_neutral_flux)
        alpha_ion = np.full(len(ion_energy),6.0e+01)
        alpha_atom = np.full(len(atom_energy),4.5e+01)

        # saving average data to one .dat files
        header = "Index                              \t\tr1                             \tr2                       \tz1                       \tz2                \tFlux_Ion               \tFlux_Atom                     \tE_ion                       \tE_atom                    \talpha_ion              \talpha_atom                 \theat_ion                \theat_atom"
        data = np.column_stack((wall_index, r1 , r2, z1, z2, avg_ion_flux, avg_neutral_flux, ion_energy, atom_energy, alpha_ion, alpha_atom, heat_ion, heat_atom))
        np.savetxt("Background_Flux_Data", data, delimiter='\t', header=header, comments='')

    else:
        print("Error: Failed to read one or more data files.")
    