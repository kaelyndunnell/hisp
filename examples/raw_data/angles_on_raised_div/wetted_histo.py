import matplotlib.pyplot as plt 
import numpy as np 
from numpy import genfromtxt
import csv 
import math
from scipy.signal import find_peaks, argrelextrema, find_peaks_cwt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter


# plots histograms for heats in each FW row to determine wetted and shadowed regions
# produces .csv file with wetted area ratios and corresponding fractions

def load_geometry(filename):
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
    z_coord = geometry[:,0]*1e-3 # convert mm to m

    # r-coordinates are the second column in data, convert to list 
    r_coord = geometry[:,1]*1e-3 # convert mm to m

    return z_coord, r_coord

z_coord, r_coord = load_geometry("/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/FWpanelcorners.txt")

# opening wetted vs shadowed area data files
row_files = [
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_1_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_2_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_3_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_5_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_5_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_6_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_9_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_8_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_9_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_10_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_11_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_12_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_13_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_15_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_15_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_16_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_19_flat_top.csv',
    '/home/ITER/dunnelk/kaelyn-MHIMS/flux_data/wetted_rows_data/row_18_flat_top.csv'
]

# loading them into numpy arrays. index in rows +1 = first wall row
rows = [np.genfromtxt(row, delimiter=',', skip_header=1) for row in row_files]

# creating a list that holds all the sub-bins for data saving-purposes
sub_bins = [[] for _ in range(len(rows))]

# histogram plotting 
for idx in range(len(rows)):
    plt.figure()
    q = []
    q_tot = []
    for val in range(len(rows[idx])):
        q_tot.append(rows[idx][val][2])
        if rows[idx][val][2] != 0:
            q.append(rows[idx][val][2])
    
    # total surface area of each row
    cell_area = 9.95e-06 # m^2
    # total_cells = len(q_tot)

    bin_surf_area = 2*math.pi*0.5*abs(r_coord[idx+1]+r_coord[idx])*((r_coord[idx+1]-r_coord[idx])**2+(z_coord[idx+1]-z_coord[idx])**2)**(1/2) # area of one panel, # m^2

    # check with bin coordinates
    # print(total_surf_area)
    # print(bin_surf_area)

    q = np.array(q)
    q_tot = np.array(q_tot)
    hist, bin_edges = np.histogram(q, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # find peaks in histo
    peaks, _ = find_peaks(hist, distance=1)

    if len(bin_centers[peaks])>1 and idx != 19: # 3 bin-bins
        # find lowest points between peaks
        if idx == 1:
            lowest_bin_center = 2.9e+05
        else:
            lowest_bin_center = bin_centers[np.argmin(hist[peaks[0]:peaks[1]]) + peaks[0]] 
        plt.vlines(lowest_bin_center, 0, max(hist), color='red', label='Limiter Point')

        # find avgs of each sub-bin
        lowbin = []
        highbin = []
        for heat in q: # sort into low and high heat load bins
            if heat < lowest_bin_center:
                lowbin.append(heat)
            else:
                highbin.append(heat)
        lowheat = np.mean(lowbin)
        highheat = np.mean(highbin)
        plt.vlines(lowheat, 0, max(hist), color='blue', label="Low Bin Average")
        plt.vlines(highheat, 0, max(hist), color='purple', label="High Bin Average")

        if idx+1==10:
            bin_surf_area = bin_surf_area*2/3
        
        Slow = 9*cell_area*len(lowbin)
        Shigh = 9*cell_area*len(highbin)
        Sshadow = bin_surf_area - Slow - Shigh
        Stot = Slow + Shigh + Sshadow

        # need to return frac, ST, Slow, and Shigh
        frac = (lowheat*Slow)/(lowheat*Slow+highheat*Shigh)
        
        sub_bins[idx].append(Slow)
        sub_bins[idx].append(Stot)
        sub_bins[idx].append(Shigh)
        sub_bins[idx].append(frac)

        if idx+1 == 10: # fourth bin here, values all for one panel 
            dfw_area = 6.55 # m^2
            sub_bins[idx].append(dfw_area)


    else: # 2 bin-bins, same process as above
        if idx == 13 or idx == 15: # separating out port rows 
            bin_surf_area = bin_surf_area*0.5
            dfw_area = 23.6/2 #m^2
            if idx == 15:
                adjusted_q = []
                for heat in q:
                    if heat<5.0e+05:
                        adjusted_q.append(heat)
                avgheat = np.mean(adjusted_q)
                Slow = 9*cell_area*len(adjusted_q)
                Sshadow = bin_surf_area - Slow
                Stot = Sshadow + Slow
            else:
                Slow = 9*cell_area*len(q)
                Sshadow = bin_surf_area - Slow
                Stot = Sshadow + Slow
                avgheat = np.mean(q)
            sub_bins[idx].append(Slow)
            sub_bins[idx].append(Stot)
            sub_bins[idx].append(dfw_area)
        elif idx == 19:
            adjusted_q = []
            for heat in q:
                if heat<1.5e+06:
                    adjusted_q.append(heat)
            avgheat = np.mean(q)
            Slow = 9*cell_area*len(adjusted_q)
            Sshadow = bin_surf_area - Slow
            Stot = Sshadow + Slow
            sub_bins[idx].append(Slow)
            sub_bins[idx].append(Stot)
        else:
            Slow = 9*cell_area*len(q)
            Sshadow = bin_surf_area - Slow
            Stot = Sshadow + Slow
            avgheat = np.mean(q)
            sub_bins[idx].append(Slow)
            sub_bins[idx].append(Stot)
        
        plt.vlines(avgheat, 0, max(hist), color='blue', label="Bin Average")
        
    # plot histos
    plt.hist(q, bins=20, color='green', label="Binned Heat")
    plt.plot(bin_centers[peaks],hist[peaks],'x')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Heats (W/m^2)')
    plt.title('Heat Histogram Row '+str(idx+1))
    plt.savefig('Histo'+str(idx))

# writing to csv file for each access
filename = "Wetted_Frac_Bin_Data.csv"
header = ["Slow", "Stot", "Shigh", "f", "DFW"]

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerows(sub_bins)