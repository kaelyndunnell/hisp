import numpy as np
import matplotlib.pyplot as plt

# fixed values from Tom: fluxes, impact energies

ion_flux_tot = np.full(64,1.0e+18)
atom_flux_tot = np.full(64,0.7e+19)
E_ion_tot = np.full(64,50) # eV
E_atom_tot = np.full(64,200)
alpha_ion_tot = np.full(64,45.0)
alpha_atom_tot = np.full(64,60.0)
heat_tot = np.full(64,0.0)
bin_indexes = np.arange(0,65,1)

# assume no flux on straight part of divertor targets and plates
no_flux_div = np.array([5,6,7,8,9,10,11,12,13,14,15,16,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])+18

for idx in range(len(ion_flux_tot)):
    if idx+1 in no_flux_div:
        ion_flux_tot[idx] = 0.0
        atom_flux_tot[idx] = 0.0
        E_ion_tot[idx] = 0.0
        E_atom_tot[idx] = 0.0
        alpha_ion_tot[idx] = 0.0
        alpha_atom_tot[idx] = 0.0
 
header = "Bin_Index,Flux_Ion,Flux_Atom,E_ion,E_atom,alpha_ion,alpha_atom,heat_total"
data_to_save = np.array([
    [bin_index, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat]
    for bin_index, ion_flux, atom_flux, E_ion, E_atom, alpha_ion, alpha_atom, heat in zip(bin_indexes,ion_flux_tot, atom_flux_tot, E_ion_tot, E_atom_tot, alpha_ion_tot, alpha_atom_tot, heat_tot)
])
np.savetxt("ICWC_data.dat", data_to_save, delimiter=',', header=header, comments='', fmt=['%d']  + ['%.18e'] * (data_to_save.shape[1] - 1)) #, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])
