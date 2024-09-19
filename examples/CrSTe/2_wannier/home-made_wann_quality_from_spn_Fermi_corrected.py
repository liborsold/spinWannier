"""Interpolate the spin texture to DFT bands k-point grid by home-made wannierization code.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import shutil
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from spin_texture_interpolated import real_to_W_gauge, W_gauge_to_H_gauge, save_bands_and_spin_texture, \
                                        plot_bands_spin_texture
from wannier_utils import split_spn_dict, get_kpoint_path, get_2D_kpoint_mesh, load_lattice_vectors, reciprocal_lattice_vectors, unite_spn_dict,\
        hr_wann90_to_dict

# import sys
# sys.path.append('/home/lv268562/bin/python_scripts/wannier_quality/wann_quality_calculation')
# from wannier_quality import parse_eigenval_file, plot_err_vs_energy


# ================================================== INPUTS ==================================================================

nsc_calculation_path  =   '../0_nsc_for_wann_25x25_frozmaxmargin_0.2eV' #      = '../0_nsc_for_wann_25x25_frozmaxmargin_0.2eV'
band_for_Fermi_correction   = 11      # 1-based indexing
kpoint_for_Fermi_correction = '0.0000000E+00  0.0000000E+00  0.0000000E+00'

seedname = 'wannier90'

kpoint_matrix = [
    [(0.3333333333,  0.3333333333,  0.000000),    (0.00,  0.00,  0.00)],
    [(0.00,  0.00,  0.00),    (0.50, 0.00,  0.00)],
    [(0.50, 0.00,  0.00),    (0.3333333333,  0.3333333333,  0.000000)]
]
NK = 51
num_wann = 22
labels = ["K", "G", "M", "K"]

tb_model_folder = 'tb_model_wann90/'
dft_bands_folder = '../bands'
sc_folder = '../sc'

S_DFT_fname = "Sxyz_exp_values_from_spn_file.dat"
hr_R_name_sym = tb_model_folder + "hr_R_dict.pickle" #"hr_R_dict_sym.pickle"
spn_R_name_sym = tb_model_folder + "spn_R_dict.pickle"

discard_first_bands = 10

deltaE_around_EF = 0.5 #eV; integrate error in this +- window around E_F for plotting
deltaE2_around_EF = 0.1 #eV; integrate error in this +- window around E_F for plotting

# =============================================================================================================================


def get_fermi(path="."):
    with open(path + "/DOSCAR", 'r') as fr:
        for i, line in enumerate(fr):
            if i == 5:
                EF = line.split()[-2]
                with open(path + '/FERMI_ENERGY.in', 'w') as fw:
                    fw.write(f"# FERMI ENERGY extracted with the get_EF.py script\n{EF}")    
                return float(EF)


def get_band_at_kpoint_from_EIGENVAL(EIGENVAL_path='./EIGENVAL', target_band=1, target_kpoint_string='0.0000000E+00  0.0000000E+00  0.0000000E+00'):
    """
    Get the energy of the target_band at the target_kpoint from the EIGENVAL file.
    
    Parameters
    ----------
    EIGENVAL_path : str
        Path to the EIGENVAL file.
    target_band : int
        The target band (1-based indexing).
    target_kpoint : string from EIGENVAL file notation
    """
    with open(EIGENVAL_path, 'r') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            if target_kpoint_string in line:
                energy = float(lines[i+target_band].split()[1])
                return energy


def get_fermi_for_nsc_calculation_from_sc_calc_corrected_by_matching_bands(path=".", \
                                nsc_calculation_path='../0_nsc_for_wann_25x25_frozmaxmargin_0.2eV', \
                                corrected_at_kpoint='0.0000000E+00  0.0000000E+00  0.0000000E+00', \
                                    corrected_at_band=11, sc_calculation_path="../sc", \
                                        fout_name="FERMI_ENERGY_corrected.in"):

    Fermi_sc = get_fermi(sc_calculation_path)
    
    sc_band = get_band_at_kpoint_from_EIGENVAL(EIGENVAL_path=sc_calculation_path + "/EIGENVAL", target_band=corrected_at_band, target_kpoint_string=corrected_at_kpoint)
    nsc_band = get_band_at_kpoint_from_EIGENVAL(EIGENVAL_path=nsc_calculation_path + "/EIGENVAL", target_band=corrected_at_band, target_kpoint_string=corrected_at_kpoint)

    # WE REQUIRE THAT:
    #   band_nsc - Fermi_nsc = band_sc - Fermi_sc
    Fermi_nsc = Fermi_sc + nsc_band - sc_band

    with open(nsc_calculation_path + '/' + fout_name, 'w') as fw:
        fw.write(f"# {Fermi_nsc:.8f} eV = {Fermi_sc:.8f} + ({nsc_band:.8f} - {sc_band:.8f}) = Fermi_sc + (E_band{corrected_at_band}_nsc - E_band{corrected_at_band}_sc) ... self-consistent Fermi from {sc_calculation_path} corrected so that band {corrected_at_band} at k-point {corrected_at_kpoint} is at the same energy (relative to the recpective Fermi energies) for the self-consistent (path {sc_calculation_path}) and non-self-consistent (path {nsc_calculation_path}) calculation\n{Fermi_nsc:.8f}")    
    
    return float(Fermi_nsc)


def compare_eigs_bandstructure_at_exact_kpts(dft_bands, wann_bands, num_kpoints, f_name_out='WannierBerri_quality_error_Fermi_corrected.dat'):
    error_by_energy = np.zeros((num_kpoints*num_wann, 2))
    for i in range(num_kpoints):
        # save the DFT energy as x variable
        error_by_energy[i*num_wann:(i+1)*num_wann, 0] = dft_bands[i,:]

        # save the Wannierization error as y variable
        error_by_energy[i*num_wann:(i+1)*num_wann, 1] = (dft_bands[i,:] - wann_bands[i,:])**2

    # sort 'error_by_energy' by energy - by the first column
    error_by_energy = error_by_energy[ error_by_energy[:,0].argsort() ]

    # save to file
    header = "E (eV)\t(E_wann - E_DFT)**2"
    fmt = "%.12f\t%.12e"
    np.savetxt(f_name_out, error_by_energy, delimiter="\t", header=header, fmt=fmt)
    return error_by_energy


def duplicate_kpoints_for_home_made(data, NK):
    """duplicate also the last k-point (in  dictionary the keys are unique, so actually the data in the dictionaries where keys are k-points contain only one of each k-point, so if k-path starts and ends with the same k-point, only the first one is recorded"""
    dimensions = list(np.array(data).shape)
    print(dimensions[0])
    print(NK)
    n_blocs = (dimensions[0])//(NK-1)
    dimensions[0] += n_blocs

    data_duplicates = np.zeros(dimensions)

    data_duplicates[:NK, :] = data[:NK, :]
    for i in range(1, n_blocs-1):
        data_duplicates[i*NK, :] = data[i*(NK-1), :]
        data_duplicates[i*NK+1:i*NK+1+NK, :] = data[1+i*(NK-1):1+i*(NK-1)+NK, :]
    # last block different - missing last value (supposed to be the same as the first value) 
    i = n_blocs-1
    data_duplicates[i*NK, :] = data[i*(NK-1), :]
    data_duplicates[i*NK+1:i*NK+1+NK-2, :] = data[1+i*(NK-1):1+i*(NK-1)+NK, :]
    data_duplicates[-1, :] = data[0, :]
    return data_duplicates


def get_NKpoints(OUTCAR='OUTCAR'):
    """Return number of kpoints from bands calculation from OUTCAR.

    Args:
        OUTCAR (str, optional): OUTCAR path. Defaults to 'OUTCAR'.

    Returns:
        int: number of k-points stated in the OUTCAR
    """
    with open(OUTCAR, 'r') as fr:
        for line in fr:
            if "NKPTS =" in line:
                return int(line.split()[3])


def parse_eigenval_file(fin, spin=0):
    flag_kpoint = False
    flag_bands = False
    i_k = -1
    i_b = -1
    num_bands = 99999999
    with open(fin, 'r') as fr:
        for i, line in enumerate(fr):

            l = line.split()

            if i == 5:
                num_bands = int(l[2])
                num_kpoints = int(l[1])
                kpoints = np.zeros((num_kpoints, 3))
                bands = np.zeros((num_kpoints, num_bands))

            if flag_bands == True:
                i_b += 1
                bands[i_k, i_b] = l[spin+1]

            if flag_kpoint == True:
                i_k += 1
                kpoints[i_k, :] = l[0:3]
                flag_kpoint = False
                flag_bands = True

            # empty line
            if l == []:
                flag_kpoint = True
            else:
                if l[0] == str(num_bands):
                    flag_bands = False
                    i_b = -1

    kpoints = kpoints.astype(np.float32)
    bands = bands.astype(np.float32)

    kpoints_path = np.zeros((num_kpoints,))
    for i in range(num_kpoints-1):
        kpoints_path[i+1] =  kpoints_path[i] + np.sum( (kpoints[i+1,:] - kpoints[i,:])**2 ) ** 0.5

    return kpoints_path, bands, num_kpoints, num_bands


def plot_err_vs_energy(error_by_energy, Ef, title="Wannierization RMS error vs. energy", fig_name_out="wannier_quality_error_by_energy.png"):
    plt.semilogy(error_by_energy[:,0]-Ef, error_by_energy[:,1]**.5, 'ok', linewidth=1, markersize=1)
    plt.title(title, fontsize=8)
    plt.xlabel(r"$E - E_\mathrm{F}$ (eV)")
    plt.ylabel(r"$\|E_\mathrm{wann} - E_\mathrm{DFT}\|$ (eV)")
    #plt.show()
    plt.savefig(fig_name_out, dpi=400)
    plt.close()


# ======================== GET HOME-MADE SPIN TEXTURE ================================ 
A = load_lattice_vectors(win_file="wannier90.win")
G = reciprocal_lattice_vectors(A)
kpoints, kpoints_cart, kpath = get_kpoint_path(kpoint_matrix, G, Nk=NK)

# interpolate Hamiltonian
with open(hr_R_name_sym, 'rb') as fr:
    hr_R_dict = pickle.load(fr)

H_k_W = real_to_W_gauge(kpoints, hr_R_dict)
Eigs_k, U_mn_k = W_gauge_to_H_gauge(H_k_W, U_mn_k={}, hamiltonian=True)

# get Fermi for the wannier non-self-consistent calculation
Fermi_nsc_wann = get_fermi_for_nsc_calculation_from_sc_calc_corrected_by_matching_bands(path=".", \
                                nsc_calculation_path=nsc_calculation_path, \
                                corrected_at_kpoint=kpoint_for_Fermi_correction, \
                                    corrected_at_band=band_for_Fermi_correction, sc_calculation_path=sc_folder, \
                                        fout_name="FERMI_ENERGY_corrected.in")

shutil.copyfile(nsc_calculation_path + "/FERMI_ENERGY_corrected.in", './FERMI_ENERGY_corrected.in')
E_F = Fermi_nsc_wann
# take all elements from Eigs_k and subtract Fermi_nsc_wann
for key in Eigs_k.keys():
    Eigs_k[key] -= Fermi_nsc_wann

# interpolate spin operator
with open(spn_R_name_sym, 'rb') as fr:
    spn_R_dict = pickle.load(fr)

S_mn_R_x, S_mn_R_y, S_mn_R_z = split_spn_dict(spn_R_dict, spin_names=['x','y','z'])

Sx_k_W = real_to_W_gauge(kpoints, S_mn_R_x)
Sy_k_W = real_to_W_gauge(kpoints, S_mn_R_y)
Sz_k_W = real_to_W_gauge(kpoints, S_mn_R_z)

print('len kpoints', len(kpoints))
print('len H_k_W', len(list(H_k_W.keys())))

S_mn_k_H_x = W_gauge_to_H_gauge(Sx_k_W, U_mn_k=U_mn_k, hamiltonian=False)
S_mn_k_H_y = W_gauge_to_H_gauge(Sy_k_W, U_mn_k=U_mn_k, hamiltonian=False)
S_mn_k_H_z = W_gauge_to_H_gauge(Sz_k_W, U_mn_k=U_mn_k, hamiltonian=False)

# =====================================================================================


# COMPARE with DFT
dft_kpoints, dft_bands, num_kpoints_dft, num_bands = parse_eigenval_file(dft_bands_folder + "/EIGENVAL")
dft_bands = dft_bands[:,discard_first_bands:discard_first_bands+num_wann]

# get Fermi for the bands non-self-consistent calculation
Fermi_nsc_bands = get_fermi_for_nsc_calculation_from_sc_calc_corrected_by_matching_bands(path=".", \
                                nsc_calculation_path=dft_bands_folder, \
                                corrected_at_kpoint=kpoint_for_Fermi_correction, \
                                    corrected_at_band=band_for_Fermi_correction, sc_calculation_path=sc_folder, \
                                        fout_name="FERMI_ENERGY_corrected.in")
dft_bands -= Fermi_nsc_bands

# make 2D again: (NKpoints, num_wann)
E_to_compare = np.array( [Eigs_k[key] for key in Eigs_k.keys()])
# print("E_to_compare_shape", E_to_compare.shape)
E_to_compare_with_duplicates = duplicate_kpoints_for_home_made(E_to_compare, NK)
# print("E_to_compare_with_duplicates_shape", E_to_compare_with_duplicates.shape)

print("50", E_to_compare_with_duplicates[50,:])
print("51", E_to_compare_with_duplicates[51,:])

S_mn_k_H_x_to_compare = duplicate_kpoints_for_home_made(np.array( [np.diag(S_mn_k_H_x[key]) for key in S_mn_k_H_x.keys()]), NK)
S_mn_k_H_y_to_compare = duplicate_kpoints_for_home_made(np.array( [np.diag(S_mn_k_H_y[key]) for key in S_mn_k_H_y.keys()]), NK)
S_mn_k_H_z_to_compare = duplicate_kpoints_for_home_made(np.array( [np.diag(S_mn_k_H_z[key]) for key in S_mn_k_H_z.keys()]), NK)

# print('k-points from dict keys', list(S_mn_k_H_x.keys()))

S_to_compare_with_duplicates = np.array([S_mn_k_H_x_to_compare, S_mn_k_H_y_to_compare, S_mn_k_H_z_to_compare])
S_shape = S_to_compare_with_duplicates.shape

# make the spin axis the last instead of the first one
S_to_compare_with_duplicates = S_to_compare_with_duplicates.swapaxes(0,1)
S_to_compare_with_duplicates = S_to_compare_with_duplicates.swapaxes(1,2)

# S_to_compare_with_duplicates = duplicate_kpoints(S_to_compare, NK)


error_by_energy = compare_eigs_bandstructure_at_exact_kpts(dft_bands, E_to_compare_with_duplicates, num_kpoints_dft, f_name_out='home-made_quality_error_Fermi_corrected.dat')

plot_err_vs_energy(error_by_energy, Ef=0, title="Wannierization RMS error vs. energy", fig_name_out="wannier_quality_error_by_energy_home-made_Fermi_corrected.png")


# ------------- COMPARE spin texture --------------------

# load spin expectation values and select relevant bands
S_DFT = np.loadtxt(f"{dft_bands_folder}/{S_DFT_fname}")
# select relevant bands
NK = get_NKpoints(OUTCAR=f'{dft_bands_folder}/OUTCAR')
S_DFT = S_DFT.reshape(NK, -1, 3)
S_DFT_to_compare = S_DFT[:,discard_first_bands:discard_first_bands+num_wann,:]

# print("S_to_compare dimensions", S_to_compare_with_duplicates.shape)
# print("S_to_compare", S_to_compare)

print("50", S_to_compare_with_duplicates[50,:,:])
print("51", S_to_compare_with_duplicates[51,:,:])

S_diff = np.abs( S_DFT_to_compare.reshape(-1,3) - S_to_compare_with_duplicates.reshape(-1,3) )
E_diff = np.abs( dft_bands.reshape(-1) - E_to_compare_with_duplicates.reshape(-1) )

# E_F was already subtracted
E = dft_bands.reshape(-1)

# make final matrix
error_E_S_by_energy = np.vstack([E, E_diff, S_diff[:,0], S_diff[:,1], S_diff[:,2]]).T

print("Error matrix shape:", error_E_S_by_energy.shape)

# order the '' matrix by energy (0th column)
error_E_S_by_energy = error_E_S_by_energy[error_E_S_by_energy[:, 0].argsort()]

np.savetxt("home-made_quality_error_S_E_Fermi_corrected.dat", error_E_S_by_energy, header="E (eV)\t|Delta E| (eV)\t|Delta S_x|\t|Delta S_y|\t|Delta S_z|")

# plot all the errors
fig, axes = plt.subplots(2, 2, figsize=[6,6])

axes[0,0].semilogy(error_E_S_by_energy[:,0], error_E_S_by_energy[:,1], 'ko', markersize=2)
axes[0,0].set_title(r'$E$', fontsize=14)
axes[0,0].set_ylabel(r"|$E_\mathrm{DFT} - E_\mathrm{wann}$| (eV)")

axes[0,1].semilogy(error_E_S_by_energy[:,0], error_E_S_by_energy[:,4], 'bo', markersize=2)
axes[0,1].set_title(r'$S_z$', fontsize=14)
axes[0,1].set_ylabel(r"|$S_{z, \mathrm{DFT}} - S_{z, \mathrm{wann}}$|")

axes[1,0].semilogy(error_E_S_by_energy[:,0], error_E_S_by_energy[:,2], 'ro', markersize=2)
axes[1,0].set_title(r'$S_x$', fontsize=14)
axes[1,0].set_ylabel(r"|$S_{x, \mathrm{DFT}} - S_{x, \mathrm{wann}}$|")
axes[1,0].set_xlabel(r"$E - E_\mathrm{F}$ (eV)")

axes[1,1].semilogy(error_E_S_by_energy[:,0], error_E_S_by_energy[:,3], 'go', markersize=2)
axes[1,1].set_title(r'$S_y$', fontsize=14)
axes[1,1].set_ylabel(r"|$S_{y, \mathrm{DFT}} - S_{y, \mathrm{wann}}$|")
axes[1,1].set_xlabel(r"$E - E_\mathrm{F}$ (eV)")

plt.tight_layout()

plt.savefig("ERRORS_all_home-made_Fermi_corrected.png", dpi=400)
plt.close()


# integrate error and write it out

def integrate_error(error_by_energy, E_min=-1e3, E_max=1e3):
    """Integrate the error in 'f_name_in' in the energy range [E_min, E_max] included."""
    E = error_by_energy[:,0]
    err_array = error_by_energy[ (E_min <= E) & (E <= E_max) ][:,1:]
    err_array_squared = np.power(err_array, 2)
    return np.power(err_array_squared.mean(axis=0), 0.5)


def get_frozen_window_min_max(wannier90winfile='wannier90.win'):
    dis_froz_min = None
    dis_froz_max = None

    with open(wannier90winfile, 'r') as fr:
        for line in fr:
            if "dis_froz_min=" in line:
                dis_froz_min = float(line.split('dis_froz_min=')[1].split()[0])
            if "dis_froz_max=" in line:
                dis_froz_max = float(line.split('dis_froz_max=')[1].split()[0])
    return dis_froz_min, dis_froz_max


dis_froz_min, dis_froz_max = get_frozen_window_min_max()

Fermi_sc = float(np.loadtxt(f"{sc_folder}/FERMI_ENERGY.in"))
mean_error_whole_range = integrate_error(error_E_S_by_energy, E_min=-1e6, E_max=1e6)
mean_error_up_to_Fermi = integrate_error(error_E_S_by_energy, E_min=-1e6, E_max=0)
mean_error_frozen_window = integrate_error(error_E_S_by_energy, E_min=dis_froz_min-Fermi_nsc_wann, E_max=dis_froz_max-Fermi_nsc_wann)
mean_error_around_Fermi = integrate_error(error_E_S_by_energy, E_min=-deltaE_around_EF, E_max=deltaE_around_EF)
mean_error_around_Fermi2 = integrate_error(error_E_S_by_energy, E_min=-deltaE2_around_EF, E_max=deltaE_around_EF)

with open("error_home-made_integrated_Fermi_corrected.dat", 'w') as fw:
    fw.write("#                                           \tRMSE_E (eV)\tRMSE_Sx  \tRMSE_Sy  \tRMSE_Sz\n")
    fw.write(f"in the whole energy range                \t" + '\t'.join([f"{val:.6f}" for val in mean_error_whole_range]) + '\n')
    fw.write(f"up to Fermi                                 \t" + '\t'.join([f"{val:.6f}" for val in mean_error_up_to_Fermi]) + '\n')
    fw.write(f"in the frozen window ({dis_froz_min-Fermi_nsc_wann:.3f} to {dis_froz_max-Fermi_sc:.3f} eV)\t" + '\t'.join([f"{val:.6f}" for val in mean_error_frozen_window]) + '\n')
    fw.write(f"window +- {deltaE_around_EF:.3f} eV around Fermi level    \t" + '\t'.join([f"{val:.6f}" for val in mean_error_around_Fermi]) + '\n')
    fw.write(f"window +- {deltaE2_around_EF:.3f} eV around Fermi level    \t" + '\t'.join([f"{val:.6f}" for val in mean_error_around_Fermi2]))

# ------------ ABSOLUTE VALUE OF SPIN ------------------

S_abs = np.linalg.norm(S_to_compare_with_duplicates.reshape(-1,3), axis=1)
print('S_abs shape', S_abs.shape)

# combined matrix
E_S_abs = np.vstack([E, S_abs]).T
print('S_abs_E shape', E_S_abs.shape)

# order the '' matrix by energy (0th column)
E_S_abs = E_S_abs[E_S_abs[:, 0].argsort()]

# save txt file, with statistics
Sabs_mean       = np.mean(S_abs)
Sabs_median     = np.median(S_abs)
Sabs_over_one   = len(S_abs[S_abs > 1.0]) / len(S_abs)
header = f"S over 1 ratio = {Sabs_over_one:.8f}\nS mean = {Sabs_mean:.8f}\nS median = {Sabs_median:.8f}\nE (eV)\t|S|"
np.savetxt("home-made_Sabs_vs_E_Fermi_corrected.dat", E_S_abs, header=header)

# plot and save S_abs vs. E
plot_title = "S_abs"

fig, ax = plt.subplots(1, 1, figsize=[4,4])
ax.plot(E_S_abs[:,0], E_S_abs[:,1], 'ko', markersize=2)
ax.set_title(r'|$S$|', fontsize=14)
ax.set_ylabel(r"|$S$|")
ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)")
plt.tight_layout()
plt.savefig(plot_title+"_vs_E_home-made_Fermi_corrected.png", dpi=400)
plt.close()

# plot and save S_abs histogram
fig, ax = plt.subplots(1, 1, figsize=[4,4])
plt.hist(S_abs.flatten(), bins=100)
plt.xlabel('|S|')
plt.title(f"histogram of abs values of diagonal elements of spin operator\n- in k-space (eigenvalues) home-made interpolation", fontsize=8)
plt.tight_layout()
plt.savefig(plot_title+"_S_histogram_home-made_Fermi_corrected.png", dpi=400)
plt.close()
