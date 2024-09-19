"""Interpolate the spin texture to DFT bands k-point grid by home-made wannierization code.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import shutil
from spinWannier.wannier_utils import real_to_W_gauge, W_gauge_to_H_gauge, \
                    split_spn_dict, get_kpoint_path, load_lattice_vectors, \
                        reciprocal_lattice_vectors, check_file_exists
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import sys
# sys.path.append('/home/lv268562/bin/python_scripts/wannier_quality/wann_quality_calculation')
# from wannier_quality import parse_eigenval_file, plot_err_vs_energy

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


def compare_eigs_bandstructure_at_exact_kpts(dft_bands, wann_bands, num_kpoints, num_wann, f_name_out='WannierBerri_quality_error_Fermi_corrected.dat'):
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
    # print(dimensions[0])
    # print(NK)
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


def plot_err_vs_bands(kpoints, kpath, kpath_ticks, Eigs_k, E_diff, S_diff, fout='ERRORS_ALL_band_structure.jpg', yaxis_lim=None):

    """Output a figure with RMSE_E, RMSE_Sx, RMSE_Sy, and RMSE_Sz-projected band structure."""
    NW = len(Eigs_k[list(Eigs_k.keys())[0]])
    Nk = len(kpoints)//(len(kpath_ticks)-1)

    fig, axes = plt.subplots(2, 2, figsize=[9, 6])
    fig.suptitle('Error of Wannier interpolation', fontsize=14)
    spin_name = [r'Energy (eV)', r'$S_z$', r'$S_x$', r'$S_y$']
    for i, S in enumerate([E_diff, S_diff[:,2], S_diff[:,0], S_diff[:,1]]):
        ax = axes[i//2, i%2]
        ax.axhline(linestyle='--', color='k')
        if yaxis_lim:
                ax.set_ylim(yaxis_lim)
        ax.set_xlim([min(kpath), max(kpath)])
        ax.set_title(spin_name[i], fontsize=13, pad=8)
         
        if i == 0:
            # colorbar limits for energy
            vmin = 0
            vmax = 0.002
        else:
            # colorbar limits for spin
            vmin = 0
            vmax = 0.05
        sc = ax.scatter([[k_dist for i in range(NW)] for k_dist in kpath], [Eigs_k[kpoint] for kpoint in kpoints],
                        c=S.reshape(-1,NW), cmap='YlOrRd', s=0.2, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sc, cax=cax, orientation='vertical')

        for j in range(1, len(kpath_ticks)-1):
            ax.axvline(x=kpath[j*Nk], color='#000000', linestyle='-', linewidth=0.75)

        # if i == 0 or i == 2:
        ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)', fontsize=13)
        # if i == 0 or i == 1:
        secax = ax.secondary_xaxis('top')
        secax.tick_params(labelsize=9) #axis='both', which='major', 
        secax.set_xlabel(r"$k$-distance (1/$\mathrm{\AA}$)", fontsize=10)
        # no primary x-axis
        ax.set_xticks([])

        # if i == 2 or i == 3:
        idx = np.array(range(0, Nk*len(kpath_ticks), Nk))
        idx[-1] += -1
        ax.set_xticks(kpath[idx])
        ax.set_xticklabels(kpath_ticks, fontsize=12)
        ax.yaxis.set_tick_params(labelsize=11)

        #cbar.set_label(r'$S_\mathrm{z}$')
        #sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    # plt.suptitle(fig_caption)
    plt.tight_layout()
    # plt.show()
    fout = check_file_exists(fout)
    plt.savefig(fout, dpi=400)
    plt.close()
    print('ERROR band structure printed\n\n--------------------\n\n=======================')
    # plt.show()


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
            if "dis_froz_min" in line:
                dis_froz_min = float(line.split('dis_froz_min')[1].split('=')[1].split(' ')[0])
            if "dis_froz_max" in line:
                dis_froz_max = float(line.split('dis_froz_max')[1].split('=')[1].split(' ')[0])
    return dis_froz_min, dis_froz_max


def wannier_quality_calculation(kpoint_matrix, NK, kpath_ticks, Fermi_nsc_wann, num_wann, discard_first_bands=0, sc_dir='0_self-consistent', nsc_dir='1_non-self-consistent', wann_dir='2_wannier', \
                    bands_dir='1_band_structure', tb_model_dir='2_wannier/tb_model_wann90', \
                        band_for_Fermi_correction=None, kpoint_for_Fermi_correction='0.0000000E+00  0.0000000E+00  0.0000000E+00', \
                            yaxis_lim=None):
    """Needed files from VASP:

        = nsc_calculation_path:
            - EIGENVAL
            - DOSCAR

        = dft_bands_folder:
            - EIGENVAL
            - "Sxyz_exp_values_from_spn_file.dat" (get automatically from wannier90.spn_formatted!!)
            - OUTCAR

    """

    # make sure directories don't have / at the end
    for dir_name in [sc_dir, nsc_dir, wann_dir, bands_dir, tb_model_dir]:
        if dir_name[-1] == '/':
            dir_name = dir_name[:-1]

    if band_for_Fermi_correction is None: # if not specified, take the first non-discarded band
        band_for_Fermi_correction = discard_first_bands + 1

    # ================================================== CONSTANTS ==================================================================
    
    labels = ["G", "K", "M", "G"]

    S_DFT_fname = "Sxyz_exp_values_from_spn_file.dat"
    hr_R_name = tb_model_dir + "/hr_R_dict.pickle" #"hr_R_dict_sym.pickle"
    spn_R_name = tb_model_dir + "/spn_R_dict.pickle"

    deltaE_around_EF = 0.5 #eV; integrate error in this +- window around E_F for plotting
    deltaE2_around_EF = 0.1 #eV; integrate error in this +- window around E_F for plotting

    # =============================================================================================================================

    # ======================== GET HOME-MADE SPIN TEXTURE ================================ 
    A = load_lattice_vectors(win_file=f"{wann_dir}/wannier90.win")
    G = reciprocal_lattice_vectors(A)
    kpoints, kpoints_cart, kpath = get_kpoint_path(kpoint_matrix, G, Nk=NK)

    # interpolate Hamiltonian
    with open(hr_R_name, 'rb') as fr:
        hr_R_dict = pickle.load(fr)

    H_k_W = real_to_W_gauge(kpoints, hr_R_dict)
    Eigs_k, U_mn_k = W_gauge_to_H_gauge(H_k_W, U_mn_k={}, hamiltonian=True)

    shutil.copyfile(nsc_dir + "/FERMI_ENERGY_corrected.in", './FERMI_ENERGY_corrected.in')
    E_F = Fermi_nsc_wann
    # take all elements from Eigs_k and subtract Fermi_nsc_wann
    for key in Eigs_k.keys():
        Eigs_k[key] -= Fermi_nsc_wann

    # interpolate spin operator
    with open(spn_R_name, 'rb') as fr:
        spn_R_dict = pickle.load(fr)

    S_mn_R_x, S_mn_R_y, S_mn_R_z = split_spn_dict(spn_R_dict, spin_names=['x','y','z'])

    Sx_k_W = real_to_W_gauge(kpoints, S_mn_R_x)
    Sy_k_W = real_to_W_gauge(kpoints, S_mn_R_y)
    Sz_k_W = real_to_W_gauge(kpoints, S_mn_R_z)

    # print('len kpoints', len(kpoints))
    # print('len H_k_W', len(list(H_k_W.keys())))

    S_mn_k_H_x = W_gauge_to_H_gauge(Sx_k_W, U_mn_k=U_mn_k, hamiltonian=False)
    S_mn_k_H_y = W_gauge_to_H_gauge(Sy_k_W, U_mn_k=U_mn_k, hamiltonian=False)
    S_mn_k_H_z = W_gauge_to_H_gauge(Sz_k_W, U_mn_k=U_mn_k, hamiltonian=False)

    # =====================================================================================

    # COMPARE with DFT
    dft_kpoints, dft_bands, num_kpoints_dft, num_bands = parse_eigenval_file(bands_dir + "/EIGENVAL")
    dft_bands = dft_bands[:,discard_first_bands:discard_first_bands+num_wann]

    # get Fermi for the bands non-self-consistent calculation
    Fermi_nsc_bands = get_fermi_for_nsc_calculation_from_sc_calc_corrected_by_matching_bands(path=".", \
                                    nsc_calculation_path=bands_dir, \
                                    corrected_at_kpoint=kpoint_for_Fermi_correction, \
                                        corrected_at_band=band_for_Fermi_correction, sc_calculation_path=sc_dir, \
                                            fout_name="FERMI_ENERGY_corrected.in")
    dft_bands -= Fermi_nsc_bands

    # make 2D again: (NKpoints, num_wann)
    E_to_compare = np.array( [Eigs_k[key] for key in Eigs_k.keys()])
    # print("E_to_compare_shape", E_to_compare.shape)
    E_to_compare_with_duplicates = duplicate_kpoints_for_home_made(E_to_compare, NK)
    # print("E_to_compare_with_duplicates_shape", E_to_compare_with_duplicates.shape)

    # print("50", E_to_compare_with_duplicates[50,:])
    # print("51", E_to_compare_with_duplicates[51,:])

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


    error_by_energy = compare_eigs_bandstructure_at_exact_kpts(dft_bands, E_to_compare_with_duplicates, num_kpoints_dft, num_wann, f_name_out='home-made_quality_error_Fermi_corrected.dat')

    plot_err_vs_energy(error_by_energy, Ef=0, title="Wannierization RMS error vs. energy", \
                       fig_name_out="wannier_quality_error_by_energy_home-made_Fermi_corrected.png")


    # ------------- COMPARE spin texture --------------------

    # load spin expectation values and select relevant bands
    S_DFT = np.loadtxt(f"{bands_dir}/{S_DFT_fname}")
    # select relevant bands
    NK = get_NKpoints(OUTCAR=f'{bands_dir}/OUTCAR')
    S_DFT = S_DFT.reshape(NK, -1, 3)
    S_DFT_to_compare = S_DFT[:,discard_first_bands:discard_first_bands+num_wann,:]

    # print("S_to_compare dimensions", S_to_compare_with_duplicates.shape)
    # print("S_to_compare", S_to_compare)

    # print("50", S_to_compare_with_duplicates[50,:,:])
    # print("51", S_to_compare_with_duplicates[51,:,:])

    S_diff = np.abs( S_DFT_to_compare.reshape(-1,3) - S_to_compare_with_duplicates.reshape(-1,3) )
    E_diff = np.abs( dft_bands.reshape(-1) - E_to_compare_with_duplicates.reshape(-1) )

    # print('!!!!!!!!!!!Fermi_nsc_wann', Fermi_nsc_wann)

    # plot error-colored band structure
    plot_err_vs_bands(kpoints, kpath, kpath_ticks, Eigs_k, E_diff, S_diff, \
                      fout='ERRORS_ALL_band_structure_home-made_Fermi_corrected.jpg', \
                        yaxis_lim=yaxis_lim)

    # E_F was already subtracted
    E = dft_bands.reshape(-1)

    # make final matrix
    error_E_S_by_energy = np.vstack([E, E_diff, S_diff[:,0], S_diff[:,1], S_diff[:,2]]).T

    print("Error matrix shape:", error_E_S_by_energy.shape)

    # order the '' matrix by energy (0th column)
    error_E_S_by_energy = error_E_S_by_energy[error_E_S_by_energy[:, 0].argsort()]

    np.savetxt("home-made_quality_error_S_E_Fermi_corrected.dat", error_E_S_by_energy, header="E (eV)\t|Delta E| (eV)\t|Delta S_x|\t|Delta S_y|\t|Delta S_z|")

    # plot all the errors
    fig, axes = plt.subplots(2, 2, figsize=[6,4])

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

    # apply ylim
    if yaxis_lim:
        for ax in axes.flatten():
            ax.set_xlim(yaxis_lim)

    plt.savefig("ERRORS_all_home-made_Fermi_corrected.png", dpi=400)
    plt.close()

    dis_froz_min, dis_froz_max = get_frozen_window_min_max(wannier90winfile=f"{wann_dir}/wannier90.win")

    Fermi_sc = float(np.loadtxt(f"{sc_dir}/FERMI_ENERGY.in"))
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