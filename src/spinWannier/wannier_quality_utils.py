"""Utility functions for the wannier_quality method of the spinWannier.WannierTBmodel.WannierTBmodel class.
"""

import numpy as np
import matplotlib.pyplot as plt
from spinWannier.wannier_utils import (
    real_to_W_gauge,
    W_gauge_to_H_gauge,
    split_spn_dict,
    get_kpoint_path,
    load_lattice_vectors,
    reciprocal_lattice_vectors,
    check_file_exists,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import sys
# sys.path.append('/home/lv268562/bin/python_scripts/wannier_quality/wann_quality_calculation')
# from wannier_quality import parse_eigenval_file, plot_err_vs_energy


def get_fermi(path="."):
    """Extract Fermi energy from the DOSCAR file.

    Args:
        path (str, optional): Path to the directory with the DOSCAR file. Defaults to ".".

    Returns:
        float: Fermi energy in eV.
    """
    with open(path + "/DOSCAR", "r") as fr:
        for i, line in enumerate(fr):
            if i == 5:
                EF = line.split()[-2]
                with open(path + "/FERMI_ENERGY.in", "w") as fw:
                    fw.write(
                        f"# FERMI ENERGY extracted with the get_EF.py script\n{EF}"
                    )
                return float(EF)


def get_band_at_kpoint_from_EIGENVAL(
    EIGENVAL_path="./EIGENVAL",
    target_band=1,
    target_kpoint_string="0.0000000E+00  0.0000000E+00  0.0000000E+00",
):
    """
    Get the energy of the target_band at the target_kpoint from the EIGENVAL file.

    Args:
        EIGENVAL_path (str, optional): Path to the EIGENVAL file. Defaults to './EIGENVAL'.
        target_band (int, optional): Target band. Defaults to 1.
        target_kpoint_string (str, optional): Target k-point string. Defaults to '0.0000000E+00  0.0000000E+00  0.0000000E+00'.

    Returns:
        float: Energy of the target_band at the target_kpoint.
    """
    with open(EIGENVAL_path, "r") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            if target_kpoint_string in line:
                energy = float(lines[i + target_band].split()[1])
                return energy

def vasp_calc_collinear(EIGENVAL_path="./EIGENVAL"):
    """
    Get the N_eig from the EIGENVAL file.

    Args:
        EIGENVAL_path (str, optional): Path to the EIGENVAL file. Defaults to './EIGENVAL'.

    Returns:
        int: N_eig (1 - non-spin-polarized, 2 - spin-polarized).
    """
    with open(EIGENVAL_path, "r") as fr:
        for i, line in enumerate(fr):
            if i == 0:
                N_eig = int(line.split()[3])
                if N_eig == 2:
                    return True
                elif N_eig == 1:
                    return False
                else:
                    raise ValueError(f"The 4th number on the first line of the EIGENVAL file {EIGENVAL_path} is neither 1 (non-collinear) nor 2 (collinear)!")
            

def get_fermi_corrected_by_matching_bands(
    path=".",
    nsc_calculation_path="../0_nsc_for_wann_25x25_frozmaxmargin_0.2eV",
    corrected_at_kpoint="0.0000000E+00  0.0000000E+00  0.0000000E+00",
    corrected_at_band=11,
    sc_calculation_path="../sc",
    fout_name="FERMI_ENERGY_corrected.in",
):
    """Get the Fermi energy from the self-consistent calculation and correct it so that the band at the target_kpoint and target_band is at the same energy in the non-self-consistent calculation.

    Args:
        path (str, optional): Path to the directory with the DOSCAR file. Defaults to ".".
        nsc_calculation_path (str, optional): Path to the non-self-consistent calculation directory. Defaults to '../0_nsc_for_wann_25x25_frozmaxmargin_0.2eV'.
        corrected_at_kpoint (str, optional): Target k-point string. Defaults to '0.0000000E+00  0.0000000E+00  0.0000000E+00'.
        corrected_at_band (int, optional): Target band. Defaults to 11.
        sc_calculation_path (str, optional): Path to the self-consistent calculation directory. Defaults to "../sc".
        fout_name (str, optional): Output file name. Defaults to "FERMI_ENERGY_corrected.in".

    Returns:
        float: Corrected Fermi energy in eV.
    """

    Fermi_sc = get_fermi(sc_calculation_path)

    collinear_sc = vasp_calc_collinear(EIGENVAL_path=sc_calculation_path+"/EIGENVAL")
    collinear_nsc = vasp_calc_collinear(EIGENVAL_path=nsc_calculation_path+"/EIGENVAL") 
    if collinear_sc != collinear_nsc:
        corrected_at_band_sc = corrected_at_band//2 + corrected_at_band%2 if collinear_sc else corrected_at_band
        corrected_at_band_nsc = corrected_at_band//2 + corrected_at_band%2 if collinear_nsc else corrected_at_band

        sc_is = 'collinear' if collinear_sc else 'non-collinear'
        nsc_is = 'collinear' if collinear_nsc else 'non-collinear'
        print(f'!! The self-consistent calculation in {sc_calculation_path} is {sc_is}, while the non-self-consistent calculation in {nsc_calculation_path} is {nsc_is}.\n \
              The band index for correction of sc is now {corrected_at_band_sc} and for nsc is {corrected_at_band_nsc}.')
    else:
        corrected_at_band_sc = corrected_at_band
        corrected_at_band_nsc = corrected_at_band

    sc_band = get_band_at_kpoint_from_EIGENVAL(
        EIGENVAL_path=sc_calculation_path + "/EIGENVAL",
        target_band=corrected_at_band_sc,
        target_kpoint_string=corrected_at_kpoint,
    )
    nsc_band = get_band_at_kpoint_from_EIGENVAL(
        EIGENVAL_path=nsc_calculation_path + "/EIGENVAL",
        target_band=corrected_at_band_nsc,
        target_kpoint_string=corrected_at_kpoint,
    )

    # WE REQUIRE THAT:
    #   band_nsc - Fermi_nsc = band_sc - Fermi_sc
    Fermi_nsc = Fermi_sc + nsc_band - sc_band

    with open(nsc_calculation_path + "/" + fout_name, "w") as fw:
        fw.write(
            f"# {Fermi_nsc:.8f} eV = {Fermi_sc:.8f} + ({nsc_band:.8f} - {sc_band:.8f}) = Fermi_sc + (E_band{corrected_at_band_nsc}_nsc - E_band{corrected_at_band_sc}_sc) ... self-consistent Fermi from {sc_calculation_path} corrected so that band {corrected_at_band_nsc} of non-self-consistent calculation and {corrected_at_band_sc} of self-consistent calculation at k-point {corrected_at_kpoint} is at the same energy (relative to the recpective Fermi energies) for the self-consistent (path {sc_calculation_path}) and non-self-consistent (path {nsc_calculation_path}) calculation\n{Fermi_nsc:.8f}"
        )

    return float(Fermi_nsc)


def compare_eigs_bandstructure_at_exact_kpts(
    dft_bands,
    wann_bands,
    num_kpoints,
    num_wann,
    f_name_out="WannierBerri_quality_error_Fermi_corrected.dat",
):
    """Compare the DFT and Wannierized band structures at the exact k-points.

    Args:
        dft_bands (np.array): DFT bands.
        wann_bands (np.array): Wannierized bands.
        num_kpoints (int): Number of k-points.
        num_wann (int): Number of Wannierized bands.
        f_name_out (str, optional): Output file name. Defaults to 'WannierBerri_quality_error_Fermi_corrected.dat'.

    Returns:
        np.array: Array with the DFT energy in the first column and the Wannierization error in the second column.
    """

    error_by_energy = np.zeros((num_kpoints * num_wann, 2))
    for i in range(num_kpoints):
        # save the DFT energy as x variable
        error_by_energy[i * num_wann : (i + 1) * num_wann, 0] = dft_bands[i, :]

        # save the Wannierization error as y variable
        error_by_energy[i * num_wann : (i + 1) * num_wann, 1] = (
            dft_bands[i, :] - wann_bands[i, :]
        ) ** 2

    # sort 'error_by_energy' by energy - by the first column
    error_by_energy = error_by_energy[error_by_energy[:, 0].argsort()]

    # save to file
    header = "E (eV)\t(E_wann - E_DFT)**2"
    fmt = "%.12f\t%.12e"
    np.savetxt(f_name_out, error_by_energy, delimiter="\t", header=header, fmt=fmt)
    return error_by_energy


def duplicate_kpoints_for_home_made(data, NK):
    """Duplicate also the last k-point (in  dictionary the keys are unique, so actually the data in the dictionaries where keys are k-points contain only one of each k-point, so if k-path starts and ends with the same k-point, only the first one is recorded.

    Args:
        data (np.array): Data to duplicate.
        NK (int): Number of k-points.

    Returns:
        np.array: Duplicated data.
    """
    dimensions = list(np.array(data).shape)
    # print(dimensions[0])
    # print(NK)
    n_blocs = (dimensions[0]) // (NK - 1)
    dimensions[0] += n_blocs

    data_duplicates = np.zeros(dimensions)

    data_duplicates[:NK, :] = np.real(data[:NK, :])
    for i in range(1, n_blocs - 1):
        data_duplicates[i * NK, :] = np.real(data[i * (NK - 1), :])
        data_duplicates[i * NK + 1 : i * NK + 1 + NK, :] = np.real(data[
            1 + i * (NK - 1) : 1 + i * (NK - 1) + NK, :
        ])
    # last block different - missing last value (supposed to be the same as the first value)
    i = n_blocs - 1
    data_duplicates[i * NK, :] = np.real(data[i * (NK - 1), :])
    data_duplicates[i * NK + 1 : i * NK + 1 + NK - 2, :] = np.real(data[
        1 + i * (NK - 1) : 1 + i * (NK - 1) + NK - 2, :
    ])
    data_duplicates[-1, :] = np.real(data[0, :])
    return data_duplicates


def get_NKpoints(OUTCAR="OUTCAR"):
    """Return number of kpoints from bands calculation from OUTCAR.

    Args:
        OUTCAR (str, optional): OUTCAR path. Defaults to 'OUTCAR'.

    Returns:
        int: number of k-points stated in the OUTCAR
    """
    with open(OUTCAR, "r") as fr:
        for line in fr:
            if "NKPTS =" in line:
                return int(line.split()[3])


def parse_eigenval_file(fin, spin=0):
    """Parse the EIGENVAL file and return the kpoints, bands, number of kpoints, and number of bands.

    Args:
        fin (str): Path to the EIGENVAL file.
        spin (int, optional): Spin index. Defaults to 0.

    Returns:
        np.array: kpoints.
        np.array: bands.
        int: number of kpoints.
        int: number of bands.
    """
    flag_kpoint = False
    flag_bands = False
    i_k = -1
    i_b = -1
    num_bands = 99999999
    with open(fin, "r") as fr:
        for i, line in enumerate(fr):

            l = line.split()

            if i == 5:
                num_bands = int(l[2])
                num_kpoints = int(l[1])
                kpoints = np.zeros((num_kpoints, 3))
                bands = np.zeros((num_kpoints, num_bands))

            if flag_bands == True:
                i_b += 1
                bands[i_k, i_b] = l[spin + 1]

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
    for i in range(num_kpoints - 1):
        kpoints_path[i + 1] = (
            kpoints_path[i] + np.sum((kpoints[i + 1, :] - kpoints[i, :]) ** 2) ** 0.5
        )

    return kpoints_path, bands, num_kpoints, num_bands


def plot_err_vs_energy(
    error_by_energy,
    Ef,
    title="Wannierization RMS error vs. energy",
    fig_name_out="wannier_quality_error_by_energy.png",
    savefig=True,
    showfig=True,
):
    """Plot the error vs. energy.

    Args:
        error_by_energy (np.array): Error vs. energy.
        Ef (float): Fermi energy.
        title (str, optional): Title of the plot. Defaults to "Wannierization RMS error vs. energy".
        fig_name_out (str, optional): Output file name. Defaults to "wannier_quality_error_by_energy.png".
        savefig (bool, optional): Save the figure. Defaults to True.
        showfig (bool, optional): Show the figure. Defaults to True.
    """
    plt.semilogy(
        error_by_energy[:, 0] - Ef,
        error_by_energy[:, 1] ** 0.5,
        "ok",
        linewidth=1,
        markersize=1,
    )
    plt.title(title, fontsize=8)
    plt.xlabel(r"$E - E_\mathrm{F}$ (eV)")
    plt.ylabel(r"$\|E_\mathrm{wann} - E_\mathrm{DFT}\|$ (eV)")
    if savefig:
        plt.savefig(fig_name_out, dpi=400)
    if showfig:
        plt.show()
    plt.close()


def plot_err_vs_bands(
    kpoints,
    kpath,
    kpath_ticks,
    Eigs_k,
    E_diff,
    S_diff,
    NW,
    fout="ERRORS_ALL_band_structure.jpg",
    yaxis_lim=None,
    savefig=True,
    showfig=True,
):
    """Output a figure with RMSE_E, RMSE_Sx, RMSE_Sy, and RMSE_Sz-projected band structure.

    Args:
        kpoints (np.array): kpoints.
        kpath (np.array): kpath.
        kpath_ticks (list): kpath ticks.
        Eigs_k (dict): Eigs_k.
        E_diff (np.array): E_diff.
        S_diff (np.array): S_diff.
        fout (str, optional): Output file name. Defaults to 'ERRORS_ALL_band_structure.jpg'.
        yaxis_lim (list, optional): y-axis limits. Defaults to None.
        savefig (bool, optional): Save the figure. Defaults to True.
        showfig (bool, optional): Show the figure. Defaults to True.
    """
    Nk = len(kpoints) // (len(kpath_ticks) - 1)

    fig, axes = plt.subplots(2, 2, figsize=[9, 6])
    fig.suptitle("Error of Wannier interpolation", fontsize=14)
    spin_name = [r"Energy (eV)", r"$S_z$", r"$S_x$", r"$S_y$"]
    for i, S in enumerate([E_diff, S_diff[:, 2], S_diff[:, 0], S_diff[:, 1]]):
        ax = axes[i // 2, i % 2]
        ax.axhline(linestyle="--", color="k")
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
        sc = ax.scatter(
            [[k_dist for i in range(NW)] for k_dist in kpath],
            [Eigs_k[i] for i in range(len(kpoints))],
            c=S.reshape(-1, NW),
            cmap="YlOrRd",
            s=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sc, cax=cax, orientation="vertical")

        for j in range(1, len(kpath_ticks) - 1):
            ax.axvline(x=kpath[j * Nk], color="#000000", linestyle="-", linewidth=0.75)

        # if i == 0 or i == 2:
        ax.set_ylabel(r"$E - E_\mathrm{F}$ (eV)", fontsize=13)
        # if i == 0 or i == 1:
        secax = ax.secondary_xaxis("top")
        secax.tick_params(labelsize=9)  # axis='both', which='major',
        secax.set_xlabel(r"$k$-distance (1/$\mathrm{\AA}$)", fontsize=10)
        # no primary x-axis
        ax.set_xticks([])

        # if i == 2 or i == 3:
        idx = np.array(range(0, Nk * len(kpath_ticks), Nk))
        idx[-1] += -1
        ax.set_xticks(kpath[idx])
        ax.set_xticklabels(kpath_ticks, fontsize=12)
        ax.yaxis.set_tick_params(labelsize=11)

        # cbar.set_label(r'$S_\mathrm{z}$')
        # sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    # plt.suptitle(fig_caption)
    plt.tight_layout()
    # plt.show()
    fout = check_file_exists(fout)
    if savefig:
        plt.savefig(fout, dpi=400)
    if showfig:
        plt.show()
    plt.close()
    # print(
    #     "ERROR band structure printed\n\n--------------------\n\n======================="
    # )


def integrate_error(error_by_energy, E_min=-1e3, E_max=1e3):
    """Integrate the error in 'f_name_in' in the energy range [E_min, E_max] included.

    Args:
        error_by_energy (np.array): Error vs. energy.
        E_min (float, optional): Minimum energy. Defaults to -1e3.
        E_max (float, optional): Maximum energy. Defaults to 1e3.

    Returns:
        np.array: Array with the integrated error.
    """
    E = error_by_energy[:, 0]
    err_array = error_by_energy[(E_min <= E) & (E <= E_max)][:, 1:]
    err_array_squared = np.power(err_array, 2)
    return np.power(err_array_squared.mean(axis=0), 0.5)


def get_frozen_window_min_max(wannier90winfile="wannier90.win"):
    """Get the frozen window min and max from the wannier90.win file.

    Args:
        wannier90winfile (str, optional): Path to the wannier90.win file. Defaults to 'wannier90.win'.

    Returns:
        float: Frozen window min.
        float: Frozen window max.
    """
    dis_froz_min = None
    dis_froz_max = None

    with open(wannier90winfile, "r") as fr:
        for line in fr:
            if "dis_froz_min" in line:
                dis_froz_min = float(
                    line.split("dis_froz_min")[1].split("=")[1].split(" ")[0]
                )
            if "dis_froz_max" in line:
                dis_froz_max = float(
                    line.split("dis_froz_max")[1].split("=")[1].split(" ")[0]
                )
    return dis_froz_min, dis_froz_max

