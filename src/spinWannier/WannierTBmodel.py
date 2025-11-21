"""Contains the WannierTBmodel class for the spin-texture interpolation and plotting.
"""

import numpy as np
import pickle
from os import makedirs
from os.path import exists
import copy
from sys import exit
import shutil
import matplotlib.pyplot as plt
from spinWannier.wannier_utils import (
    files_wann90_to_dict_pickle,
    eigenval_dict,
    load_dict,
    load_lattice_vectors,
    reciprocal_lattice_vectors,
    get_kpoint_path,
    get_2D_kpoint_mesh,
    interpolate_operator,
    unite_spn_dict,
    save_bands_and_spin_texture,
    uniform_real_space_grid,
    get_DFT_kgrid,
    plot_bands_spin_texture,
    magmom_OOP_or_IP,
    fermi_surface_spin_texture,
    u_to_dict,
    spn_to_dict,
    parse_KPOINTS_file,
    save_bands_and_spin_texture_old,
)
from spinWannier.wannier_quality_utils import (
    get_fermi,
    get_fermi_corrected_by_matching_bands,
    get_band_at_kpoint_from_EIGENVAL,
    compare_eigs_bandstructure_at_exact_kpts,
    duplicate_kpoints_for_home_made,
    get_NKpoints,
    parse_eigenval_file,
    plot_err_vs_bands,
    plot_err_vs_energy,
    integrate_error,
    get_frozen_window_min_max,
)
from spinWannier.vaspspn import vasp_to_spn


class WannierTBmodel:
    """Calculates interpolated spin-texture on a dense k-point path.
    Plots the spin-texture on a 2D k-point mesh (Fermi surface).
    Calculates the quality of the Wannier functions.

    Needed files are
        (1) wannier90.eig
        (2) wannier90.win
        (3) wannier90.spn (or the WAVECAR file of VASP)
        (4) wannier90_u.mat
        (5) wannier90_u_dis.mat (if disentanglement was performed)
        (6) (FERMI_ENERGY.in file containing the Fermi energy value)

    PROCEDURE:
        Take the
            (1) spn_dict.dat  and
            (2) u_dis_dict.dat  along with
            (3) u_dict.dat
        to obtain the S_mn(W), i.e., the spn matrix in the Wannier gauge.

        Then interpolate on an arbitrary k-point by performing the
            (a) Fourier transform to real space (on the coarse DFT k-point grid) and then
            (b) inverse Fourier transform with an arbitrary k-point.

        Interpolate the Hamiltonian on a dense k-point grid:
            (1) Take the diagonal Hamiltonian from DFT (the eigenvalues for the coarse k-point grid).
            (2) Apply the U(dis) and U to get the Hamiltonian in Wannier gauge H_mn(W).
            (3) inverse Fourier transform for an arbitrary k-point (dense k-point mesh).

    """

    def __init__(
        self,
        seedname="wannier90",
        spin_polarized=True,
        sc_dir="0_self-consistent",
        nsc_dir="1_non-self-consistent",
        wann_dir="2_wannier",
        bands_dir="1_band_structure",
        output_dir = "3_wann_interp_plot_and_quality",
        tb_model_dir_in_output_dir="tb_model_wann90",
        spn_formatted=False,
        spn_file_extension="spn",
        data_saving_format="npz", #"parquet",
        band_for_Fermi_correction=None,
        kpoint_for_Fermi_correction="0.0000000E+00  0.0000000E+00  0.0000000E+00",
        verbose=False,
    ):
        """Initialize the WannierTBmodel class.

        Args:
            seedname (str, optional): Seedname of the wannier90 files. Defaults to 'wannier90'.
            spin_polarized (bool, optional): Whether the calculation is spin-polarized or not. Defaults to True.
            sc_dir (str, optional): Directory of the self-consistent calculation. Defaults to '0_self-consistent'.
            nsc_dir (str, optional): Directory of the non-self-consistent calculation. Defaults to '1_non-self-consistent'.
            wann_dir (str, optional): Directory of the wannier calculation. Defaults to '2_wannier'.
            bands_dir (str, optional): Directory of the band structure calculation. Defaults to '1_band_structure'.
            tb_model_dir_in_output_dir (str, optional): Directory of the tight-binding model INSIDE THE wann_dir. Defaults to 'tb_model_wann90'.
            spn_formatted (bool, optional): Whether the spn file is formatted (human-readable) or not (i.e., it is binary). Defaults to False.
            spn_file_extension (str, optional): Extension of the spn file. Defaults to 'spn'.
            data_saving_format (str, optional): Format to save the data. Defaults to 'parquet'.
            band_for_Fermi_correction (int, optional): Band number for Fermi correction. Defaults to None -> the lowest Wannier band wil be used.
            kpoint_for_Fermi_correction (str, optional): K-point for Fermi correction. Defaults to '0.0000000E+00  0.0000000E+00  0.0000000E+00' (Gamma point).
        """

        self.data_saving_format = data_saving_format
        self.verbose = verbose
        self.kpoint_for_Fermi_correction = kpoint_for_Fermi_correction
        self.spin_polarized = spin_polarized

        # ensure that directories have a backslash at the end
        if sc_dir is not None and sc_dir[-1] != "/": sc_dir += "/"
        if nsc_dir is not None and nsc_dir[-1] != "/": nsc_dir += "/"
        if wann_dir is not None and wann_dir[-1] != "/": wann_dir += "/"
        if bands_dir is not None and bands_dir[-1] != "/": bands_dir += "/"
        if tb_model_dir_in_output_dir is not None and tb_model_dir_in_output_dir[-1] != "/": tb_model_dir_in_output_dir += "/"
        if output_dir is not None and output_dir[-1] != "/": output_dir += "/"

        self.output_dir = output_dir

        # load important info from wannier90.win
        discard_first_bands = 0
        num_wann = -1
        num_bands = -1
        with open(wann_dir + f"{seedname}.win", "r") as f:
            for line in f:
                if "exclude_bands" in line:
                    # e.g. "exclude_bands = 1-10, 37-64"
                    discard_first_bands = int(
                        line.split("=")[1].split("-")[1].split(",")[0].strip()
                    )
                    if verbose: print(f"Excluding the first {discard_first_bands} bands (read from {wann_dir + seedname}.win)")
                if "num_wann" in line:
                    num_wann = int(line.split("=")[1].split("#")[0].strip())
                    if verbose: print(f"Number of Wannier functions: {num_wann} (read from {wann_dir + seedname}.win)")
                if "num_bands" in line:
                    num_bands = int(line.split("=")[1].split("#")[0].strip())
                    if verbose: print(f"Number of bands: {num_bands} (read from {wann_dir + seedname}.win)")

        if band_for_Fermi_correction is None:
            band_for_Fermi_correction = discard_first_bands + 1

        if num_wann == -1 or num_bands == -1:
            print("num_wann or num_bands not found in wannier90.win !")
            exit(1)
        self.NW_dis = num_bands
        self.NW = num_wann

        # create the save_folder if it does not exist
        tb_model_dir = output_dir + tb_model_dir_in_output_dir
        if not exists(output_dir):
            makedirs(output_dir)
            if verbose: print(f"Created the output directory: {output_dir}")
        if not exists(tb_model_dir):
            makedirs(tb_model_dir)
            if verbose: print(f"Created the tb_model directory: {tb_model_dir}")

        # load the model files and convert to pickle dictionaries
        disentanglement = exists(wann_dir + f"{seedname}_u_dis.mat")
        if verbose and disentanglement: print(f"Disentanglement is ON (u_dis.mat exists).")
        if verbose and not disentanglement: print(f"Disentanglement is OFF (u_dis.mat does not exist).")

        # Convert wannier90 files to pickled dictionaries files
        num_bands_from_u, num_wann_from_u = u_to_dict(
            fin=wann_dir + f"{seedname}_u.mat",
            fout=wann_dir + "u_dict.pickle",
            text_file=False,
            write_sparse=False,
        )
        if disentanglement is True:
            num_bands_from_u, num_wann_from_u = u_to_dict(
                fin=wann_dir + f"{seedname}_u_dis.mat",
                fout=wann_dir + "u_dis_dict.pickle",
                text_file=False,
                write_sparse=False,
            )

        # if num_bands_from_u != num_bands or num_wann_from_u != num_wann:
        #     print(
        #         f"Discrepancy in the number of bands {num_bands_from_u} != {num_bands} or the wannier functions {num_wann_from_u} != {num_wann} between the dis_u.mat file and the wannier90.win file, respectively."
        #     )
        #     print("Please check the u.mat and u_dis.mat files.")
        #     exit(1)

        if spin_polarized is True:
            # check if wannier90.spn_formatted exists; if not, generated it from WAVECAR
            if not exists(wann_dir + f"{seedname}.{spn_file_extension}"):
                print(
                    f"{seedname}.{spn_file_extension} does not exist. Generating it from WAVECAR."
                )
                # generate wannier90.spn_formatted from WAVECAR
                if not exists(nsc_dir + "WAVECAR"):
                    print(f"WAVECAR does not exist in {nsc_dir}.")
                    print(
                        f"Please provide the WAVECAR file to generate {seedname}.{spn_file_extension}."
                    )
                    exit(1)
                else:
                    if spn_formatted is False:
                        print("Generating wannier90.spn from WAVECAR.")
                        vasp_to_spn(
                            formatted=False,
                            fin=nsc_dir + "WAVECAR",
                            fout=wann_dir + f"{seedname}.spn",
                            NBout=self.NW_dis,
                            IBstart=discard_first_bands + 1,
                        )
                    elif spn_formatted is True:
                        print("Generating wannier90.spn_formatted from WAVECAR.")
                        vasp_to_spn(
                            formatted=True,
                            fin=nsc_dir + "WAVECAR",
                            fout=wann_dir + f"{seedname}.spn_formatted",
                            NBout=self.NW_dis,
                            IBstart=discard_first_bands + 1,
                        )

            # convert wannier90.spn to a pickled dictionary
            spn_to_dict(
                model_dir=wann_dir,
                fwin=f"{seedname}.win",
                fin=f"{seedname}.{spn_file_extension}",
                formatted=spn_formatted,
            )
            
            spn_dict = load_dict(fin=wann_dir + "spn_dict.pickle")

        eig_dict = eigenval_dict(
            eigenval_file=wann_dir + f"{seedname}.eig",
            win_file=wann_dir + f"{seedname}.win",
        )
        u_dict = load_dict(fin=wann_dir + "u_dict.pickle")
        if disentanglement is True:
            u_dis_dict = load_dict(fin=wann_dir + "u_dis_dict.pickle")
        else:
            u_dis_dict = copy.copy(u_dict)
            NW = int(u_dict[list(u_dict.keys())[0]].shape[0])
            for key in u_dis_dict.keys():
                u_dis_dict[key] = np.eye(NW)
            self.NW = NW

        # calculate the Fermi level
        # get Fermi for the wannier non-self-consistent calculation
        EF_nsc_fout = "FERMI_ENERGY_corrected.in"
        if sc_dir is None or nsc_dir is None:
            print("sc_dir or nsc_dir is None. Skipping Fermi level calculation.")
            self.EF_nsc = float(np.loadtxt(wann_dir + "FERMI_ENERGY.in"))
            if verbose: print(f"Fermi level read from {wann_dir + 'FERMI_ENERGY.in'}: {self.EF_nsc:.3f} eV")
        else:
            if verbose: print("Calculating the Fermi level corrected by matching bands...")
            self.EF_nsc = get_fermi_corrected_by_matching_bands(
                nsc_calculation_path=nsc_dir,
                corrected_at_kpoint=kpoint_for_Fermi_correction,
                corrected_at_band=band_for_Fermi_correction,
                sc_calculation_path=sc_dir,
                fout_name=EF_nsc_fout,
            )
            if verbose: print(f'Calculated Fermi level: {self.EF_nsc:.3f} eV saved to {nsc_dir}{EF_nsc_fout}')

        # store paths to directories
        self.sc_dir = sc_dir
        self.nsc_dir = nsc_dir
        self.wann_dir = wann_dir
        self.bands_dir = bands_dir
        self.tb_model_dir = tb_model_dir

        # store the model
        self.eig_dict = eig_dict
        self.u_dict = u_dict
        self.u_dis_dict = u_dis_dict
        self.discard_first_bands = discard_first_bands
        self.seedname = seedname
        self.band_for_Fermi_correction = band_for_Fermi_correction
        
        if spin_polarized:
            self.spn_dict = spn_dict
            # split the spn_dict into x, y, z components
            spn_dict_x = {}
            spn_dict_y = {}
            spn_dict_z = {}
            for k, v in spn_dict.items():
                spn_dict_x[k[0]] = spn_dict[(k[0], "x")]
                spn_dict_y[k[0]] = spn_dict[(k[0], "y")]
                spn_dict_z[k[0]] = spn_dict[(k[0], "z")]
            self.spn_dict_x = spn_dict_x
            self.spn_dict_y = spn_dict_y
            self.spn_dict_z = spn_dict_z

            # define constants
            self.spn_x_R_dict_name = "spn_x_R_dict.pickle"
            self.spn_y_R_dict_name = "spn_y_R_dict.pickle"
            self.spn_z_R_dict_name = "spn_z_R_dict.pickle"
            self.spn_R_dict_name = "spn_R_dict.pickle"

        # store the real space grid
        self.R_grid = uniform_real_space_grid(
            R_mesh_ijk=get_DFT_kgrid(fin=wann_dir + f"{seedname}.win")
        )  # real_space_grid_from_hr_dat(fname=f"{seedname}_hr.dat") #

        print("Wannier model constructed!")

    def interpolate_bands_and_spin(
        self,
        kpoint_matrix,
        fout="bands_spin",
        kpath_ticks=None,
        kmesh_2D=False,
        kmesh_density=101,
        kmesh_2D_limits=[-0.5, 0.5],
        force_recalculate=False,
        save_real_space_operators=True,
        save_folder_in_output_dir="tb_model_wann90/",
        save_bands_spin_texture=True,
    ):
        """Interpolate the bands and the spin texture on a dense k-point path or 2D mesh.

        Args:
            kpoint_matrix (list): List of k-points in the path.
            fout (str, optional): Output file name. Defaults to 'bands_spin'.
            kpath_ticks (list, optional): List of k-point ticks. Defaults to None.
            kmesh_2D (bool, optional): Whether to interpolate on a 2D mesh. Defaults to False.
            kmesh_density (int, optional): Density of the k-point mesh. Defaults to 101.
            kmesh_2D_limits (list, optional): Limits of the 2D mesh. Defaults to [-0.5, 0.5].
            save_real_space_operators (bool, optional): Whether to save the real space operators as dictionary files. Defaults to True.
            save_folder_in_output_dir (str, optional): Folder to save the data in the model directory. Defaults to "tb_model_wann90/".
            save_bands_spin_texture (bool, optional): Whether to save the bands and spin texture. Defaults to True.
        """

        save_folder = self.output_dir + save_folder_in_output_dir
        if save_folder[-1] != "/":
            save_folder += "/"

        fout = f"{fout}.{self.data_saving_format}"
        #  !!!!!!!!!!!!!!!!!! continue

        A = load_lattice_vectors(win_file=self.wann_dir + f"{self.seedname}.win")
        G = reciprocal_lattice_vectors(A)

        dimension = "2D" if kmesh_2D is True else "1D"

        # initialize local variables
        self.Eigs_k = {}
        self.S_mn_k_H_x = {}
        self.S_mn_k_H_y = {}
        self.S_mn_k_H_z = {}
        self.kpoints_rec = {}
        self.kpoints_cart = {}
        self.kpath = {}

        if kmesh_2D is True:
            self.kpoints_rec[dimension], self.kpoints_cart[dimension] = (
                get_2D_kpoint_mesh(G, limits=kmesh_2D_limits, Nk=kmesh_density)
            )
            self.kpath[dimension] = []
        else:
            (
                self.kpoints_rec[dimension],
                self.kpoints_cart[dimension],
                self.kpath[dimension],
            ) = get_kpoint_path(kpoint_matrix, G, Nk=kmesh_density)
            if kpath_ticks is None:
                kpath_ticks = [f"P{i}" for i in range(len(kpoint_matrix) + 1)]
            self.kpath_ticks = kpath_ticks
            # -> interpolate at the original k-points:
            # kpoints_rec = list(set([kpoint[0] for kpoint in list(spn_dict.keys())]))
            # kpath = list(range(len(kpoints_rec)))  # just to have some path

        # --- only calculate if the files do not exist ---
        if exists(save_folder + fout) and force_recalculate is False:
            print(
                f"The file {fout} already exists in f'{save_folder}'.\nSkipping the calculation, loading the data instead!\n"
            )
            # load the data
            with open(save_folder + fout, "rb") as fr:
                bands_spin_dat = pickle.load(fr)
            self.Eigs_k[dimension] = bands_spin_dat["bands"]
            self.S_mn_k_H_x[dimension] = bands_spin_dat["Sx"]
            self.S_mn_k_H_y[dimension] = bands_spin_dat["Sy"]
            self.S_mn_k_H_z[dimension] = bands_spin_dat["Sz"]
            return

        if self.verbose: print("Interpolating the Hamiltonian...")
        self.Eigs_k[dimension], self.U_mn_k = interpolate_operator(
            self.eig_dict,
            self.u_dis_dict,
            self.u_dict,
            hamiltonian=True,
            latt_params=A,
            reciprocal_latt_params=G,
            R_grid=self.R_grid,
            kpoints=self.kpoints_rec[dimension],
            save_real_space=save_real_space_operators,
            real_space_fname="hr_R_dict.pickle",
            save_folder=save_folder,
            verbose=self.verbose,
        )
        
        if self.spin_polarized:
            if self.verbose: print("Interpolating Sx...")
            self.S_mn_k_H_x[dimension] = interpolate_operator(
                self.spn_dict_x,
                self.u_dis_dict,
                self.u_dict,
                hamiltonian=False,
                U_mn_k=self.U_mn_k,
                latt_params=A,
                reciprocal_latt_params=G,
                R_grid=self.R_grid,
                kpoints=self.kpoints_rec[dimension],
                save_real_space=save_real_space_operators,
                real_space_fname=self.spn_x_R_dict_name,
                save_folder=save_folder,
                verbose=self.verbose,
            )

            if self.verbose: print("Interpolating Sy...")
            self.S_mn_k_H_y[dimension] = interpolate_operator(
                self.spn_dict_y,
                self.u_dis_dict,
                self.u_dict,
                hamiltonian=False,
                U_mn_k=self.U_mn_k,
                latt_params=A,
                reciprocal_latt_params=G,
                R_grid=self.R_grid,
                kpoints=self.kpoints_rec[dimension],
                save_real_space=save_real_space_operators,
                real_space_fname=self.spn_y_R_dict_name,
                save_folder=save_folder,
                verbose=self.verbose,
            )

            if self.verbose: print("Interpolating Sz...")
            self.S_mn_k_H_z[dimension] = interpolate_operator(
                self.spn_dict_z,
                self.u_dis_dict,
                self.u_dict,
                hamiltonian=False,
                U_mn_k=self.U_mn_k,
                latt_params=A,
                reciprocal_latt_params=G,
                R_grid=self.R_grid,
                kpoints=self.kpoints_rec[dimension],
                save_real_space=save_real_space_operators,
                real_space_fname=self.spn_z_R_dict_name,
                save_folder=save_folder,
                verbose=self.verbose,
            )

            # unite the spn real space dictionaries to one
            with open(save_folder + self.spn_x_R_dict_name, "rb") as fxr:
                spn_x_dict = pickle.load(fxr)
            with open(save_folder + self.spn_y_R_dict_name, "rb") as fyr:
                spn_y_dict = pickle.load(fyr)
            with open(save_folder + self.spn_z_R_dict_name, "rb") as fzr:
                spn_z_dict = pickle.load(fzr)

            with open(save_folder + self.spn_R_dict_name, "wb") as fw:
                spn_dict = unite_spn_dict(
                    spn_x_dict, spn_y_dict, spn_z_dict, spin_names=["x", "y", "z"]
                )
                pickle.dump(spn_dict, fw)

            if save_bands_spin_texture is True: # and kmesh_2D is False:
                save_bands_and_spin_texture(
                    self.kpoints_rec[dimension],
                    self.kpoints_cart[dimension],
                    self.kpath[dimension],
                    self.Eigs_k[dimension],
                    self.S_mn_k_H_x[dimension],
                    self.S_mn_k_H_y[dimension],
                    self.S_mn_k_H_z[dimension],
                    save_folder=save_folder,
                    kmesh_2D=kmesh_2D,
                    fout=fout,
                )
            # elif save_bands_spin_texture is True and kmesh_2D is True:
            #     save_bands_and_spin_texture_old(
            #         self.kpoints_rec[dimension],
            #         self.kpoints_cart[dimension],
            #         self.kpath[dimension],
            #         self.Eigs_k[dimension],
            #         self.S_mn_k_H_x[dimension],
            #         self.S_mn_k_H_y[dimension],
            #         self.S_mn_k_H_z[dimension],
            #         save_folder=save_folder,
            #         kmesh_2D=kmesh_2D,
            #         fout=fout,
            #     )

            # clean standard output
        

    def plot1D_bands(self, fout="spin_texture_1D_home_made.jpg", yaxis_lim=[-8, 6], savefig=True, showfig=True):
        """Plot the bands and the spin texture on a 1D path.

        Args:
            fout (str, optional): Output figure name. Defaults to 'spin_texture_1D_home_made.jpg'. Will be saved to the 'output_dir' directory.
            yaxis_lim (list, optional): Y-axis limits. Defaults to [-8, 6].
            savefig (bool, optional): Whether to save the figure. Defaults to True.
            showfig (bool, optional): Whether to show the figure. Defaults to True.
        """
        plot_bands_spin_texture(
            self.kpoints_rec["1D"],
            self.kpath["1D"],
            self.kpath_ticks,
            self.Eigs_k["1D"],
            self.S_mn_k_H_x["1D"],
            self.S_mn_k_H_y["1D"],
            self.S_mn_k_H_z["1D"],
            NW=self.NW,
            E_F=self.EF_nsc,
            fout=self.output_dir + fout,
            yaxis_lim=yaxis_lim,
            savefig=savefig,
            showfig=showfig,
        )

    def plot2D_spin_texture(
        self,
        fin_2D="bands_spin_2D",
        fin_1D="bands_spin",
        fig_name="spin_texture_2D_home_made.jpg",
        E_to_cut=None,
        savefig=True,
        showfig=True,
        E_window=0.020,
        n_points_for_one_angstrom_radius=60,
        quiver_scale=1.6,
    ):
        """Plot the spin texture on a 2D mesh (Fermi surface).

        Args:
            fin_2D (str, optional): File name of the 2D mesh data. Defaults to 'bands_spin_2D'.
            fin_1D (str, optional): File name of the 1D path data. Defaults to 'bands_spin'.
            fig_name (str, optional): Output figure name. Defaults to "spin_texture_2D_home_made.jpg".
            E_to_cut (float, optional): Energy to cut. Defaults to None.
            savefig (bool, optional): Whether to save the figure. Defaults to True.
            showfig (bool, optional): Whether to show the figure. Defaults to True.
            E_window (float, optional): Energy window in eV around the Fermi level where the points for Fermi surface determination will be looked for. Defaults to 0.020.
            n_points_for_one_angstrom_radius (int, optional): Number of points for one Angstrom radius. Defaults to 180.
        """
        # ==========  USER DEFINED  ===============
        fin_1D = self.tb_model_dir + fin_1D + "." + self.data_saving_format
        fin_2D = self.tb_model_dir + fin_2D + "." + self.data_saving_format
        E_F = float(np.loadtxt(self.wann_dir + "FERMI_ENERGY.in"))
        E_to_cut_2D = E_F  # E_F # + 1.44
        kmesh_limits = None  # [-.5, .5] #None #   # unit 1/A; put 'None' if no limits should be applied
        colorbar_Sx_lim = [
            -1,
            1,
        ]  # [-0.2, 0.2] #None # put None if they should be determined automatically
        colorbar_Sy_lim = [
            -1,
            1,
        ]  # [-0.2, 0.2] #None # put None if they should be determined automatically
        colorbar_Sz_lim = [
            -1,
            1,
        ]  # None # put None if they should be determined automatically

        bands_ylim = [-6, 11]

        band = 8  # 160   # 1-indexed
        quiver_scale_magmom_OOP = (
            quiver_scale  # quiver_scale for calculations with MAGMOM purely out-of-plane
        )
        quiver_scale_magmom_IP = (
            10  # quiver_scale for calculations with MAGMOM purely in-plane
        )
        reduce_by_factor = (
            1  # take each 'reduce_by_factor' point in the '_all_in_one.jpg' plot
        )
        arrow_linewidth = 0.005  # default 0.005 (times width of the plot)
        arrow_head_width = 2.5  # default 3

        scatter_size = 0.8

        scatter_for_quiver = False
        scatter_size_quiver = 0.1

        contour_for_quiver = True
        contour_line_width = 1.5
        # ========================================

        magmom_OOP = True  # magmom_OOP_or_IP('../../sc/INCAR')
        quiver_scale = (
            quiver_scale_magmom_OOP if magmom_OOP is True else quiver_scale_magmom_IP
        )

        # load the data
        with open(fin_2D, "rb") as fr:
            bands_spin_dat = pickle.load(fr)

        kpoints2D = np.array(bands_spin_dat["kpoints"])
        bands2D = np.array(bands_spin_dat["bands"])
        Sx2D = np.array(bands_spin_dat["Sx"])
        # take the real part of the diagonal over axes 1 and 2
        Sx2D = np.real(np.diagonal(Sx2D, axis1=1, axis2=2))
        Sy2D = np.array(bands_spin_dat["Sy"])
        Sy2D = np.real(np.diagonal(Sy2D, axis1=1, axis2=2))
        Sz2D = np.array(bands_spin_dat["Sz"])
        Sz2D = np.real(np.diagonal(Sz2D, axis1=1, axis2=2))
        # S = np.linalg.norm([Sx2D, Sy2D, Sz2D], axis=0)

        # load 1D the data
        if fin_1D is not None:
            with open(fin_1D, "rb") as fr:
                bands_spin_dat = pickle.load(fr)

        kpoints1D = np.array(bands_spin_dat["kpoints"])
        kpath1D = np.array(bands_spin_dat["kpath"])
        bands1D = np.array(bands_spin_dat["bands"])
        Sx1D = np.array(bands_spin_dat["Sx"])
        Sy1D = np.array(bands_spin_dat["Sy"])
        Sz1D = np.array(bands_spin_dat["Sz"])

        # for band in [1]: #14, 15]: #12, 13]: #range(22):
        #     selected_band_plot(band)
        # for deltaE in np.arange(-1.2, 1.2, 0.1): #[-0.4, -0.3]: #-0.1, 0.0, 0.1):
        #     fermi_surface_spin_texture(E=E_F+deltaE, E_thr=E_thr)

        # selected_band_plot(band)
        if E_to_cut is None:
            E_to_cut = E_to_cut_2D

        fermi_surface_spin_texture(
            kpoints2D,
            bands2D,
            Sx2D,
            Sy2D,
            Sz2D,
            E=E_to_cut_2D,
            E_F=E_F,
            E_thr=E_window,
            fig_name=self.output_dir + fig_name,
            quiver_scale=quiver_scale,
            scatter_for_quiver=scatter_for_quiver,
            scatter_size_quiver=scatter_size_quiver,
            scatter_size=scatter_size,
            reduce_by_factor=reduce_by_factor,
            kmesh_limits=kmesh_limits,
            n_points_for_one_angstrom_radius=n_points_for_one_angstrom_radius,
            contour_for_quiver=contour_for_quiver,
            arrow_head_width=arrow_head_width,
            arrow_linewidth=arrow_linewidth,
            contour_line_width=contour_line_width,
            savefig=savefig,
            showfig=showfig,
        )

        
    def wannier_quality_calculation(
        self,
        kpoint_matrix,
        NK,
        kpath_ticks,
        yaxis_lim=None,
        savefig=True,
        showfig=True,
    ):
        """Calculate the quality of the Wannierization.

        Needed files from VASP:

            = nsc_calculation_path:
                - EIGENVAL
                - DOSCAR

            = dft_bands_folder:
                - EIGENVAL
                - "Sxyz_exp_values_from_spn_file.dat" (get automatically from wannier90.spn_formatted!!)
                - OUTCAR

        Args:
            kpoint_matrix (np.array): K-point matrix.
            NK (int): Number of k-points.
            kpath_ticks (list): K-path ticks.
            Fermi_nsc_wann (float): Fermi energy in the non-self-consistent calculation.
            yaxis_lim (list, optional): Y-axis limits. Defaults to None.
            savefig (bool, optional): Save the figure. Defaults to True.
            showfig (bool, optional): Show the figure. Defaults to True.

        Returns:
            np.array: Error by energy.
        """

        # ================================================== CONSTANTS ==================================================================

        labels = ["G", "K", "M", "G"]

        S_DFT_fname = "Sxyz_exp_values_from_spn_file.dat"
        hr_R_name = self.tb_model_dir + "/hr_R_dict.pickle"  # "hr_R_dict_sym.pickle"
        spn_R_name = self.tb_model_dir + "/spn_R_dict.pickle"

        deltaE_around_EF = (
            0.5  # eV; integrate error in this +- window around E_F for plotting
        )
        deltaE2_around_EF = (
            0.1  # eV; integrate error in this +- window around E_F for plotting
        )

        # =============================================================================================================================

        # DFT bands for comparison
        dft_kpoints, dft_bands, num_kpoints_dft, num_bands = parse_eigenval_file(
            self.bands_dir + "/EIGENVAL"
        )
        dft_bands = dft_bands[:, self.discard_first_bands : self.discard_first_bands + self.NW]
        print('NW', self.NW)
        print('dft_bands_shape', dft_bands.shape)

        # ======================== GET HOME-MADE SPIN TEXTURE ================================
        A = load_lattice_vectors(win_file=f"{self.wann_dir}/wannier90.win")
        G = reciprocal_lattice_vectors(A)
        kpoints, kpoints_cart, kpath = get_kpoint_path(kpoint_matrix, G, Nk=NK)


        if self.verbose: print("Interpolating the Hamiltonian for Wannier quality...")
        Eigs_k, U_mn_k = interpolate_operator(
            self.eig_dict,
            self.u_dis_dict,
            self.u_dict,
            hamiltonian=True,
            latt_params=A,
            reciprocal_latt_params=G,
            R_grid=self.R_grid,
            kpoints=kpoints,
            save_real_space=False,
            save_folder=self.output_dir,
            verbose=self.verbose,
        )

        # take all elements from Eigs_k and subtract EF_nsc
        Eigs_k -= self.EF_nsc
        print("Eigs_k.shape", Eigs_k.shape)

        if self.verbose: print("Interpolating Sx for Wannier quality...")
        S_mn_k_H_x = interpolate_operator(
            self.spn_dict_x,
            self.u_dis_dict,
            self.u_dict,
            hamiltonian=False,
            U_mn_k=U_mn_k,
            latt_params=A,
            reciprocal_latt_params=G,
            R_grid=self.R_grid,
            kpoints=kpoints,
            save_real_space=False,
            save_folder=self.output_dir,
            verbose=self.verbose,
        )
        print("S_mn_k.shape", S_mn_k_H_x.shape)

        if self.verbose: print("Interpolating Sy for Wannier quality...")
        S_mn_k_H_y = interpolate_operator(
            self.spn_dict_y,
            self.u_dis_dict,
            self.u_dict,
            hamiltonian=False,
            U_mn_k=U_mn_k,
            latt_params=A,
            reciprocal_latt_params=G,
            R_grid=self.R_grid,
            kpoints=kpoints,
            save_real_space=False,
            save_folder=self.output_dir,
            verbose=self.verbose,
        )

        if self.verbose: print("Interpolating Sz for Wannier quality...")
        S_mn_k_H_z = interpolate_operator(
            self.spn_dict_z,
            self.u_dis_dict,
            self.u_dict,
            hamiltonian=False,
            U_mn_k=U_mn_k,
            latt_params=A,
            reciprocal_latt_params=G,
            R_grid=self.R_grid,
            kpoints=kpoints,
            save_real_space=False,
            save_folder=self.output_dir,
            verbose=self.verbose,
        )

        # # interpolate Hamiltonian
        # with open(hr_R_name, "rb") as fr:
        #     hr_R_dict = pickle.load(fr)

        # H_k_W = real_to_W_gauge(kpoints, hr_R_dict)
        # Eigs_k, U_mn_k = W_gauge_to_H_gauge(H_k_W, U_mn_k={}, hamiltonian=True)

        # shutil.copyfile(
        #     nsc_dir + "/FERMI_ENERGY_corrected.in", "./FERMI_ENERGY_corrected.in"
        # )
        # E_F = Fermi_nsc_wann
        # # take all elements from Eigs_k and subtract Fermi_nsc_wann
        # for key in Eigs_k.keys():
        #     Eigs_k[key] -= Fermi_nsc_wann

        # # interpolate spin operator
        # with open(spn_R_name, "rb") as fr:
        #     spn_R_dict = pickle.load(fr)

        # S_mn_R_x, S_mn_R_y, S_mn_R_z = split_spn_dict(
        #     spn_R_dict, spin_names=["x", "y", "z"]
        # )

        # Sx_k_W = real_to_W_gauge(kpoints, S_mn_R_x)
        # Sy_k_W = real_to_W_gauge(kpoints, S_mn_R_y)
        # Sz_k_W = real_to_W_gauge(kpoints, S_mn_R_z)

        # # print('len kpoints', len(kpoints))
        # # print('len H_k_W', len(list(H_k_W.keys())))

        # S_mn_k_H_x = W_gauge_to_H_gauge(Sx_k_W, U_mn_k=U_mn_k, hamiltonian=False)
        # S_mn_k_H_y = W_gauge_to_H_gauge(Sy_k_W, U_mn_k=U_mn_k, hamiltonian=False)
        # S_mn_k_H_z = W_gauge_to_H_gauge(Sz_k_W, U_mn_k=U_mn_k, hamiltonian=False)

        # =====================================================================================

        # get Fermi for the bands non-self-consistent calculation
        Fermi_dft_bands = get_fermi_corrected_by_matching_bands(
            nsc_calculation_path=self.bands_dir,
            corrected_at_kpoint=self.kpoint_for_Fermi_correction,
            corrected_at_band=self.band_for_Fermi_correction,
            sc_calculation_path=self.sc_dir,
            fout_name="FERMI_ENERGY_corrected.in",
        )
        dft_bands -= Fermi_dft_bands

        # make 2D again: (NKpoints, num_wann)
        # E_to_compare = np.array([Eigs_k[key] for key in Eigs_k.keys()])
        # print("E_to_compare_shape", E_to_compare.shape)
        # E_to_compare_with_duplicates = duplicate_kpoints_for_home_made(E_to_compare, NK)
        # print("E_to_compare_with_duplicates_shape", E_to_compare_with_duplicates.shape)

        # above NOT NECESSARY SINCE NOW NUMPY ARRAYS PRODUCED DURING INTERPOLATION, NOT DICTIONARIES
        E_to_compare_with_duplicates = Eigs_k

        # print("50", E_to_compare_with_duplicates[50,:])
        # print("51", E_to_compare_with_duplicates[51,:])

        # S_mn_k_H_x_to_compare = duplicate_kpoints_for_home_made(
        #     np.array([np.diag(S_mn_k_H_x[key]) for key in S_mn_k_H_x.keys()]), NK
        # )
        # S_mn_k_H_y_to_compare = duplicate_kpoints_for_home_made(
        #     np.array([np.diag(S_mn_k_H_y[key]) for key in S_mn_k_H_y.keys()]), NK
        # )
        # S_mn_k_H_z_to_compare = duplicate_kpoints_for_home_made(
        #     np.array([np.diag(S_mn_k_H_z[key]) for key in S_mn_k_H_z.keys()]), NK
        # )

        S_mn_k_H_x_to_compare = np.array([np.diag(S_mn) for S_mn in S_mn_k_H_x])
        S_mn_k_H_y_to_compare = np.array([np.diag(S_mn) for S_mn in S_mn_k_H_y])
        S_mn_k_H_z_to_compare = np.array([np.diag(S_mn) for S_mn in S_mn_k_H_z])

        print("S_mn_k_H_x_to_compare.shape", S_mn_k_H_x_to_compare.shape)

        # print('k-points from dict keys', list(S_mn_k_H_x.keys()))

        S_to_compare_with_duplicates = np.array(
            [S_mn_k_H_x_to_compare, S_mn_k_H_y_to_compare, S_mn_k_H_z_to_compare]
        )
        S_shape = S_to_compare_with_duplicates.shape

        # make the spin axis the last instead of the first one
        S_to_compare_with_duplicates = S_to_compare_with_duplicates.swapaxes(0, 1)
        S_to_compare_with_duplicates = S_to_compare_with_duplicates.swapaxes(1, 2)

        # S_to_compare_with_duplicates = duplicate_kpoints(S_to_compare, NK)

        # error_by_energy = compare_eigs_bandstructure_at_exact_kpts(
        #     dft_bands,
        #     E_to_compare_with_duplicates,
        #     num_kpoints_dft,
        #     num_wann,
        #     f_name_out="home-made_quality_error_Fermi_corrected.dat",
        # )

        # plot_err_vs_energy(
        #     error_by_energy,
        #     Ef=0,
        #     title="Wannierization RMS error vs. energy",
        #     fig_name_out="wannier_quality_error_by_energy_home-made_Fermi_corrected.png",
        #     savefig=savefig,
        #     showfig=showfig,
        # )

        # ------------- COMPARE spin texture --------------------

        # load spin expectation values and select relevant bands
        S_DFT = np.loadtxt(f"{self.bands_dir}/{S_DFT_fname}")
        # select relevant bands
        NK = get_NKpoints(OUTCAR=f"{self.bands_dir}/OUTCAR")
        S_DFT = S_DFT.reshape(NK, -1, 3)
        # S_DFT_to_compare = S_DFT[:, self.discard_first_bands : self.discard_first_bands + self.NW, :]
        S_DFT_to_compare = S_DFT[:, :, :]
        print('S_DFT_to_compare.shape', S_DFT_to_compare.shape)

        # print("S_to_compare dimensions", S_to_compare_with_duplicates.shape)
        # print("S_to_compare", S_to_compare)

        # print("50", S_to_compare_with_duplicates[50,:,:])
        # print("51", S_to_compare_with_duplicates[51,:,:])

        S_diff = np.abs(
            S_DFT_to_compare.reshape(-1, 3) - S_to_compare_with_duplicates.reshape(-1, 3)
        )
        E_diff = np.abs(dft_bands.reshape(-1) - E_to_compare_with_duplicates.reshape(-1))

        # print('!!!!!!!!!!!Fermi_nsc_wann', Fermi_nsc_wann)

        # plot error-colored band structure
        plot_err_vs_bands(
            kpoints,
            kpath,
            kpath_ticks,
            Eigs_k,
            E_diff,
            S_diff,
            NW=self.NW,
            fout=self.output_dir + "ERRORS_ALL_band_structure_home-made_Fermi_corrected.jpg",
            yaxis_lim=yaxis_lim,
            savefig=savefig,
            showfig=showfig,
        )

        # E_F was already subtracted
        E = dft_bands.reshape(-1)

        # make final matrix
        error_E_S_by_energy = np.vstack(
            [E, E_diff, S_diff[:, 0], S_diff[:, 1], S_diff[:, 2]]
        ).T

        # print("Error matrix shape:", error_E_S_by_energy.shape)

        # order the '' matrix by energy (0th column)
        error_E_S_by_energy = error_E_S_by_energy[error_E_S_by_energy[:, 0].argsort()]

        np.savetxt(
            self.output_dir + "home-made_quality_error_S_E_Fermi_corrected.dat",
            error_E_S_by_energy,
            header="E (eV)\t|Delta E| (eV)\t|Delta S_x|\t|Delta S_y|\t|Delta S_z|",
        )

        # plot all the errors
        fig, axes = plt.subplots(2, 2, figsize=[6, 4])

        axes[0, 0].semilogy(
            error_E_S_by_energy[:, 0], error_E_S_by_energy[:, 1], "ko", markersize=2
        )
        axes[0, 0].set_title(r"$E$", fontsize=14)
        axes[0, 0].set_ylabel(r"|$E_\mathrm{DFT} - E_\mathrm{wann}$| (eV)")

        axes[0, 1].semilogy(
            error_E_S_by_energy[:, 0], error_E_S_by_energy[:, 4], "bo", markersize=2
        )
        axes[0, 1].set_title(r"$S_z$", fontsize=14)
        axes[0, 1].set_ylabel(r"|$S_{z, \mathrm{DFT}} - S_{z, \mathrm{wann}}$|")

        axes[1, 0].semilogy(
            error_E_S_by_energy[:, 0], error_E_S_by_energy[:, 2], "ro", markersize=2
        )
        axes[1, 0].set_title(r"$S_x$", fontsize=14)
        axes[1, 0].set_ylabel(r"|$S_{x, \mathrm{DFT}} - S_{x, \mathrm{wann}}$|")
        axes[1, 0].set_xlabel(r"$E - E_\mathrm{F}$ (eV)")

        axes[1, 1].semilogy(
            error_E_S_by_energy[:, 0], error_E_S_by_energy[:, 3], "go", markersize=2
        )
        axes[1, 1].set_title(r"$S_y$", fontsize=14)
        axes[1, 1].set_ylabel(r"|$S_{y, \mathrm{DFT}} - S_{y, \mathrm{wann}}$|")
        axes[1, 1].set_xlabel(r"$E - E_\mathrm{F}$ (eV)")

        fig.suptitle("Error of Wannier interpolation", fontsize=14)
        plt.tight_layout()

        # apply ylim
        if yaxis_lim:
            for ax in axes.flatten():
                ax.set_xlim(yaxis_lim)
        if savefig is True:
            plt.savefig(self.output_dir + "ERRORS_all_home-made_Fermi_corrected.png", dpi=400)
        if showfig is True:
            plt.show()
        plt.close()

        dis_froz_min, dis_froz_max = get_frozen_window_min_max(
            wannier90winfile=f"{self.wann_dir}/wannier90.win"
        )

        Fermi_sc = float(np.loadtxt(f"{self.sc_dir}/FERMI_ENERGY.in"))
        mean_error_whole_range = integrate_error(error_E_S_by_energy, E_min=-1e6, E_max=1e6)
        mean_error_up_to_Fermi = integrate_error(error_E_S_by_energy, E_min=-1e6, E_max=0)
        mean_error_frozen_window = integrate_error(
            error_E_S_by_energy,
            E_min=dis_froz_min - self.EF_nsc,
            E_max=dis_froz_max - self.EF_nsc,
        )
        mean_error_around_Fermi = integrate_error(
            error_E_S_by_energy, E_min=-deltaE_around_EF, E_max=deltaE_around_EF
        )
        mean_error_around_Fermi2 = integrate_error(
            error_E_S_by_energy, E_min=-deltaE2_around_EF, E_max=deltaE_around_EF
        )

        with open(self.output_dir + "error_home-made_integrated_Fermi_corrected.dat", "w") as fw:
            fw.write(
                "#                                           \tRMSE_E (eV)\tRMSE_Sx  \tRMSE_Sy  \tRMSE_Sz\n"
            )
            fw.write(
                f"in the whole energy range                \t"
                + "\t".join([f"{val:.6f}" for val in mean_error_whole_range])
                + "\n"
            )
            fw.write(
                f"up to Fermi                                 \t"
                + "\t".join([f"{val:.6f}" for val in mean_error_up_to_Fermi])
                + "\n"
            )
            fw.write(
                f"in the frozen window ({dis_froz_min-self.EF_nsc:.3f} to {dis_froz_max-Fermi_sc:.3f} eV)\t"
                + "\t".join([f"{val:.6f}" for val in mean_error_frozen_window])
                + "\n"
            )
            fw.write(
                f"window +- {deltaE_around_EF:.3f} eV around Fermi level    \t"
                + "\t".join([f"{val:.6f}" for val in mean_error_around_Fermi])
                + "\n"
            )
            fw.write(
                f"window +- {deltaE2_around_EF:.3f} eV around Fermi level    \t"
                + "\t".join([f"{val:.6f}" for val in mean_error_around_Fermi2])
            )

        # ------------ ABSOLUTE VALUE OF SPIN ------------------

        S_abs = np.linalg.norm(S_to_compare_with_duplicates.reshape(-1, 3), axis=1)
        # print("S_abs shape", S_abs.shape)

        # combined matrix
        E_S_abs = np.vstack([E, S_abs]).T
        # print("S_abs_E shape", E_S_abs.shape)

        # order the '' matrix by energy (0th column)
        E_S_abs = E_S_abs[E_S_abs[:, 0].argsort()]

        # save txt file, with statistics
        Sabs_mean = np.mean(S_abs)
        Sabs_median = np.median(S_abs)
        Sabs_over_one = len(S_abs[S_abs > 1.0]) / len(S_abs)
        header = f"S over 1 ratio = {Sabs_over_one:.8f}\nS mean = {Sabs_mean:.8f}\nS median = {Sabs_median:.8f}\nE (eV)\t|S|"
        np.savetxt(self.output_dir + "home-made_Sabs_vs_E_Fermi_corrected.dat", E_S_abs, header=header)

        # plot and save S_abs vs. E
        plot_title = "S_abs"

        fig, ax = plt.subplots(1, 1, figsize=[3.7, 2.5])
        ax.plot(E_S_abs[:, 0], E_S_abs[:, 1], "ko", markersize=2)
        ax.set_title("Interpolated spin magnitudes", fontsize=13)
        ax.set_ylabel(r"|$S$|", fontsize=13)
        ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)", fontsize=13)
        plt.tight_layout()
        if savefig is True:
            plt.savefig(self.output_dir + plot_title + "_vs_E_home-made_Fermi_corrected.png", dpi=400)
        if showfig is True:
            plt.show()
        plt.close()

        # plot and save S_abs histogram
        fig, ax = plt.subplots(1, 1, figsize=[3.5, 2.5])
        plt.hist(S_abs.flatten(), bins=100)
        plt.xlabel("|S|", fontsize=13)
        plt.ylabel("counts", fontsize=13)
        plt.title(
            "Spin magnitude histogram", #f"histogram of abs values of diagonal elements of spin operator\n- in k-space (eigenvalues) home-made interpolation",
            fontsize=13,
        )
        plt.tight_layout()
        if savefig is True:
            plt.savefig(self.output_dir + plot_title + "_S_histogram_home-made_Fermi_corrected.png", dpi=400)
        if showfig is True:
            plt.show()
        plt.close()


    def wannier_quality(
        self,
        yaxis_lim=[-10, 10],
        savefig=True,
        showfig=True,
    ):
        """Calculate the quality of the Wannier functions compared to the original DFT band structure.

        Args:
            band_for_Fermi_correction (_type_, optional): Band number for Fermi correction. Defaults to None -> the lowest Wannier band wil be used.
            kpoint_for_Fermi_correction (_type_, optional): K-point for Fermi correction. Defaults to '0.0000000E+00  0.0000000E+00  0.0000000E+00' (Gamma point).
            yaxis_lim (list, optional): Limits of the y-axis. Defaults to [-10, 10].
            savefig (bool, optional): Whether to save the figure. Defaults to True.
            showfig (bool, optional): Whether to show the figure. Defaults to
        """
        kpoint_matrix, NK, kpath_ticks = parse_KPOINTS_file(self.bands_dir + "KPOINTS")
        if self.verbose: print(f"K-point matrix: {kpoint_matrix} (read from {self.bands_dir}KPOINTS)")
        if self.verbose: print(f"Number of k-points: {NK} (read from {self.bands_dir}KPOINTS)")
        if self.verbose: print(f"K-point ticks: {kpath_ticks} (read from {self.bands_dir}KPOINTS)")

        self.wannier_quality_calculation(
            kpoint_matrix,
            NK,
            kpath_ticks,
            yaxis_lim=yaxis_lim,
            savefig=savefig,
            showfig=showfig,
        )
