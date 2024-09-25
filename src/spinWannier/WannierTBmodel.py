"""Contains the WannierTBmodel class for the spin-texture interpolation and plotting.
"""

import numpy as np
import pickle
from os import makedirs
from os.path import exists
import copy
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
    plot_bands_spin_texture,
    u_to_dict,
    spn_to_dict,
    parse_KPOINTS_file,
    save_bands_and_spin_texture_old,
)
from spinWannier.wannier_quality_utils import (
    wannier_quality_calculation,
    get_fermi_corrected_by_matching_bands,
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
        sc_dir="0_self-consistent",
        nsc_dir="1_non-self-consistent",
        wann_dir="2_wannier",
        bands_dir="1_band_structure",
        tb_model_dir="2_wannier/tb_model_wann90",
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
            sc_dir (str, optional): Directory of the self-consistent calculation. Defaults to '0_self-consistent'.
            nsc_dir (str, optional): Directory of the non-self-consistent calculation. Defaults to '1_non-self-consistent'.
            wann_dir (str, optional): Directory of the wannier calculation. Defaults to '2_wannier'.
            bands_dir (str, optional): Directory of the band structure calculation. Defaults to '1_band_structure'.
            tb_model_dir (str, optional): Directory of the tight-binding model. Defaults to '2_wannier/tb_model_wann90'.
            spn_formatted (bool, optional): Whether the spn file is formatted (human-readable) or not (i.e., it is binary). Defaults to False.
            spn_file_extension (str, optional): Extension of the spn file. Defaults to 'spn'.
            data_saving_format (str, optional): Format to save the data. Defaults to 'parquet'.
            band_for_Fermi_correction (int, optional): Band number for Fermi correction. Defaults to None -> the lowest Wannier band wil be used.
            kpoint_for_Fermi_correction (str, optional): K-point for Fermi correction. Defaults to '0.0000000E+00  0.0000000E+00  0.0000000E+00' (Gamma point).
        """

        self.data_saving_format = data_saving_format
        self.verbose = verbose

        # ensure that directories have a backslash at the end
        if sc_dir[-1] != "/":
            sc_dir += "/"
        if nsc_dir[-1] != "/":
            nsc_dir += "/"
        if wann_dir[-1] != "/":
            wann_dir += "/"
        if bands_dir[-1] != "/":
            bands_dir += "/"
        if tb_model_dir[-1] != "/":
            tb_model_dir += "/"

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
                if "num_wann" in line:
                    num_wann = int(line.split("=")[1].split("#")[0].strip())
                if "num_bands" in line:
                    num_bands = int(line.split("=")[1].split("#")[0].strip())

        if band_for_Fermi_correction is None:
            band_for_Fermi_correction = discard_first_bands + 1

        if num_wann == -1 or num_bands == -1:
            print("num_wann or num_bands not found in wannier90.win !")
            exit(1)
        self.NW_dis = num_bands
        self.NW = num_wann

        # create the save_folder if it does not exist
        if not exists(tb_model_dir):
            makedirs(tb_model_dir)

        # load the model files and convert to pickle dictionaries
        disentanglement = exists(wann_dir + f"{seedname}_u_dis.mat")

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

        if num_bands_from_u != num_bands or num_wann_from_u != num_wann:
            print(
                "The number of bands or wannier functions in the u.mat files do not match the number of bands or wannier functions in the wannier90.win file."
            )
            print("Please check the u.mat files.")
            exit(1)

        # check if wannier90.spn_formatted exists; if not, generated it from WAVECAR
        if not exists(wann_dir + f"{seedname}.{spn_file_extension}"):
            print(
                f"{seedname}.{spn_file_extension} does not exist. Generating it from WAVECAR."
            )
            # generate wannier90.spn_formatted from WAVECAR
            if not exists(nsc_dir + "WAVECAR"):
                print("WAVECAR does not exist in the self-consistent directory.")
                print(
                    "Please provide the WAVECAR file to generate wannier90.spn_formatted."
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

        # convert wannier90.spn_formatted to a pickled dictionary
        spn_to_dict(
            model_dir=wann_dir,
            fwin=f"{seedname}.win",
            fin=f"{seedname}.{spn_file_extension}",
            formatted=spn_formatted,
        )

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

        spn_dict = load_dict(fin=wann_dir + "spn_dict.pickle")

        # calculate the Fermi level
        # get Fermi for the wannier non-self-consistent calculation
        self.EF_nsc = get_fermi_corrected_by_matching_bands(
            path=".",
            nsc_calculation_path=nsc_dir,
            corrected_at_kpoint=kpoint_for_Fermi_correction,
            corrected_at_band=band_for_Fermi_correction,
            sc_calculation_path=sc_dir,
            fout_name="FERMI_ENERGY_corrected.in",
        )

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
        self.spn_dict = spn_dict
        self.discard_first_bands = discard_first_bands
        self.seedname = seedname

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
        save_real_space_operators=True,
        save_folder_in_model_dir="tb_model_wann90/",
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
            save_folder_in_model_dir (str, optional): Folder to save the data in the model directory. Defaults to "tb_model_wann90/".
            save_bands_spin_texture (bool, optional): Whether to save the bands and spin texture. Defaults to True.
        """

        save_folder = self.wann_dir + save_folder_in_model_dir
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
        if exists(save_folder + fout):
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

        spn_dict_x = {}
        spn_dict_y = {}
        spn_dict_z = {}
        for k, v in self.spn_dict.items():
            spn_dict_x[k[0]] = self.spn_dict[(k[0], "x")]
            spn_dict_y[k[0]] = self.spn_dict[(k[0], "y")]
            spn_dict_z[k[0]] = self.spn_dict[(k[0], "z")]

        if self.verbose: print("Interpolating Sx...")
        self.S_mn_k_H_x[dimension] = interpolate_operator(
            spn_dict_x,
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
            spn_dict_y,
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
            spn_dict_z,
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

        if save_bands_spin_texture is True and kmesh_2D is False:
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
        elif save_bands_spin_texture is True and kmesh_2D is True:
            save_bands_and_spin_texture_old(
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

        # clean standard output
        

    def plot1D_bands(self, fout="spin_texture_1D_home_made.jpg", yaxis_lim=[-8, 6], savefig=True, showfig=True):
        """Plot the bands and the spin texture on a 1D path.

        Args:
            fout (str, optional): Output figure name. Defaults to 'spin_texture_1D_home_made.jpg'.
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
            E_F=self.EF_nsc,
            fout=fout,
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
            1.6  # quiver_scale for calculations with MAGMOM purely out-of-plane
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
        Sy2D = np.array(bands_spin_dat["Sy"])
        Sz2D = np.array(bands_spin_dat["Sz"])
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
            fig_name=fig_name,
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

    def wannier_quality(
        self,
        band_for_Fermi_correction=None,
        kpoint_for_Fermi_correction="0.0000000E+00  0.0000000E+00  0.0000000E+00",
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

        wannier_quality_calculation(
            kpoint_matrix,
            NK,
            kpath_ticks,
            self.EF_nsc,
            num_wann=self.NW,
            discard_first_bands=self.discard_first_bands,
            sc_dir=self.sc_dir,
            nsc_dir=self.nsc_dir,
            wann_dir=self.wann_dir,
            bands_dir=self.bands_dir,
            tb_model_dir=self.tb_model_dir,
            band_for_Fermi_correction=band_for_Fermi_correction,
            kpoint_for_Fermi_correction=kpoint_for_Fermi_correction,
            yaxis_lim=yaxis_lim,
            savefig=savefig,
            showfig=showfig,
        )
