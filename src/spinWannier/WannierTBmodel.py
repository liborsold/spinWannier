import numpy as np
import pickle
import sys
from wannier_utils import files_wann90_to_dict_pickle, eigenval_dict, load_dict, load_lattice_vectors, reciprocal_lattice_vectors, \
                            get_kpoint_path, get_2D_kpoint_mesh, interpolate_operator, unite_spn_dict, save_bands_and_spin_texture, \
                                uniform_real_space_grid, get_DFT_kgrid, plot_bands_spin_texture, magmom_OOP_or_IP, \
                                fermi_surface_spin_texture, plot_bands_spin_texture
from os import makedirs
from os.path import exists
import copy


class WannierTBmodel():

    def __init__(self, save_folder="./tb_model_wann90/"):
        """Initialize and load the model."""

        # ensure that save_folder has a backslash at the end
        if save_folder[-1] != "/": save_folder += "/"

        # create the save_folder if it does not exist
        if not exists(save_folder): makedirs(save_folder)

        # load the model files and convert to pickle dictionaries
        disentanglement = exists('wannier90_u_dis.mat')
        files_wann90_to_dict_pickle(disentanglement=disentanglement)
        eig_dict = eigenval_dict(eigenval_file="wannier90.eig",  win_file="wannier90.win")
        u_dict = load_dict(fin="u_dict.pickle")
        if disentanglement is True:
            u_dis_dict = load_dict(fin="u_dis_dict.pickle")
        else:
            u_dis_dict = copy.copy(u_dict)
            NW = int(u_dict[list(u_dict.keys())[0]].shape[0])
            for key in u_dis_dict.keys():
                u_dis_dict[key] = np.eye(NW)
        spn_dict = load_dict(fin="spn_dict.pickle")

        # store the model
        self.eig_dict = eig_dict
        self.u_dict = u_dict
        self.u_dis_dict = u_dis_dict
        self.spn_dict = spn_dict
        self.NW = NW

        # define constants
        self.spn_x_R_dict_name = "spn_x_R_dict.pickle"
        self.spn_y_R_dict_name = "spn_y_R_dict.pickle"
        self.spn_z_R_dict_name = "spn_z_R_dict.pickle"
        self.spn_R_dict_name   = "spn_R_dict.pickle"
        

        # store the real space grid
        self.R_grid = uniform_real_space_grid(R_mesh_ijk=get_DFT_kgrid(fin="wannier90.win")) #real_space_grid_from_hr_dat(fname="wannier90_hr.dat") #


    def interpolate_bands_and_spin(self, kmesh_2D=False, kmesh_density=100, kmesh_2D_limits=[-0.5, 0.5], save_real_space_operators=False, save_folder="", \
                                   save_bands_spin_texture=True):
        
        # create the kpoint_matrix if it does not exist
        if not kpoint_matrix:
            kpoint_matrix = [
                [(0.333333,  0.333333,  0.000000),    (0.00,  0.00,  0.00)],
                [(0.00,  0.00,  0.00),    (0.50, 0.00,  0.00)],
                [(0.50, 0.00,  0.00),    (0.333333,  0.333333,  0.000000)]
            ]

        A = load_lattice_vectors(win_file="wannier90.win")
        G = reciprocal_lattice_vectors(A)

        # save to 1D or 2D variables depending if the interpolation is done for 1D or 2D
        Eigs_k_to_save = self.Eigs_k_2D if kmesh_2D else self.Eigs_k_1D
        S_mn_k_H_x_to_save = self.S_mn_k_H_x_2D if kmesh_2D else self.S_mn_k_H_x_1D
        S_mn_k_H_y_to_save = self.S_mn_k_H_y_2D if kmesh_2D else self.S_mn_k_H_y_1D
        S_mn_k_H_z_to_save = self.S_mn_k_H_z_2D if kmesh_2D else self.S_mn_k_H_z_1D
        kpath_to_save = self.kpath_2D if kmesh_2D else self.kpath_1D
        kpoints_rec_to_save = self.kpoints_rec_2D if kmesh_2D else self.kpoints_rec_1D
        kpoints_cart_to_save = self.kpoints_cart_2D if kmesh_2D else self.kpoints_cart_1D

        if kmesh_2D is True:
            kpoints_rec, kpoints_cart = get_2D_kpoint_mesh(G, limits=kmesh_2D_limits, Nk=kmesh_density)
            kpath = []
        else:
            kpoints_rec, kpoints_cart, self.kpath = get_kpoint_path(kpoint_matrix, G, Nk=kmesh_density)
            # -> interpolate at the original k-points:
            # kpoints_rec = list(set([kpoint[0] for kpoint in list(spn_dict.keys())]))
            # kpath = list(range(len(kpoints_rec)))  # just to have some path
        
        Eigs_k_to_save, self.U_mn_k = interpolate_operator(self.eig_dict, self.u_dis_dict, self.u_dict, hamiltonian=True, 
                                                latt_params=A, reciprocal_latt_params=G, R_grid=self.R_grid, kpoints=kpoints_rec, 
                                                save_real_space=save_real_space_operators, 
                                                real_space_fname="hr_R_dict.pickle")
        
        spn_dict_x = {}
        spn_dict_y = {}
        spn_dict_z = {}
        for k, v in spn_dict.items():
            spn_dict_x[k[0]] = spn_dict[(k[0], 'x')]
            spn_dict_y[k[0]] = spn_dict[(k[0], 'y')]
            spn_dict_z[k[0]] = spn_dict[(k[0], 'z')]

        S_mn_k_H_x_to_save = interpolate_operator(spn_dict_x, self.u_dis_dict, self.u_dict, hamiltonian=False, U_mn_k=self.U_mn_k, 
                                            latt_params=A, reciprocal_latt_params=G, R_grid=self.R_grid, kpoints=kpoints_rec,
                                            save_real_space=save_real_space_operators, real_space_fname=self.spn_x_R_dict_name)
        S_mn_k_H_y_to_save = interpolate_operator(spn_dict_y, self.u_dis_dict, self.u_dict, hamiltonian=False, U_mn_k=self.U_mn_k, 
                                            latt_params=A, reciprocal_latt_params=G, R_grid=self.R_grid, kpoints=kpoints_rec,
                                            save_real_space=save_real_space_operators, real_space_fname=self.spn_y_R_dict_name)
        S_mn_k_H_z_to_save = interpolate_operator(spn_dict_z, self.u_dis_dict, self.u_dict, hamiltonian=False, U_mn_k=self.U_mn_k, 
                                            latt_params=A, reciprocal_latt_params=G, R_grid=self.R_grid, kpoints=kpoints_rec,
                                            save_real_space=save_real_space_operators, real_space_fname=self.spn_z_R_dict_name)

        # unite the spn real space dictionaries to one
        with open(save_folder+self.spn_x_R_dict_name, 'rb') as fxr:
            spn_x_dict = pickle.load(fxr)
        with open(save_folder+self.spn_y_R_dict_name, 'rb') as fyr:
            spn_y_dict = pickle.load(fyr)
        with open(save_folder+self.spn_z_R_dict_name, 'rb') as fzr:
            spn_z_dict = pickle.load(fzr)
                    
        with open(save_folder+self.spn_R_dict_name, 'wb') as fw:
            spn_dict = unite_spn_dict(spn_x_dict, spn_y_dict, spn_z_dict, spin_names=['x','y','z'])
            pickle.dump(spn_dict, fw)

        if save_bands_spin_texture is True:
            save_bands_and_spin_texture(kpoints_rec_to_save, kpoints_cart_to_save, kpath_to_save, Eigs_k_to_save, S_mn_k_H_x_to_save, S_mn_k_H_y_to_save, S_mn_k_H_z_to_save, kmesh_2D=kmesh_2D)

    
    def plot1D_bands(self):
        plot_bands_spin_texture(self.kpoints_rec_1D, self.kpath_1D, self.Eigs_k_1D, self.S_mn_k_H_x_1D, self.S_mn_k_H_y_1D, self.S_mn_k_H_z_1D, fout='spin_texture_1D_home_made.jpg')
    
    
    def plot2D_spin_texture(fig_name="spin_texture_2D_home_made.jpg"):
        # ==========  USER DEFINED  ===============
        fin_1D = "bands_spin.pickle" #"bands_spin_model.pickle" #"./tb_model_wann90/bands_spin.pickle" #"bands_spin_model.pickle" #
        fin_2D = "bands_spin_2D.pickle" #"bands_spin_2D_model.pickle" #"./tb_model_wann90/bands_spin_2D.pickle" #"bands_spin_2D_model.pickle"
        E_F = float(np.loadtxt('./FERMI_ENERGY.in')) #-2.31797502 # for CrTe2 with EFIELD: -2.31797166
        E_to_cut_2D = E_F #E_F # + 1.44
        kmesh_limits = None #[-.5, .5] #None #   # unit 1/A; put 'None' if no limits should be applied
        colorbar_Sx_lim = [-1, 1] #[-0.2, 0.2] #None # put None if they should be determined automatically
        colorbar_Sy_lim = [-1, 1] #[-0.2, 0.2] #None # put None if they should be determined automatically
        colorbar_Sz_lim = [-1, 1] #None # put None if they should be determined automatically

        bands_ylim = [-6, 11]

        band = 8 #160   # 1-indexed
        E_thr = 0.005 #0.020  # eV
        quiver_scale_magmom_OOP = 1.6 # quiver_scale for calculations with MAGMOM purely out-of-plane
        quiver_scale_magmom_IP = 10   # quiver_scale for calculations with MAGMOM purely in-plane
        reduce_by_factor = 1 # take each 'reduce_by_factor' point in the '_all_in_one.jpg' plot
        n_points_for_one_angstrom_radius = 180     #120
        arrow_linewidth = 0.005   # default 0.005 (times width of the plot)
        arrow_head_width = 2.5   # default 3

        scatter_size = 0.8

        scatter_for_quiver = False
        scatter_size_quiver = 0.1

        contour_for_quiver = True
        contour_line_width = 1.5
        # ========================================

        global kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, kpoints1D, kpath1D, bands1D, Sx1D, Sy1D, Sz1D, quiver_scale

        magmom_OOP = magmom_OOP_or_IP('../../sc/INCAR')
        quiver_scale = quiver_scale_magmom_OOP if magmom_OOP is True else quiver_scale_magmom_IP

        # load the data
        with open(fin_2D, 'rb') as fr:
            bands_spin_dat = pickle.load(fr)

        kpoints2D = np.array(bands_spin_dat['kpoints'])
        bands2D = np.array(bands_spin_dat['bands'])
        Sx2D = np.array(bands_spin_dat['Sx'])
        Sy2D = np.array(bands_spin_dat['Sy'])
        Sz2D = np.array(bands_spin_dat['Sz'])
        # S = np.linalg.norm([Sx2D, Sy2D, Sz2D], axis=0)

        # load 1D the data
        if fin_1D is not None:
            with open(fin_1D, 'rb') as fr:
                bands_spin_dat = pickle.load(fr)

        kpoints1D = np.array(bands_spin_dat['kpoints'])
        kpath1D = np.array(bands_spin_dat['kpath'])
        bands1D = np.array(bands_spin_dat['bands'])
        Sx1D = np.array(bands_spin_dat['Sx'])
        Sy1D = np.array(bands_spin_dat['Sy'])
        Sz1D = np.array(bands_spin_dat['Sz'])

        # for band in [1]: #14, 15]: #12, 13]: #range(22):
        #     selected_band_plot(band)
        # for deltaE in np.arange(-1.2, 1.2, 0.1): #[-0.4, -0.3]: #-0.1, 0.0, 0.1):
        #     fermi_surface_spin_texture(E=E_F+deltaE, E_thr=E_thr)

        # selected_band_plot(band)
        if E_to_cut is None: E_to_cut = E_to_cut_2D
        fermi_surface_spin_texture(kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, E=E_to_cut_2D, E_F=E_F, \
                                    E_thr=E_thr, savefig=True, \
                                    fig_name=fig_name, quiver_scale=quiver_scale, scatter_for_quiver=scatter_for_quiver, \
                                    scatter_size_quiver=scatter_size_quiver, scatter_size=scatter_size, \
                                    reduce_by_factor=reduce_by_factor, kmesh_limits=kmesh_limits, \
                                    n_points_for_one_angstrom_radius=n_points_for_one_angstrom_radius, \
                                    contour_for_quiver=contour_for_quiver, arrow_head_width=arrow_head_width, \
                                    arrow_linewidth=arrow_linewidth, contour_line_width=contour_line_width
                                )
    
    def Wannier_quality():
        raise NotImplementedError