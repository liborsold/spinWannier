"""
Calculates interpolated spin-texture on a dense k-point path.

Needed files are
   (1) wannier90.eig
   (2) wannier90.win
   (3) wannier90.spn_formatted
   (4) wannier90_u.mat
   (5) wannier90_u_dis.mat
   (6) FERMI_ENERGY.in

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
    (3) inverse Fourier transform for an arbitrary k-point (dense k-point mesh)."""

import numpy as np
import pickle
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import get_kpoint_names, eigenval_dict, load_dict, load_lattice_vectors, \
                            reciprocal_lattice_vectors, get_kpoint_path, uniform_real_space_grid, \
                            real_space_grid_from_hr_dat, get_DFT_kgrid, get_2D_kpoint_mesh, \
                            unite_spn_dict
from cmath import exp
import matplotlib.pyplot as plt
from os.path import exists
import os
from spn_to_dict import spn_to_dict
from u_to_dict import u_to_dict
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

# _______ USER INPUT __________
E_F = float(np.loadtxt('FERMI_ENERGY.in')) # -2.31797502 #-2.31797166 # for CrTe2 with EFIELD: 

kmesh_density_1D = 30
kmesh_density_2D = 30

disentanglement = exists('wannier90_u_dis.mat')

kmesh_2D_limits = [-0.5, 0.5]
yaxis_lim = [-6, 11]

run_kmesh_2D = True
save_bands_spin_texture = True
save_real_space_operators = True
save_as_pickle = True

spn_x_R_dict_name = "spn_x_R_dict.pickle"
spn_y_R_dict_name = "spn_y_R_dict.pickle"
spn_z_R_dict_name = "spn_z_R_dict.pickle"
spn_R_dict_name   = "spn_R_dict.pickle"

save_folder = "./tb_model_wann90/"  # !! with backslash at the end
# _____________________________

if not exists(save_folder):
    os.makedirs(save_folder)

R_grid = uniform_real_space_grid(R_mesh_ijk = get_DFT_kgrid(fin="wannier90.win")) #real_space_grid_from_hr_dat(fname="wannier90_hr.dat") #


def files_wann90_to_dict_pickle():
    """Convert wannier90 files to pickled dictionaries files."""
    if not exists("spn_dict.pickle"):
        spn_to_dict()
    if not exists("u_dict.pickle"):
        u_to_dict(fin="wannier90_u.mat", fout="u_dict.pickle", text_file=False, write_sparse=False)
    if disentanglement is True and not exists("u_dis_dict.pickle"):
        u_to_dict(fin="wannier90_u_dis.mat", fout="u_dis_dict.pickle", text_file=False, write_sparse=False)


def interpolate_operator(operator_dict, u_dis_dict, u_dict, latt_params, reciprocal_latt_params, 
                            R_grid, U_mn_k=None, hamiltonian=True, kpoints=[(0,0,0)], 
                            save_real_space=False, real_space_fname="hr_R_dict.dat"):
    """Takes operator O (Hamiltonian, spin-operator ...) evaluated on a coarse DFT k-mesh and Hamiltonian eigenstates
        onto 'kpoints' using the 
        (1) (semi)unitary transformation defined by U_dis*U from the Hamiltonian gauge (eigenstate gauge)
             to the Wannier gauge, 
        (2) performing Fourier transform to the real-space using a set of lattice vectors
        (3) inverse Fourier transforming to the k-space for all the k-points from 'kpoints'."""
    
    global NW, NK

    A = latt_params
    G = reciprocal_latt_params
    NK = len(u_dict.keys())

    if NK != len(R_grid):
        raise Exception("The number of R vectors for discrete Fourier needs to be the same as number of k-points from the DFT mesh!")
    
    NW = u_dict[list(u_dict.keys())[0]].shape[0]

    # PROCEDURE TO GET AUTOMATICALLY THE REAL-SPACE MESH (from the reciprocal Monkhorst-Pack)

    # the DFT k-point grid in direct coordinates
    kpoints_coarse = operator_dict.keys()
    Nq = len(kpoints_coarse)

    # the DFT k-point grid in cartesian coordinates (units of 1/A)
    kpoints_coarse_cart = [np.array(kpoint).T @ G for kpoint in kpoints_coarse]

    # (1) get the operator in the Wannier gauge (see Eg. 16 in Ryoo 2019 PRB or for instance Eq. 30 in Qiao 2018 PRB)
    #     see Eq. 16 in Qiao 2018 and express H(W) from that, use Eq. 10 for definition of U
    #     alternatively see Eq. 15 in Qiao 2018
    O_mn_q_W = {}
    for kpoint, O_k in operator_dict.items():
        kpoint = tuple(0.0 if kpoint[i] == 0.0 else kpoint[i] for i in range(3))
        O_mn_q_W[kpoint] = u_dict[kpoint].conj().T @ u_dis_dict[kpoint].conj().T @ O_k @ u_dis_dict[kpoint] @ u_dict[kpoint]

    # (2) Fourier transform in the range of real-space lattice vectors given by R_mesh_ijk
    O_mn_R_W = {}
    for Rijk in R_grid:
        R = np.array(Rijk).T @ A
        # Eq. 17 in Qiao 2018
        O_mn_R_W[Rijk] = 1/Nq * np.sum( [exp(-1j*np.dot(R, k_cart)) * O_mn_q_W[k_direct] 
                                                            for k_cart, k_direct in zip(kpoints_coarse_cart, kpoints_coarse)], 
                                                            axis = 0)
    # save real-space representation
    if save_real_space is True:
        if save_as_pickle is True:
            with open(save_folder+real_space_fname, 'wb') as fw:
                pickle.dump(O_mn_R_W, fw)
        else:
            O_mn_R_W_tosave = {}
            for k, val in O_mn_R_W.items():
                O_mn_R_W_tosave[k] = repr(O_mn_R_W[k]).replace('\n', '').replace(', dtype=complex64)', '').replace(',      dtype=complex64)', '').replace('array(', '').replace(']])', ']]') # << format
            with open(save_folder+real_space_fname, 'w') as fw:
                fw.write(str(O_mn_R_W_tosave).replace("\'", ""))

    # (3a) inverse Fourier
    O_mn_k_W = real_to_W_gauge(kpoints, O_mn_R_W)

    # (3b) transformation to Hamiltonian gauge
    # if Hamiltonian is being calculated, perform diagonalization
    if hamiltonian is True:
        Eigs_k, U_mn_k = W_gauge_to_H_gauge(O_mn_k_W, U_mn_k={}, hamiltonian=True)
        return Eigs_k, U_mn_k
    else:
        O_mn_k_H = W_gauge_to_H_gauge(O_mn_k_W, U_mn_k=U_mn_k, hamiltonian=False)
        return O_mn_k_H


def real_to_W_gauge(kpoints, O_mn_R_W):
    """Perform inverse Fourier transformation from real-space operator to k-space in Wannier gauge."""
    O_mn_k_W = {}
    for kpoint in kpoints:
        # Eq. 18 in Qiao 2018
        O_mn_k_W[kpoint] = 0
        # sum over all real-space vectors into which the real-space Hamiltonian was expanded 
        for R_direct, O_mn in O_mn_R_W.items():
            O_mn_k_W[kpoint] += exp(1j*2*np.pi*np.dot(kpoint, R_direct)) * O_mn
    return O_mn_k_W


def W_gauge_to_H_gauge(O_mn_k_W, U_mn_k={}, hamiltonian=True):
    """Transform form Wannier gauge to Hamiltonian gauge, i.e., for every k-point either diagonalize Hamiltonian
        or use the matrix from previous Hamiltonian diagonalization ('U_mn_k') to transform some other operator 
        to H gauge."""
    if hamiltonian is True:
        U_mn_k = {}
        Eigs_k = {}
        for kpoint, O_mn in O_mn_k_W.items():
            # (A) get by reverse multiplication than in step (1): works only for k-points which are in the coarse grid
            # O_mn_k_H[kpoint] = u_dis_dict[kpoint] @ u_dict[kpoint] @ O_mn @ u_dict[kpoint].conj().T @ u_dis_dict[kpoint].conj().T
            # (B)
            w, U_mn = np.linalg.eigh(O_mn)
            Eigs_k[kpoint] = w
            U_mn_k[kpoint] = U_mn
        # return a list of (NW x NW) U_mn matrices for each of the kpoints 
        return Eigs_k, U_mn_k
    # if any other operator than Hamiltonian, you need to transform with the unitary matrices obtained 
    #   during the Hamiltonian diagonalization
    else:
        O_mn_k_H = {}
        for kpoint, O_mn in O_mn_k_W.items():
            O_mn_k_H[kpoint] = U_mn_k[kpoint].conj().T @ O_mn @ U_mn_k[kpoint]
        # return a list of (NW x NW) O_mn matrices for each of the kpoints 
        return O_mn_k_H


def plot_bands_spin_texture(kpoints, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, EF=0, fout='spin_texture_1D_home_made.jpg', fig_caption="home-made interpolation"):
    """Output a figure with Sx, Sy, and Sz-projected band structure."""
    NW = len(Eigs_k[list(Eigs_k.keys())[0]])

    # if spin information missing, plot just bands
    if S_mn_k_H_x == {}:
        fig, ax = plt.subplots(1, 1, figsize=[3,4.5])
        ax.axhline(linestyle='--', color='k')
        ax.set_ylim(yaxis_lim)
        ax.set_xlim([min(kpath), max(kpath)])
        ax.set_ylabel('E - E_F (eV)')
        ax.set_xlabel('k-path (1/A)')
        sc = ax.scatter([[k_dist for i in range(NW)] for k_dist in kpath], [Eigs_k[kpoint] - E_F for kpoint in kpoints],
                        c='b', s=0.2)

    else:
        fig, axes = plt.subplots(1, 3, figsize=[11,4.5])
        spin_name = ['Sx', 'Sy', 'Sz']
        for i, S in enumerate([S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z]):
            ax = axes[i]
            ax.axhline(linestyle='--', color='k')
            ax.set_ylim(yaxis_lim)
            ax.set_xlim([min(kpath), max(kpath)])
            ax.set_title(spin_name[i])
            if i == 0:
                ax.set_ylabel('E - E_F (eV)')
            ax.set_xlabel('k-path (1/A)')
            sc = ax.scatter([[k_dist for i in range(NW)] for k_dist in kpath], [Eigs_k[kpoint] - E_F for kpoint in kpoints],
                            c=[np.diag(S[kpoint]) for kpoint in kpoints], cmap='seismic', s=0.2, vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
            #cbar.set_label(r'$S_\mathrm{z}$')
            #sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    plt.suptitle(fig_caption)
    plt.tight_layout()
    plt.savefig(fout, dpi=400)
    plt.close()
    # plt.show()


def save_bands_and_spin_texture(kpoints_rec, kpoints_cart, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, kmesh_2D=False, 
                                fout='bands_spin_30.pickle', fout2D='bands_spin_2D_30x30.pickle'):
    """Save the bands and spin texture information for given kpoints."""
    bands_spin_dat = {}
    bands_spin_dat['kpoints'] = kpoints_cart
    bands_spin_dat['bands'] = [Eigs_k[kpoint] for kpoint in kpoints_rec]
    bands_spin_dat['Sx'] = [np.diagonal(S_mn_k_H_x[kpoint]).real for kpoint in kpoints_rec]
    bands_spin_dat['Sy'] = [np.diagonal(S_mn_k_H_y[kpoint]).real for kpoint in kpoints_rec]
    bands_spin_dat['Sz'] = [np.diagonal(S_mn_k_H_z[kpoint]).real for kpoint in kpoints_rec]
    if kmesh_2D is not True:
        bands_spin_dat['kpath'] = kpath
    bands_spin_name = fout2D if kmesh_2D is True else fout
    with open(save_folder+bands_spin_name, 'wb') as fw:
        pickle.dump(bands_spin_dat, fw)


def procedure(kmesh_density=100, kmesh_2D=False):
    files_wann90_to_dict_pickle()

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
    A = load_lattice_vectors(win_file="wannier90.win")
    G = reciprocal_lattice_vectors(A)

    kpoint_matrix = [
        [(0.00,  0.50,  0.00),    (0.00,  0.00,  0.00)],
        [(0.00,  0.00,  0.00),    (0.33,  0.33,  0.00)],
        [(0.33,  0.33,  0.00),    (0.00,  0.50,  0.00)]
    ]

    if kmesh_2D is True:
        kpoints_rec, kpoints_cart = get_2D_kpoint_mesh(G, limits=kmesh_2D_limits, Nk=kmesh_density)
        kpath = []
    else:
        kpoints_rec, kpoints_cart, kpath = get_kpoint_path(kpoint_matrix, G, Nk=kmesh_density)
    
    # kpoints = [(0.1176470588,  0.0000000000,  0.0000000000)]
    Eigs_k, U_mn_k = interpolate_operator(eig_dict, u_dis_dict, u_dict, hamiltonian=True, 
                                            latt_params=A, reciprocal_latt_params=G, R_grid=R_grid, kpoints=kpoints_rec, 
                                            save_real_space=save_real_space_operators, 
                                            real_space_fname="hr_R_dict.pickle")
    
    spn_dict_x = {}
    spn_dict_y = {}
    spn_dict_z = {}
    for k, v in spn_dict.items():
        spn_dict_x[k[0]] = spn_dict[(k[0], 'x')]
        spn_dict_y[k[0]] = spn_dict[(k[0], 'y')]
        spn_dict_z[k[0]] = spn_dict[(k[0], 'z')]

    S_mn_k_H_x = interpolate_operator(spn_dict_x, u_dis_dict, u_dict, hamiltonian=False, U_mn_k=U_mn_k, 
                                        latt_params=A, reciprocal_latt_params=G, R_grid=R_grid, kpoints=kpoints_rec,
                                        save_real_space=save_real_space_operators, real_space_fname=spn_x_R_dict_name)
    S_mn_k_H_y = interpolate_operator(spn_dict_y, u_dis_dict, u_dict, hamiltonian=False, U_mn_k=U_mn_k, 
                                        latt_params=A, reciprocal_latt_params=G, R_grid=R_grid, kpoints=kpoints_rec,
                                        save_real_space=save_real_space_operators, real_space_fname=spn_y_R_dict_name)
    S_mn_k_H_z = interpolate_operator(spn_dict_z, u_dis_dict, u_dict, hamiltonian=False, U_mn_k=U_mn_k, 
                                        latt_params=A, reciprocal_latt_params=G, R_grid=R_grid, kpoints=kpoints_rec,
                                        save_real_space=save_real_space_operators, real_space_fname=spn_z_R_dict_name)

    # unite the spn real space dictionaries to one
    with open(save_folder+spn_x_R_dict_name, 'rb') as fxr:
        spn_x_dict = pickle.load(fxr)
    with open(save_folder+spn_y_R_dict_name, 'rb') as fyr:
        spn_y_dict = pickle.load(fyr)
    with open(save_folder+spn_z_R_dict_name, 'rb') as fzr:
        spn_z_dict = pickle.load(fzr)
                
    with open(save_folder+spn_R_dict_name, 'wb') as fw:
        spn_dict = unite_spn_dict(spn_x_dict, spn_y_dict, spn_z_dict, spin_names=['x','y','z'])
        pickle.dump(spn_dict, fw)

    if save_bands_spin_texture is True:
        save_bands_and_spin_texture(kpoints_rec, kpoints_cart, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, kmesh_2D=kmesh_2D)

    if kmesh_2D is not True:
        plot_bands_spin_texture(kpoints_rec, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, fout='spin_texture_1D_home_made.jpg')


def main():
    procedure(kmesh_density=kmesh_density_1D, kmesh_2D=False)
    if run_kmesh_2D is True:
        procedure(kmesh_density=kmesh_density_2D, kmesh_2D=True)


if __name__ == '__main__':
    main()
