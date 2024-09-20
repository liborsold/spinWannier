
# from spin_texture_interpolated import R_grid
import numpy as np
import ast
from math import floor, ceil
import pickle
from itertools import product
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import pandas as pd
import pickle
from os.path import exists
from cmath import exp
from scipy.io import FortranFile


def get_kpoint_names(fwin="wannier90.win"):
    """Parse wannier90.win file to get k-point names as a list of tuples."""
    kpoints = []
    reading = False
    with open(fwin, 'r') as fr:
        for line in fr:
            if reading:
                if "end kpoints" in line:
                    return kpoints
                kpoints.append(tuple([0.0 if coordinate == 0 else float(f"{float(coordinate):.10f}") for coordinate in line.split()]) )
            else:
                if "begin kpoints" in line:
                    reading = True


def load_eigenvals(eigenval_file="wannier90.eig"):
    """Return 2D list (n_kpoints X n_bands) containing the eigenvalues."""
    A = np.loadtxt(eigenval_file)
    NB = max(A[:,0])
    with open(eigenval_file, 'r') as fr:
        Eig_kn = []
        Eig_k = []
        for i, line in enumerate(fr):
            l_split = line.split()
            # if going to a next k-point
            Eig_k.append(float(l_split[2]))
            if i%NB == NB-1:
                Eig_kn.append(Eig_k)
                Eig_k = []
    return Eig_kn


def eigenval_dict(eigenval_file="wannier90.eig",  win_file="wannier90.win"):
    """Return the eigenvalues as a dictionary with the keys being the k-point tuples."""
    eigenvalues = load_eigenvals(eigenval_file=eigenval_file)
    kpoints = get_kpoint_names(fwin=win_file)
    # make k-points with 10 decimal places and no minus sign before a zero
    kpoints = [(float(f"{kpoint[0]:.10f}"), float(f"{kpoint[1]:.10f}"), float(f"{kpoint[2]:.10f}")) for kpoint in kpoints] 
    eigenval_dict = {}
    for kpoint, eigenval_array in zip(kpoints, eigenvalues):
        eigenval_dict[kpoint] = np.diag(eigenval_array)
    return eigenval_dict


def load_dict(fin="spn_dict.pickle", text_file=False):
    """If not text_file, than it's binary.
    fin = spn_dict.pickle / hr_R_dict.pickle / u_dict.pickle / u_dis_dict.pickle
    fin = spin_dict.dat / hr_R_dict.dat / u_dict.dat / u_dis_dict.dat 
    """
    if text_file is True:
        # load dictionary from a human-readable text file
        with open(fin, 'r') as fr:
            spn_dict = ast.literal_eval(fr.read())

        # transform values to numpy arrays
        numpy_dict = {}
        for key, val in spn_dict.items():
            numpy_dict[key] = np.array(val)

    else:
        with open(fin, 'rb') as fr:
            numpy_dict = pickle.load(fr)

    return numpy_dict


def load_lattice_vectors(win_file="wannier90.win"):
    """Return a 3x3 matrix where 1st row is the 1st lattice vector etc."""
    A = []
    with open(win_file, 'r') as fr:
        read = False
        for line in fr:
            if "end unit_cell_cart" in line:
                return np.array(A)
            if read is True:
                A.append([float(item) for item in line.split()])
            if "begin unit_cell_cart" in line:
                read = True


def reciprocal_lattice_vectors(real_space_lattice_vector_matrix):
    A = real_space_lattice_vector_matrix
    unit_cell_volume = abs( np.dot(np.cross(A[0,:], A[1,:]), A[2,:] ) )
    G = []
    for i in range(3):
        G.append( 2*np.pi * np.cross( A[(i+1)%3, :], A[(i+2)%3, :] ) / unit_cell_volume )
    return G


def get_kpoint_path(kpoint_matrix, G, Nk):
    """Interpolate between kpoints in the kpoint_matrix to obtain a k-point path with Nk points in each segment."""
    kpoints_rec = []
    for k0, k1 in kpoint_matrix[:]:
        kpoints_rec += list(zip(    np.linspace(k0[0], k1[0], Nk),
                                np.linspace(k0[1], k1[1], Nk),
                                np.linspace(k0[2], k1[2], Nk) 
                            ))

    kpoints_cart = np.array( [np.array(kpoint) @ G for kpoint in kpoints_rec] )

    # calculate the kpath (k-point distances)
    kpath = np.cumsum( [0.0] + [ np.linalg.norm(kpoints_cart[i+1] - kpoints_cart[i]) for i in range(len(kpoints_rec)-1)] )

    # save to array of tuples
    kpoints_rec = [tuple(kpoint) for kpoint in kpoints_rec]
    kpoints_cart = [tuple(kpoint) for kpoint in kpoints_cart]
    return kpoints_rec, kpoints_cart, kpath


def uniform_real_space_grid(R_mesh_ijk = (5, 5, 1)):
    """From the maxima in each direction, make a regular mesh 
        (-R_max_i, +R_max_i) x (-R_max_j, +R_max_j) x (-R_max_k, +R_max_k)."""
    R_grid = []
    for Ri in range(-floor((R_mesh_ijk[0]-1)/2), ceil((R_mesh_ijk[0]-1)/2) + 1):
        for Rj in range(-floor((R_mesh_ijk[1]-1)/2), ceil((R_mesh_ijk[1]-1)/2) + 1):
            for Rk in range(-floor((R_mesh_ijk[2]-1)/2), ceil((R_mesh_ijk[2]-1)/2) + 1):
                R_grid.append((Ri, Rj, Rk))
    return R_grid


def real_space_grid_from_hr_dat(fname="wannier90_hr.dat"):
    """Get the real-space grid from the seedname_hr.dat file. 
    See Pizzi 2020 Sec. 4.2."""
    with open(fname, 'r') as fr:
        Data = np.loadtxt(fr, skiprows=5)
    R_grid = Data[:, 0:3]
    R_grid = list(set([tuple([int(i) for i in item]) for item in R_grid]))
    R_grid.sort()
    return R_grid


def get_DFT_kgrid(fin="wannier90.win"):
    """Get the 'mp_grid' from wannier90.win file which tells the k-grid used in the DFT calculation."""
    with open(fin, 'r') as fr:
        for line in fr:
            if 'mp_grid' in line:
                lsplit = line.split()
                mp_grid = (int(lsplit[-3]), int(lsplit[-2]), int(lsplit[-1]))
                return mp_grid


def get_2D_kpoint_mesh(G, limits=[-0.5, 0.5], Nk=100):
    """asdf"""
    G = np.array(G)
    # reciprocal vectors in reciprocal coordinates

    g1_rec = np.linspace(*limits, Nk)
    g2_rec = np.linspace(*limits, Nk)
    g3_rec = np.zeros((Nk**2,1))

    kpoints_rec = np.array(np.meshgrid(list(g1_rec), list(g2_rec))).T.reshape(-1, 2)
    kpoints_rec = np.hstack((kpoints_rec, g3_rec))
    kpoints_cart = kpoints_rec @ G

    kpoints_rec = [tuple(kpoint) for kpoint in kpoints_rec]
    kpoints_cart = [tuple(kpoint) for kpoint in kpoints_cart]

    return kpoints_rec, kpoints_cart


def get_skiprows_hr_dat(fin="wannier90_hr.dat"):
    """Get the number of skiprows in wannier90_hr.dat file."""
    with open(fin, 'r') as fr:
        for i, line in enumerate(fr):
            if i == 2:
                n = int(line.split()[0])
                skiprows = n//15 + 4 if n%15 != 0 else  n//15 + 3
                return skiprows


def hr_wann90_to_dict(fin_wannier90 = "wannier90_hr.dat"):
    """Convert wannier90 hr.dat to dictionary in the form that we are using: R vectors as keys 
    and hopping as a complex number matrix as the values."""

    skiprows = get_skiprows_hr_dat(fin=fin_wannier90)
    with open(fin_wannier90, 'r') as fr:
        Data = np.loadtxt(fr, skiprows=skiprows)
    
    # initialize 'hr_dict' dictionary with empty complex matrices
    R = np.unique(np.array(Data[:,0:3], dtype=int), axis=0)
    R = [tuple(R_) for R_ in R]
    hr_dict = {}
    num_wann = int(max(Data[:,3]))
    for R_ in R:
        hr_dict[R_] = np.zeros((num_wann, num_wann), dtype=np.complex64)

    # parse the 'Data'
    for row in Data:
        hr_dict[(int(row[0]), int(row[1]), int(row[2]))][int(row[3]-1), int(row[4]-1)] = float(row[5]) + float(row[6])*1j
    
    return hr_dict


def coerce_R_vectors_to_basic_supercell(R_tr=(-3,2,0), R_mesh_ijk=(5,5,1)):
    """wannier90_hr.dat has sometimes the R vectors shifted (probably to facilitate the 
    'minimal replice' interpolation. Therefore, given some uniform grid centered at zero
    (e.g. 5x5x1) and a translated R vectors from hr_dat file (e.g. (-3,2,0) ), we would like to get 
    the R vector's image in the uniform grid, to be able to compare with our results.
    For (-3,2,0) this would be (2,2,0) in case of (5,5,1) grid  (adding 5 to the 'x' coordinate).
    -> Algorithm: translate the 'R' by a supercell vector (in negative and positive direction and 
    for x, y, z separately -> 8 possible translations: so searching the immediate vicinity of the R vector.
    Return a dictionary from the 'hr_dat grid' to the uniform grid."""

    R_grid_target = uniform_real_space_grid(R_mesh_ijk=(5,5,1))
    if R_tr in R_grid_target: 
        return R_tr
    for T in product([-R_mesh_ijk[0], 0, R_mesh_ijk[0]], [-R_mesh_ijk[1], 0, R_mesh_ijk[1]], [-R_mesh_ijk[2], 0, R_mesh_ijk[2]]):
        if tuple(np.array(R_tr) + np.array(T)) in R_grid_target:
            return tuple(np.array(R_tr) + np.array(T))


def dict_to_matrix(data_dict, num_wann, spin_index=False):
    """Convert hr or spn dictionary to a matrix accepted by wannierBerri: first index is m, second n, third is the index of an R_vector from iRvec list."""
    S_names = ['x', 'y', 'z']
    if spin_index is True:
        # pick just the Rvec from the composed (Rvec, spin) keys
        iRvec_list = [key[0] for key in list(data_dict.keys()) if key[1]==S_names[0]]
        data_matrix = np.array([[[[data_dict[tuple([iRvec_list[iR], S])][m,n] for S in S_names] for iR in range(len(iRvec_list))] for n in range(num_wann)] for m in range(num_wann)])
    else:
        iRvec_list = list(data_dict.keys())
        data_matrix = np.array([[[data_dict[iRvec_list[iR]][m,n] for iR in range(len(iRvec_list))] for n in range(num_wann)] for m in range(num_wann)])
    iRvec = np.empty(len(iRvec_list), dtype=object)
    iRvec[:] = iRvec_list
    return np.array(iRvec), data_matrix


def matrix_to_dict(data_matrix, Rvecs, spin_index=False, spin_names=['x','y','z']):
    """Convert data matrix (e.g. the Ham_R or SS_R file from symmetrization) into a dictionary with 
        lattice vectors as keys and if spin_index is True then also ('x', 'y', 'z') for spin."""
    
    data_matrix = np.array(data_matrix)
    NR = len(Rvecs)

    if not data_matrix.shape[2] == NR:
        raise Exception("The data matrix dimension does not correspond to the length of the lattice vector array.")
    if not data_matrix.shape[0] == data_matrix.shape[1]:
        raise Exception("The data at each R (and spin) must be a square matrix.")

    data_dict = {}
    if spin_index is True:
        for i, R in enumerate(Rvecs):
            for s in range(3):
                data_dict[tuple([tuple(R), spin_names[s]])] = data_matrix[:,:,i,s]
    else:
        for i, R in enumerate(Rvecs):
            data_dict[tuple(R)] = data_matrix[:,:,i]

    return data_dict


def split_spn_dict(spn_dict, spin_names=['x','y','z']):
    """Split dictionary with keys as ((Rx, Ry, Rz), spin_component_name) to three dictionaries
            with keys as (Rx, Ry, Rz)."""
    R_keys = list(set([key[0] for key in spn_dict.keys()]))
    spn_x_dict = {}
    spn_y_dict = {}
    spn_z_dict = {}
    for key in R_keys:
        spn_x_dict[key] = spn_dict[tuple([key, spin_names[0]])]
        spn_y_dict[key] = spn_dict[tuple([key, spin_names[1]])]
        spn_z_dict[key] = spn_dict[tuple([key, spin_names[2]])]
    return spn_x_dict, spn_y_dict, spn_z_dict


def unite_spn_dict(spn_x_dict, spn_y_dict, spn_z_dict, spin_names=['x','y','z']):
    """Unite dictionary with keys as (Rx, Ry, Rz) to three dictionaries
            with keys as ((Rx, Ry, Rz), spin_component_name)."""
    keys_no_spin = list(spn_x_dict.keys())
    spn_dict = {}
    for key in keys_no_spin:
        spn_dict[tuple([key, spin_names[0]])] = spn_x_dict[key]
        spn_dict[tuple([key, spin_names[1]])] = spn_y_dict[key]
        spn_dict[tuple([key, spin_names[2]])] = spn_z_dict[key]
    return spn_dict


def check_file_exists(file_name):
    """Check if file exists. If yes, add him a number in parentheses that does not exist. 
        Return the new name."""
    # Get the file name and extension
    base_name, file_extension = os.path.splitext(file_name)
    # Initialize a counter for appending to the file name
    counter = 1
    # Check if the file already exists
    while os.path.exists(file_name):
        # If it does, append a number in parentheses to the file name
        file_name = f'{base_name}({counter}){file_extension}'
        counter += 1
    return file_name


def outer(s, M):
    """Outer product between a 2x2 spin matrix and a general orbital nxn M matrix, 
    so that the spin blocks are the big blocks.

    Args:
        s (np.array): Pauli (2 x 2) matrix sx, sy or sz 
        M (np.array): general orbital (n x n) matrix
    """
    n = len(M)
    A = np.zeros((2*n, 2*n), dtype=np.complex64)
    A[:n, :n] = s[0,0]*M
    A[:n, n:] = s[0,1]*M
    A[n:, :n] = s[1,0]*M
    A[n:, n:] = s[1,1]*M
    return A


def outer2(s, M):
    """Works for s matrix of any dimension, but slower than 'outer(s, M)'
    Outer product between a nxn spin matrix and a general orbital nxn M matrix, 
    so that the spin blocks are the big blocks.

    Args:
        s (np.array): Pauli (2 x 2) matrix sx, sy or sz 
        M (np.array): general orbital (n x n) matrix
    """
    return np.hstack(np.hstack(np.multiply.outer(s, M)))


def wannier_energy_windows(wann_bands_lims, eigenval_file="wannier90.eig"):
    """Get energy window limits for wannierization.

    Args:
        wann_bands_lims (tuple of two int): Zero-indexed indices of the minimum and maximum bands included in wannierization. E.g. (0, 21) if the bottom 22 bands should be wannierized.
        E_F (float, optional): The Fermi level in eV. Defaults to 0.0.
        eigenval_file (str, optional): Wannier90 .eig file name. Defaults to "wannier90.eig".

    Returns:
        4 floats: the wannierization and frozen energy minima and maxima.
    """
    wann_min_band = wann_bands_lims[0]
    wann_max_band = wann_bands_lims[1]

    Eig_kn = np.array(load_eigenvals(eigenval_file))
    Eig_kn_wann = Eig_kn[:, wann_min_band:wann_max_band]

    bands_below_wann = wann_min_band != 0
    bands_above_wann = wann_max_band != Eig_kn.shape[0]-1

    # wannierization window: energy range spanned by the bands within wann_bands_lims
    dis_win_min = np.min(Eig_kn)
    dis_win_max = np.max(Eig_kn)

    # frozen energy window: energy range spanned by the bands within wann_bands_lims WHERE no bands from outside 'wann_bands_lims' exist

    dis_froz_min = max(np.min(Eig_kn_wann), np.max(Eig_kn[:, :wann_min_band])) if bands_below_wann else np.min(Eig_kn)
    dis_froz_max = min(np.max(Eig_kn_wann), np.min(Eig_kn[:, wann_max_band+1:])) if bands_above_wann else np.max(Eig_kn)

    return dis_win_min, dis_win_max, dis_froz_min, dis_froz_max
    
   
def S_xyz_expectation_values(eigvecs):
    """For a k-space Hamiltonian Hk (function of kx and ky), calculate the eigen-energies and expectation values of spin Sx, Sy, Sz, at all the k-points in 'kpoints'.

    Args:
        Hk (function): A function returning a 2x2 FEG Hamiltonian.
        kpoints (array of tuples): Array of k-points.

    Returns:
        four arrays: array of expectation values for energy and spin expectation values
    """
    sigma_x  = np.array([[0, 1], [1, 0]])
    sigma_y  = np.array([[0, -1j], [1j, 0]])
    sigma_z  = np.array([[1, 0], [0, -1]])
    Energies = []
    S_x = []
    S_y = []
    S_z = []
    for (kx, ky, kz) in kpoints:
        # print( np.cos(kx*a1[0]+ky*a1[1]) )
        # print( np.cos(kx*(a1[0]-a2[0])+ky*(a1[1]-a2[1]))  )

        H_k = Hk(kx, ky)
        E, eigvecs = np.linalg.eigh(H_k)
        Energies.append(np.real(E))
        S_x.append(np.real([[np.conj(eigvecs[j,:,i]).T @ sigma_x @ eigvecs[j,:,i] for i in range(eigvecs.shape[1])] for j in range(len(eigvecs))]))
        S_y.append(np.real([np.conj(eigvecs[:,i]).T @ sigma_y @ eigvecs[:,i] for i in range(eigvecs.shape[1])]))
        S_z.append(np.real([np.conj(eigvecs[:,i]).T @ sigma_z @ eigvecs[:,i] for i in range(eigvecs.shape[1])]))
    # convert to numpy arrays
    Energies = np.array(Energies)
    S_x = np.array(S_x)
    S_y = np.array(S_y)
    S_z = np.array(S_z)
    # SORT
    idx_ascend = np.argsort(Energies, axis=1)
    Energies = np.take_along_axis(Energies, idx_ascend, axis=1)
    S_x = np.take_along_axis(S_x, idx_ascend, axis=1)
    S_y = np.take_along_axis(S_y, idx_ascend, axis=1)
    S_z = np.take_along_axis(S_z, idx_ascend, axis=1)
    return Energies, S_x, S_y, S_z


def operator_exp_values(eigvecs, Operator):
    """Return the expectation values of Operator (an N x N matrix, where N is the size of the eigenvectors) for all eigenvectors.

    Args:
        eigvecs (array of matrices): Array of M square matrices. Each N x N square matrix contains N eigenvectors (columns of the matrix).  
        Operator (matrix): N x N Hermition matrix acting on the eigenvectors. 

    Returns:
        M x N array of expectation values. 
    """
    O_exp_values = np.real([[np.conj(eigvecs[j,:,i]).T @ Operator @ eigvecs[j,:,i] for i in range(eigvecs.shape[2])] for j in range(len(eigvecs))])
    return O_exp_values


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
    """Transform from Wannier gauge to Hamiltonian gauge, i.e., for every k-point either diagonalize Hamiltonian
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
        
        
def save_bands_and_spin_texture_old(kpoints_rec, kpoints_cart, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, kmesh_2D=False, 
                                fout='bands_spin.pickle', save_folder='./tb_model_wann90/'):
    """Save the bands and spin texture information for given kpoints."""
    bands_spin_dat = {}
    bands_spin_dat['kpoints'] = kpoints_cart
    bands_spin_dat['bands'] = [Eigs_k[kpoint] for kpoint in kpoints_rec]
    bands_spin_dat['Sx'] = [np.diagonal(S_mn_k_H_x[kpoint]).real for kpoint in kpoints_rec]
    bands_spin_dat['Sy'] = [np.diagonal(S_mn_k_H_y[kpoint]).real for kpoint in kpoints_rec]
    bands_spin_dat['Sz'] = [np.diagonal(S_mn_k_H_z[kpoint]).real for kpoint in kpoints_rec]
    if kmesh_2D is not True:
        bands_spin_dat['kpath'] = kpath
    print(save_folder+fout)
    with open(save_folder+fout, 'wb') as fw:
        pickle.dump(bands_spin_dat, fw)


def save_bands_and_spin_texture(kpoints_rec, kpoints_cart, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, kmesh_2D=False, 
                                fout='bands_spin.pickle', save_folder='./tb_model_wann90/'):
    """Save the bands and spin texture information for given kpoints."""
    bands_spin_dat = {}
    bands_spin_dat['kpoints'] = kpoints_cart
    bands_spin_dat['bands'] = Eigs_k
    bands_spin_dat['Sx'] = S_mn_k_H_x
    bands_spin_dat['Sy'] = S_mn_k_H_y
    bands_spin_dat['Sz'] = S_mn_k_H_z
    if kmesh_2D is not True:
        bands_spin_dat['kpath'] = kpath
    print(save_folder+fout)
    with open(save_folder+fout, 'wb') as fw:
        pickle.dump(bands_spin_dat, fw)


def plot_bands_spin_texture(kpoints, kpath, kpath_ticks, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, E_F=0, fout='spin_texture_1D_home_made.jpg', fig_caption="Wannier interpolation", yaxis_lim=[-5,5]):
    """Output a figure with Sx, Sy, and Sz-projected band structure."""
    NW = len(Eigs_k[list(Eigs_k.keys())[0]])
    Nk = len(kpoints)//(len(kpath_ticks)-1)

    # if spin information missing, plot just bands
    if S_mn_k_H_x == {}:
        fig, ax = plt.subplots(1, 1, figsize=[3,4.5])
        ax.axhline(linestyle='--', color='k')
        ax.set_ylim(yaxis_lim)
        ax.set_xlim([min(kpath), max(kpath)])
        ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)', fontsize=13)
        sc = ax.scatter([[k_dist for i in range(NW)] for k_dist in kpath], [Eigs_k[kpoint] - E_F for kpoint in kpoints],
                        c='b', s=0.2)

    else:
        fig, axes = plt.subplots(1, 3, figsize=[11,4.5])
        spin_name = [r'$S_x$', r'$S_y$', r'$S_z$']
        for i, S in enumerate([S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z]):
            ax = axes[i]
            ax.axhline(linestyle='--', color='k')
            ax.set_ylim(yaxis_lim)
            ax.set_xlim([min(kpath), max(kpath)])
            # give title margin on bottom
            ax.set_title(spin_name[i], fontsize=14, pad=10)
            if i == 0:
                ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)', fontsize=13)
            secax = ax.secondary_xaxis('top')
            secax.tick_params(labelsize=9) #axis='both', which='major', 
            secax.set_xlabel(r"$k$-distance (1/$\mathrm{\AA}$)", fontsize=10)
            sc = ax.scatter([[k_dist for i in range(NW)] for k_dist in kpath], [Eigs_k[kpoint] - E_F for kpoint in kpoints],
                            c=[np.diag(S[kpoint]) for kpoint in kpoints], cmap='coolwarm', s=0.2, vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sc, cax=cax, orientation='vertical')

            idx = np.array(range(0, Nk*len(kpath_ticks), Nk))
            idx[-1] += -1
            ax.set_xticks(kpath[idx])
            ax.set_xticklabels(kpath_ticks, fontsize=12)
            ax.yaxis.set_tick_params(labelsize=11)
            for i in range(1, len(kpath_ticks)-1):
                ax.axvline(x=kpath[i*Nk], color='#000000', linestyle='-', linewidth=0.75)
            #cbar.set_label(r'$S_\mathrm{z}$')
            #sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    plt.suptitle(fig_caption)
    plt.tight_layout()
    # plt.show()
    fout = check_file_exists(fout)
    plt.savefig(fout, dpi=400)
    plt.close()
    # plt.show()


def fermi_surface_spin_texture(kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, E=0, E_F=0, E_thr=0.01, fig_name=None, quiver_scale=1, \
                                scatter_for_quiver=True, scatter_size_quiver=1, scatter_size=0.8, reduce_by_factor=1, \
                                    kmesh_limits=None, savefig=True, colorbar_Sx_lim=[-1,1], colorbar_Sy_lim=[-1,1], colorbar_Sz_lim=[-1,1]):
    """Plot scatter points with a spin texture on a constant energy xy surface (probably at E=EF)
        if the energy difference of each given point is lower than some threshold.
        If there is more such points, grab the one with minimum difference from the energy surface.

        - E and E_thr in eV"""

    energy_distances = np.abs(bands2D - E)

    include_kpoint = np.any(energy_distances <= E_thr, axis=1)
    closest_band = np.argmin(energy_distances, axis=1)[include_kpoint]

    # plot the result
    kx = kpoints2D[include_kpoint,0]
    ky = kpoints2D[include_kpoint,1]

    # ARROWS Sz
    fig, ax = plt.subplots(1, 1, figsize=[5,5])
    if scatter_for_quiver is True:
        ax.scatter(kx, ky, s=scatter_size_quiver, c='k')

    sc = ax.quiver(kx[::reduce_by_factor], ky[::reduce_by_factor], Sx2D[include_kpoint,closest_band][::reduce_by_factor],
                    Sy2D[include_kpoint,closest_band][::reduce_by_factor], Sz2D[include_kpoint,closest_band][::reduce_by_factor],
                    scale=quiver_scale, cmap='seismic')
    ax.set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)')
    ax.set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)')
    ax.set_title(f"{os.getcwd()}\npyplot.quiver scale={quiver_scale}", fontsize=6)
    ax.set_xlim(kmesh_limits)
    ax.set_ylim(kmesh_limits)
    ax.set_aspect('equal')
    ax.set_facecolor("#777777")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label(r'$S_\mathrm{z}$')
    if colorbar_Sz_lim:
        sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])
    plt.tight_layout()
    if fig_name is None:
        fig_name_all_one = f"spin_texture_2D_all_in_one_E-from-EF{E-E_F:.3f}eV.jpg"
    else:
        fig_name_split = fig_name.split('.')
        fig_name_all_one = f"{''.join(fig_name_split[:-1])}_all_in_one_E-from-EF{E-E_F:.3f}eV.{fig_name_split[-1]}"
    fig_name_all_one = check_file_exists(fig_name_all_one)

    if savefig is True:
        plt.savefig(fig_name_all_one, dpi=400)
        # plt.show()
        plt.close()


    fig2, axes = plt.subplots(1, 3, figsize=[12,4])

    sc1 = axes[0].scatter(kx, ky, c=Sx2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig2.colorbar(sc1, cax=cax, orientation='vertical')
    cbar1.set_label(r'$S_\mathrm{x}$')
    axes[0].set_title(r'$S_\mathrm{x}$')
    if colorbar_Sx_lim:
        sc1.set_clim(vmin=colorbar_Sx_lim[0], vmax=colorbar_Sx_lim[1])

    sc2 = axes[1].scatter(kx, ky, c=Sy2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig2.colorbar(sc2, cax=cax, orientation='vertical')
    cbar2.set_label(r'$S_\mathrm{y}$')
    axes[1].set_title(r'$S_\mathrm{y}$')
    if colorbar_Sy_lim:
        sc2.set_clim(vmin=colorbar_Sy_lim[0], vmax=colorbar_Sy_lim[1])

    sc3 = axes[2].scatter(kx, ky, c=Sz2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig2.colorbar(sc3, cax=cax, orientation='vertical')
    cbar3.set_label(r'$S_\mathrm{z}$')
    axes[2].set_title(r'$S_\mathrm{z}$')
    if colorbar_Sz_lim:
        sc3.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    for i in range(3):
        axes[i].set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)')
        axes[i].set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)')
        axes[i].set_aspect('equal')
        if kmesh_limits:
            axes[i].set_xlim(kmesh_limits)
            axes[i].set_ylim(kmesh_limits)

    plt.suptitle(f"{os.getcwd()}", fontsize=6)
    plt.tight_layout()
    if fig_name is None:
        fig_name_Sxyz = f"spin_texture_2D_energy_cut_E-from-EF{E-E_F:.3f}eV.jpg"
    else:
        fig_name_split = fig_name.split('.')
        fig_name_Sxyz = f"{''.join(fig_name_split[:-1])}_2D_energy_cut_E-from-EF{E-E_F:.3f}eV.{fig_name_split[-1]}"
    fig_name_Sxyz = check_file_exists(fig_name_Sxyz)

    if savefig is True:
        plt.savefig(fig_name_Sxyz, dpi=400)
        # plt.show()
        plt.close()
    else:
        return fig, fig2


def convert_wann_TB_model_for_QuT(folder_in='./', folder_out='./', seedname_out='system', winfile='wannier90.win', wann_centres_file='wannier90_centres.xyz', hr_R_file='hr_R_dict.pickle', spn_R_file='spn_R_dict.pickle'):
    """Convert the outputs of my wannier to TB to Joaquin's code.
        Generates files 
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}.uc        ....   unit cell
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}.xyz       ....   wannier centers in cartesian coordinates
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}_hr.dat    ....   real-space Hamiltonian
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}_sxhr.dat  ....   real-space spin S_x operator
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}_syhr.dat  ....   real-space spin S_y operator
            {system_name}_Mx{Mx:d}My{My:d}Mz{Mz:d}_E{E:.2f}_szhr.dat  ....   real-space spin S_z operator

    Args:
        winfile (str, optional): _description_. Defaults to 'wannier90.win'.
        hr_R_file (str, optional): _description_. Defaults to 'hr_R_dict.pickle'.
        spn_R_file (str, optional): _description_. Defaults to 'spn_R_dict.pickle'.
    """

    comment = f"# generated from files in directory {folder_in}"

    # ----------------------------------------------------
    # .uc file
    # ----------------------------------------------------
    with open(folder_in + winfile) as fr:
        uc_string = re.search('begin unit_cell_cart(.*)end unit_cell_cart', fr.read(), re.DOTALL).group(1)
    with open(folder_out + seedname_out + '.uc', 'w') as fw:
        fw.write(comment + uc_string)

    # ----------------------------------------------------
    # .xyz file
    # ----------------------------------------------------
    file_path = folder_in + wann_centres_file
    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1)
    selected_data = data[data.iloc[:,0] == 'X']
    selected_data.drop(selected_data.columns[[0]], axis=1, inplace=True)
    selected_data.insert(0, 'Row', range(1, len(selected_data) + 1))
    output_file_path = folder_out + seedname_out + '.xyz'
    total_rows = len(selected_data)
    header = f'{comment}\n{total_rows}\n'
    selected_data.to_csv(output_file_path, sep='\t', header=False, index=False)
    # Add the header to the beginning of the file
    try:
        with open(output_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        output_file_path = folder_out + 'tb_model_wann90/' + seedname_out + '.xyz'
        with open(output_file_path, 'r') as f:
            content = f.read()

    with open(output_file_path, 'w') as f:
        f.write(header + content)

    # ----------------------------------------------------
    # _hr.dat file
    # ----------------------------------------------------
    output_file_path = folder_in + hr_R_file
    try:
        with open(output_file_path, 'rb') as fr:
            hr_dict = pickle.load(fr)
    except FileNotFoundError:
        output_file_path = folder_in + 'tb_model_wann90/' + hr_R_file
        with open(output_file_path, 'rb') as fr:
            hr_dict = pickle.load(fr)
    N_wann = len(hr_dict[list(hr_dict.keys())[0]])
    hr_string_out = f"\t\t{N_wann}\n\t1\n\t1\n"
    for key in list(hr_dict.keys()):
        for i in range(N_wann):
            for j in range(N_wann):
                hr_string_out += f"{key[0]} {key[1]} {key[2]} {i+1:d} {j+1:d} {np.real(hr_dict[key][i][j]):.16f} {np.imag(hr_dict[key][i][j]):.16f}\n"
    with open(folder_out + seedname_out + '_hr.dat', 'w') as fw:
        fw.write(f"{comment}\n{hr_string_out}")

    # ----------------------------------------------------
    # s{x,y,z}hr.dat files
    # ----------------------------------------------------
    output_file_path = folder_in + spn_R_file
    try:
        with open(output_file_path, 'rb') as fr:
            spn_R_dict = pickle.load(fr)
    except FileNotFoundError:
        output_file_path = folder_in + 'tb_model_wann90/' + spn_R_file
        with open(output_file_path, 'rb') as fr:
            spn_R_dict = pickle.load(fr)

    for s in ['x', 'y', 'z']:
        spn_string_out = f"\t\t{N_wann}\n\t1\n\t1\n"
        for key in list(hr_dict.keys()):
            for i in range(N_wann):
                for j in range(N_wann):
                    spn_string_out += f"{key[0]} {key[1]} {key[2]} {i+1:d} {j+1:d} {np.real(spn_R_dict[(key, s)][i][j]):.16f} {np.imag(spn_R_dict[(key, s)][i][j]):.16f}\n"
        with open(folder_out + seedname_out + f'_s{s}hr.dat', 'w') as fw:
            fw.write(f"{comment}\n{spn_string_out}")


def material_name_to_latex(material):
    # each integer i in 'material' will be replaced by $_i$
    material = material.replace('2', '$_2$')
    material = material.replace('3', '$_3$')
    material = material.replace('4', '$_4$')
    material = material.replace('5', '$_5$')
    material = material.replace('6', '$_6$')
    material = material.replace('7', '$_7$')
    return material


def magmom_OOP_or_IP(INCAR_path):
    """Return True if MAGMOM in INCAR_path lies purely OOP, else return False."""
    with open(INCAR_path, 'r') as fin:
        for line in fin:
            if 'MAGMOM' in line:
                lsplit = line.split()
                if float(lsplit[2]) == 0. and float(lsplit[3]) == 0.:
                    return True
                else:
                    return False
                

def magmom_direction(INCAR_path):
    """Return 0 / 1 / 2  if spin is along x / y / z direction, respectively."""
    with open(INCAR_path, 'r') as fin:
        for line in fin:
            if 'MAGMOM' in line:
                MAGMOM = np.array( [float(i) for i in line.split()[2:5]] )
                return np.argmax(np.abs(MAGMOM))
                

def eigenval_for_kpoint(kpoint=(0.0, 0.0, 0.0), band=-4, eigenval_file="wannier90.eig",  win_file="wannier90.win"):
    """Return the eigenvalue at 'kpoint' and 'band'.
         !! kpoint must be one from the kpoints in the wannier90.win  file !!"""
    eigenvalues = load_eigenvals(eigenval_file=eigenval_file)
    kpoints = get_kpoint_names(fwin=win_file)
    # convert the desired kpoint to the same format as the kpoints
    kpoint = (float(f"{kpoint[0]:.10f}"), float(f"{kpoint[1]:.10f}"), float(f"{kpoint[2]:.10f}"))
    # find its index at kpoints
    kpoint_idx = kpoints.index(kpoint)
    return eigenvalues[kpoint_idx][band]


def read_PROCAR_lines_for_kpoint(kpoint=(0.0, 0.0, 0.0), PROCAR_file="PROCAR"):
    """ Get the section of PROCAR belonging to the first occurence of 'kpoint' in PROCAR_file.
    """
    kpoint_string = ' '.join([f"{kpoint[0]:.8f}", f"{kpoint[1]:.8f}", f"{kpoint[2]:.8f}"])
    with open(PROCAR_file) as fr:
        lines = []
        read_lines = False
        for line in fr:
            if read_lines is True:
                # break if you reach next k-point
                if 'k-point' in line:
                    return lines
                lines.append(line)
            if 'k-point' in line and kpoint_string in line:
                read_lines = True


def band_with_spin_projection_under_threshold_for_kpoint(kpoint=(0.0, 0.0, 0.0), spin_direction=2, threshold=0.250, PROCAR_file="PROCAR", n_ions=3, skip_lowest_bands=10):

    lines = read_PROCAR_lines_for_kpoint(kpoint, PROCAR_file)

    # get the band where the 'spin_direction' spin projection is below the threshold
    # select the relevant spin projection from file
    lines_per_kpoint = 3 + 5*(n_ions+1) + 2
    relevant_line = 3 + (2+spin_direction)*(n_ions+1)
    n_bands = int(len(lines)/lines_per_kpoint)

    projections = []
    for i in range(n_bands):
        # get the spin projection
        spin_proj = lines[lines_per_kpoint*i + relevant_line].split()[-1]
        try:
            spin_proj = float(spin_proj)
        except ValueError:
            spin_proj = 999.0
        projections.append(spin_proj)
    projections = np.abs(projections)

    # find the lowest band with the spin projection below the threshold
    band_spin_under_threshold = np.array(projections < threshold)

    # get the first True in 'band_spin_under_threshold'
    lowest_band_under_threshold = np.argmax(band_spin_under_threshold[skip_lowest_bands:])

    return lowest_band_under_threshold


def band_with_spin_projection_under_threshold_for_kpoint(kpoint=(0.0, 0.0, 0.0), orbital_characters_considered={0:[4,5,6,7,8], 1:[1,2,3], 2:[1,2,3]}, threshold=0.250, PROCAR_file="PROCAR", n_ions=3, skip_lowest_bands=10):

    lines = read_PROCAR_lines_for_kpoint(kpoint, PROCAR_file)

    lines_per_band = 3 + 5*(n_ions+1) + 2
    n_bands = int(len(lines)/lines_per_band)
    projections = []
    for i in range(n_bands):
        # get the sum of the considered characters
        orb_proj = 0.0
        for ion, orbital_types in orbital_characters_considered.items():
            lsplit = lines[i*lines_per_band + 4+ion].split()
            for orbital_type in orbital_types:
                orb_proj += float(lsplit[orbital_type+1])
        projections.append(orb_proj)
    projections = np.array(projections)
    print('projections', repr(projections)) #'[' + ]', '.join([str(i) for i in projections])
    # find the lowest band with the spin projection below the threshold
    band_character_under_threshold = np.array(projections < threshold)

    # get the first True in 'band_spin_under_threshold'
    lowest_band_under_threshold = np.argmax(band_character_under_threshold[skip_lowest_bands:])

    return lowest_band_under_threshold


def spn_to_dict(model_dir='./', fwin="wannier90.win", fin="wannier90.spn", formatted=False, fout="spn_dict.pickle", save_as_text=False):
    """Convert wannier90.spn file to a dictionary object and save as pickle. 
    If 'text_file' == True, then save as a human-readable text file."""

    if formatted is True and not fin.split('.')[-1].endswith('formatted'):
        raise Warning(f"Are you sure {fin} is a formatted (human-readable) file and not a binary FortranFile?")
    if formatted is False and fin.split('.')[-1].endswith('formatted') == 'formatted':
        raise Warning(f"Are you sure {fin} is a binary FortranFile and not a human-readable file?")

    fwin = model_dir + fwin
    fin = model_dir + fin
    fout = model_dir + fout

    spin_names = ['x', 'y', 'z']

    if formatted is True:
        # human-readable spn file ("wannier90.spn_formatted")
        with open(fin, 'r') as fr:
            fr.readline()
            # get number of bands and kpoints
            NB = int(float(fr.readline()))
            NK = int(float(fr.readline()))
        with open(fin, 'r') as fr:
            # get the spin-projection matrices
            Sskmn = np.loadtxt(fr, dtype=np.complex64, skiprows=3)
    else:
        # FortranFile spn file ("wannier90.spn")
        with FortranFile(fin, 'r') as fr:
            header = fr.read_record(dtype='c')
            NB, NK = fr.read_record(dtype=np.int32)
            Sskmn = []
            for i in range(NK):
                Sskmn.append(fr.read_record(dtype=np.complex128))
            Sskmn = np.array(Sskmn)

    # get names of k-points
    kpoint_names = get_kpoint_names(fwin=fwin)

    # construct dictionary
    Sskmn_dict = {}

    for ik, kpoint_name in enumerate(kpoint_names):
        Smn = np.zeros((3, NB, NB), dtype=np.complex64)
        # unflatten the upper-diagonal matrix and add hermitian conjugates
        for n in range(NB):
            for m in range(n+1):
                for s in range(3):
                    
                    Smn[s,m,n] = Sskmn[ik, int(3*(n*(n+1)/2 + m) + s)]
                    # hermitian conjugate
                    Smn[s,n,m] = np.conj( Smn[s,m,n] )
        
        # save Smn to dictionary
        for s in range(3):
            Sskmn_dict[(kpoint_name, spin_names[s])] = Smn[s,:,:]

    # write to file

    #  <not implemented yet>  if binary is True:
    #     f = h5py.File('test.hdf5', 'w')
    #     grp = f.create_group('.spn file')
    #     for kpoint in Sskmn_dict:
    #         dset = grp.create_dataset(kpoint, data = Sskmn_dict[kpoint])
    #         #print(grp_name, dset_name, data_dict[grp_name][dset_name])
    #     f.close()
    # else:

    if save_as_text is True:
        
        for kpoint, Smn in Sskmn_dict.items(): 
            Sskmn_dict[kpoint] = repr(Smn[s,:,:]).replace('\n', '').replace(', dtype=complex64)', '').replace('array(', '')

        with open(fout, 'w') as fw:
            fw.write(str(Sskmn_dict).replace("\'", "").replace("x", "\'x\'").replace("y", "\'y\'").replace("z", "\'z\'"))

    else:
        with open(fout, 'ab') as fw:
            pickle.dump(Sskmn_dict, fw)
    
    return Sskmn


def u_to_dict(fin="wannier90_u.mat", fout="u_dict.pickle", text_file=False, write_sparse=False):
    """Convert _u.mat from wannier90 to pickled python dictionary."""
    with open(fin, 'r') as fr:
        fr.readline()
        fr.readline()
        a = fr.readline()

    # get number of bands and kpoints
    with open(fin, 'r') as fr:
        fr.readline()
        l = fr.readline().split()
        n_k = int(l[0])
        num_wann = int(l[1])
        num_bands = int(l[2])

    # parse the file
    read = False
    kpoint = None
    Umnk = {}
    with open(fin, 'r') as fr:
        for j, line in enumerate(fr):
            if read:
            # .. skips the header
                if kpoint:
                # if kpoint is not None, save data-point to matrix
                    l = line.split()
                    Umn[i%num_bands, i//num_bands] = float(l[0]) + float(l[1]) * 1j
                    i += 1
                else:
                # it's the k-point line
                    # save kpoint
                    kpoint = tuple(float(f"{float(item):.12f}") for item in line.split())
                    i = 0
            if j % (num_wann*num_bands+2) == 1:
                if read:
                # .. basically just skips the first empty line
                    Umnk[kpoint] = Umn
                read = True
                Umn = np.zeros((num_bands, num_wann), dtype=np.complex64)
                kpoint = None

    # get the spin-projection matrices
    if text_file is True:
        Umnk_to_write = {}
        for kpoint, Umn in Umnk.items(): 
            Umnk_to_write[kpoint] = repr(Umn).replace('\n', '').replace(', dtype=complex64)', '').replace(',      dtype=complex64)', '').replace('array(', '') # << format
        with open(fout, 'w') as fw:
            fw.write(str(Umnk).replace("\'", ""))
    else:
        with open(fout, 'wb') as fw:
            pickle.dump(Umnk, fw)

    print(f"{fout} written!")
    print("Umnk keys len", len(Umnk.keys()))

    return num_bands, num_wann

    # if write_sparse is True:
    #     with open(fin, 'r') as fr:
    #         u_dict = ast.literal_eval(fr.read())
    #     write_nonzero_elements(u_dict, fout=f"{fin.split('.')[0]}_sparse.dat")


def files_wann90_to_dict_pickle(model_dir='./', disentanglement=False):
    """Convert wannier90 files to pickled dictionaries files."""
    spn_to_dict(model_dir=model_dir)
    u_to_dict(fin=model_dir+"wannier90_u.mat", fout=model_dir+"u_dict.pickle", text_file=False, write_sparse=False)
    if disentanglement is True:
        u_to_dict(fin=model_dir+"wannier90_u_dis.mat", fout=model_dir+"u_dis_dict.pickle", text_file=False, write_sparse=False)


def selected_band_plot(band=0):
    # -------------- SELECTED BAND plot ------------------

    fig, axes = plt.subplots(1, 3, figsize=[14,4])
    spin_name = ['Sx', 'Sy', 'Sz']

    ax = axes[0]
    ax.axhline(linestyle='--', color='k')
    ax.set_xlim([min(kpath1D), max(kpath1D)])
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)', fontsize=12)
    ax.set_xlabel('k-path (1/$\mathrm{\AA}$)', fontsize=12)
    ax.scatter(np.array([kpath1D for i in range(len(bands1D[0,:]))]).T, bands1D-E_F, c='grey', s=2)
    sc = ax.scatter(kpath1D, bands1D[:,band-1]-E_F, c=Sz1D[:,band-1], cmap='seismic', s=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label(r'$S_\mathrm{z}$')
    sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])
    ax.set_ylim(bands_ylim)
    ax.set_title(r"$S_\mathrm{z}$")

    sc = axes[1].scatter(kpoints2D[:,0], kpoints2D[:,1], s = scatter_size, c=bands2D[:,band-1]-E_F, cmap='copper')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label(r'$E-E_\mathrm{F}$ (eV)')
    axes[1].set_title("energies")

    if scatter_for_quiver is True:
        axes[2].scatter(kpoints2D[:,0], kpoints2D[:,1], s=scatter_size_quiver, c='k')
    reduce_by_factor_local = reduce_by_factor * 3
    sc = axes[2].quiver(kpoints2D[:,0][::reduce_by_factor_local], kpoints2D[:,1][::reduce_by_factor_local], Sx2D[:,band-1][::reduce_by_factor_local],
                             Sy2D[:,band-1][::reduce_by_factor_local], Sz2D[:,band-1][::reduce_by_factor_local], 
                             scale=quiver_scale, cmap='seismic')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label(r'$S_\mathrm{z}$')
    sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])
    # cbar.vmin = -1.0
    # cbar.vmax = 1.0
    axes[2].set_title(f"spin directions")
    axes[2].set_facecolor("#777777")

    for i in range(1,3):
        axes[i].set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)', fontsize=12)
        axes[i].set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)', fontsize=12)
        axes[i].set_aspect('equal')
        if kmesh_limits:
            axes[i].set_xlim(kmesh_limits)
            axes[i].set_ylim(kmesh_limits)

    plt.suptitle(f"{os.getcwd()}\npyplot.quiver scale={quiver_scale}\nband {band}", fontsize=6)
    plt.tight_layout()
    fout = f"selected_band{band}_plot_E{bands_ylim[0]:.1f}-{bands_ylim[1]:.1f}eV.jpg"
    # fout = check_file_exists(fout)
    plt.savefig(fout, dpi=400)
    # plt.show()
    plt.close()


def coerce_to_positive_angles(angles):
    """Coerce angles to be positive (i.e., in the range [0, 2pi])"""
    return np.where(angles < 0, angles + 2*np.pi, angles)


def replace_middle_of_cmap_with_custom_color(color_middle=(0.85, 0.85, 0.85, 1.0), middle_range=0.10):
    # create custom cmap going from blue through grey to red
    cmap = plt.cm.get_cmap('seismic')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # change the 10% of values around the middle to interpolate from the value at cmap.N * 0.4 to the value at cmap.N * 0.6 through gray (0.5, 0.5, 0.5, 1.0)
    #  i.e., from blue through gray to red
    i_start = int(cmap.N * (0.5-middle_range))
    i_middle = int(cmap.N * 0.5)
    i_end = int(cmap.N * (0.5+middle_range))
    for i in range(i_start, i_middle):
        # linear interpolation
        alpha = (i - i_start) / (i_middle - i_start)
        cmaplist[i] = tuple([alpha * color_middle[j] + (1 - alpha) * cmaplist[i][j] for j in range(4)])
    for i in range(i_middle, i_end):
        # linear interpolation
        alpha = (i - i_middle) / (i_end - i_middle)
        cmaplist[i] = tuple([(1 - alpha) * color_middle[j] + alpha * cmaplist[i][j] for j in range(4)])
    # create new cmap
    cmap = cmap.from_list('seismic_through_gray', cmaplist, cmap.N)
    return cmap


def fermi_surface_spin_texture(kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, ax=None, E=0, E_F=0, E_thr=0.01, fig_name=None, quiver_scale=1, \
                                scatter_for_quiver=True, scatter_size_quiver=1, scatter_size=0.8, reduce_by_factor=1, \
                                    kmesh_limits=None, savefig=True, colorbar_Sx_lim=[-1,1], colorbar_Sy_lim=[-1,1], \
                                        colorbar_Sz_lim=[-1,1], n_points_for_one_angstrom_radius=120, ylim_margin=0.1, \
                                            contour_for_quiver=True, contour_line_width=2.5, arrow_linewidth=0.005, \
                                                arrow_head_width=3, quiver_angles='xy', quiver_scale_units='xy', \
                                                    inset_with_units_of_arrows=True, color_middle=(0.85, 0.85, 0.85, 1.0)):
    """Plot scatter points with a spin texture on a constant energy xy surface (probably at E=EF)
        if the energy difference of each given point is lower than some threshold. 
        If there is more such points, grab the one with minimum difference from the energy surface.
        
        - E and E_thr in eV
        
        kpoints2D: 2D array of shape (Nkpoints, 3)
        bands2D: 2D array of shape (Nkpoints, Nbands)

        circumf_distance_of_arrows: distance between the arrows on the circumference of the circle = k (1/Angstrom) * phi (rad)
                        """
    # make the cut at 'E'
    energy_distances = np.abs(bands2D - E_F - E)
    include_kpoint = np.any(energy_distances <= E_thr, axis=1)
    closest_band = np.argmin(energy_distances, axis=1)[include_kpoint]

    # get the cartesian (kx, ky) and polar (k, phi) coordinates of the kpoints
    kx = kpoints2D[include_kpoint,0]
    ky = kpoints2D[include_kpoint,1]
    k = np.sqrt(kx**2 + ky**2)
    # make phi start along the y-axis and go counterclockwise
    phi = coerce_to_positive_angles(np.arctan2(ky, kx) - np.pi)

    # convert to pandas dataframe
    df = pd.DataFrame({'kx': kx, 'ky': ky, 'k': k, 'phi': phi, 'closest_band': closest_band, \
                       'Sx': Sx2D[include_kpoint, closest_band], \
                        'Sy': Sy2D[include_kpoint, closest_band], 'Sz': Sz2D[include_kpoint, closest_band]})

    # save the mean radius for each corresponding band
    df_bands = pd.DataFrame()
    df_bands['k_mean'] = df.groupby('closest_band')['k'].mean()

    # get the closest integer number of points for each band divisible by 6
    df_bands['n_points'] = df_bands['k_mean'].apply(lambda x: (int(x * n_points_for_one_angstrom_radius) // 6)*6)

    # for each band
    #   create numpy array of phi_anchors
    #       for each point get the closest anchor and the difference from it
    #           for each anchor (i.e., group by anchors) keep only the row with the minimum difference from this anchor
    # include index of df as a column
    df['index'] = df.index
    df_filtered = pd.DataFrame()
    for band in df_bands.index:
        phi_anchors = np.linspace(0, 2*np.pi, df_bands.loc[band, 'n_points'], endpoint=False)
        df_temp = df[df['closest_band'] == band]
        try:
            df_temp['closest_anchor'] = df_temp['phi'].apply(lambda x: phi_anchors[np.argmin(np.abs(phi_anchors - x))])
        except ValueError:
            print(f'No points found at Fermi; increase the k-point meshing density!')
            return -1
        df_temp.loc[:,'anchor_difference'] = (df_temp.loc[:,'phi'] - df_temp.loc[:,'closest_anchor']).abs()
        df_filtered_temp = df_temp.groupby('closest_anchor').apply(lambda x: x.loc[x['anchor_difference'].idxmin()])
        df_filtered = pd.concat([df_filtered, df_filtered_temp])

    # get the kx, ky, Sx, Sy, Sz values for the arrows
    kx_radial_filter = df_filtered['kx'].values
    ky_radial_filter = df_filtered['ky'].values
    Sx_radial_filter = df_filtered['Sx'].values
    Sy_radial_filter = df_filtered['Sy'].values
    Sz_radial_filter = df_filtered['Sz'].values

    print('average Sz', np.mean(Sz_radial_filter))

    # ARROWS Sz
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[5,5])
        plot_to_external_axis = False
    else:
        plot_to_external_axis = True

    if scatter_for_quiver is True:
        ax.scatter(kx, ky, s=scatter_size_quiver, c='k', zorder=5)
    
    if contour_for_quiver is True:
        for band in range(bands2D.shape[1]):
            Nkpoints = int(len(kpoints2D[:,0]) ** (1/2))
            x = kpoints2D[:,0].reshape(Nkpoints,Nkpoints)  # x-coordinates
            y = kpoints2D[:,1].reshape(Nkpoints,Nkpoints)  # y-coordinates
            z = bands2D[:, band].reshape(Nkpoints,Nkpoints)
            ax.contour(x, y, z, levels=[E_F+E], colors='darkgray', linestyles='solid', linewidths=contour_line_width, zorder=0)

    seismic_with_gray = replace_middle_of_cmap_with_custom_color(color_middle=color_middle, middle_range=0.07)

    sc = ax.quiver(kx_radial_filter, ky_radial_filter, Sx_radial_filter, Sy_radial_filter, Sz_radial_filter, scale=quiver_scale, cmap=seismic_with_gray, \
                   width=arrow_linewidth, headwidth=arrow_head_width, zorder=10, angles=quiver_angles, scale_units=quiver_scale_units)
    ax.set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)', fontsize=12)
    ax.set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)', fontsize=12)
    ax.set_title(f"{os.getcwd()}\npyplot.quiver scale={quiver_scale}", fontsize=6)
    # place inset text outside on the right always in the middle of the y axis, rotated 90 degrees clockwise
    if inset_with_units_of_arrows is True:
        ax.text(1.015, 0.5, r"[$S$] = " + f"{1/quiver_scale:.2f}" + r" $\mathrm{\AA}^{-1}$", fontsize=7, transform=ax.transAxes, rotation=90, va='center', ha='left')

    # set the limits of the plot
    if kmesh_limits:
        ax.set_xlim(kmesh_limits)
        ax.set_ylim(kmesh_limits)
    else:
        # take into account the margin
        ax.set_xlim([min(kx) - (max(kx) - min(kx)) * ylim_margin, max(kx) + (max(kx) - min(kx)) * ylim_margin])
        ax.set_ylim([min(ky) - (max(ky) - min(ky)) * ylim_margin, max(ky) + (max(ky) - min(ky)) * ylim_margin])

    ax.set_aspect('equal')
    # set ticks inside the plot
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    # ax.set_facecolor("#777777")
    if plot_to_external_axis is False:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
        cbar.set_label(r'$S_\mathrm{z}$')

    if colorbar_Sz_lim:
        sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])
    plt.tight_layout()
    if fig_name is None: 
        fig_name_all_one = f"spin_texture_2D_all_in_one_E-from-EF{E-E_F:.3f}eV.jpg"
    else:
        fig_name_split = fig_name.split('.')
        fig_name_all_one = f"{''.join(fig_name_split[:-1])}_all_in_one_E-from-EF{E-E_F:.3f}eV.{fig_name_split[-1]}"
    fig_name_all_one = check_file_exists(fig_name_all_one)

    if savefig is True:
        plt.savefig(fig_name_all_one, dpi=400)
        # plt.show()
        plt.close()

    fig2, axes = plt.subplots(1, 3, figsize=[12,4])

    sc1 = axes[0].scatter(kx, ky, c=Sx2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig2.colorbar(sc1, cax=cax, orientation='vertical')
    cbar1.set_label(r'$S_\mathrm{x}$')
    axes[0].set_title(r'$S_\mathrm{x}$')
    if colorbar_Sx_lim:
        sc1.set_clim(vmin=colorbar_Sx_lim[0], vmax=colorbar_Sx_lim[1])

    sc2 = axes[1].scatter(kx, ky, c=Sy2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig2.colorbar(sc2, cax=cax, orientation='vertical')
    cbar2.set_label(r'$S_\mathrm{y}$')
    axes[1].set_title(r'$S_\mathrm{y}$')
    if colorbar_Sy_lim:
        sc2.set_clim(vmin=colorbar_Sy_lim[0], vmax=colorbar_Sy_lim[1])

    sc3 = axes[2].scatter(kx, ky, c=Sz2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig2.colorbar(sc3, cax=cax, orientation='vertical')
    cbar3.set_label(r'$S_\mathrm{z}$')
    axes[2].set_title(r'$S_\mathrm{z}$')
    if colorbar_Sz_lim:
        sc3.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

    for i in range(3):
        axes[i].set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)', fontsize=12)
        axes[i].set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)', fontsize=12)
        axes[i].set_aspect('equal')
        if kmesh_limits:
            axes[i].set_xlim(kmesh_limits)
            axes[i].set_ylim(kmesh_limits)

    plt.suptitle(f"{os.getcwd()}", fontsize=6)
    plt.tight_layout()
    if fig_name is None: 
        fig_name_Sxyz = f"spin_texture_2D_energy_cut_E-from-EF{E-E_F:.3f}eV.jpg"
    else:
        fig_name_split = fig_name.split('.')
        fig_name_Sxyz = f"{''.join(fig_name_split[:-1])}_2D_energy_cut_E-from-EF{E-E_F:.3f}eV.{fig_name_split[-1]}"
    fig_name_Sxyz = check_file_exists(fig_name_Sxyz)

    if savefig is True:
        plt.savefig(fig_name_Sxyz, dpi=400)
        # plt.show()
        plt.close()
    else:
        plt.close()
        if plot_to_external_axis is False:
            return fig, fig2
        

def interpolate_operator(operator_dict, u_dis_dict, u_dict, latt_params, reciprocal_latt_params, 
                            R_grid, U_mn_k=None, hamiltonian=True, kpoints=[(0,0,0)], 
                            save_real_space=False, real_space_fname="hr_R_dict.dat", \
                                save_folder="./tb_model_wann90/", save_as_pickle=True):
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
    

def parse_KPOINTS_file(KPOINTS_file_path):
    with open(KPOINTS_file_path, 'r') as fr:
        lines = fr.readlines()
        Nkpoints = int(lines[1].split()[0])
        kpoint_matrix = []
        kpath_ticks = []

        empty_line = True
        for i in range(4, len(lines)):
            if lines[i].strip() == '':
                # if two empty lines in a row, break
                if empty_line == True:
                    break
                else:
                    empty_line = True
                    continue
            lsplit = lines[i].split()
            current_kpoint = tuple([float(num) for num in lsplit[:3]]) 
            if empty_line is True:
                kpoint_matrix.append([])
            kpoint_matrix[-1].append(current_kpoint)
            
            # if there is no k-point label, just make it an empty string
            ktick = ''
            if '!' in lines[i]:
                ktick = lines[i].split('!')[-1].strip()

            if kpath_ticks == []:
                # if its the first k-point label, just add it
                kpath_ticks.append(ktick)
            elif empty_line is False:
                # if it's the second point of a given segment, just add it
                kpath_ticks.append(ktick)
            elif empty_line is True and ktick != kpath_ticks[-1]:
                # else if this is the beginning of a new k-path segment BUT the ktick is NOT the same as the last one, 
                # (meaning discontinuous kpath), then you need to merge the two high-symmetry point labels into one label
                kpath_ticks[-1] = f'{kpath_ticks[-1]}|{ktick}'
                # otherwise means they are the same and in that case do not add anything to kpath_ticks

            if empty_line is True:
                empty_line = False

    return kpoint_matrix, Nkpoints, kpath_ticks