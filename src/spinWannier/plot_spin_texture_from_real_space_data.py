from spin_texture_interpolated import real_to_W_gauge, W_gauge_to_H_gauge, save_bands_and_spin_texture, \
                                        plot_bands_spin_texture
from wannier_utils import split_spn_dict, get_kpoint_path, get_2D_kpoint_mesh, load_lattice_vectors, reciprocal_lattice_vectors, unite_spn_dict,\
        hr_wann90_to_dict
import pickle


# ================================================== INPUTS ==================================================================
EF = -2.31797502 #-2.31797166 # for CrTe2 with EFIELD

wannSymm_hr = False
wannSymm_hr_file = "wannier90_symmed_hr.dat"
spn_present = True

fout = 'spin_texture_1D_home_made_symmetric_wannierBerri.jpg'
fig_caption = "symmetrized by wannierBerri"

hr_R_name_sym = "hr_R_dict_sym.pickle"
spn_R_name_sym = "spn_R_dict_sym.pickle"

kmesh_2D = False
kmesh_density = 100

kmesh_2D_limits = [-0.5, 0.5]
yaxis_lim = [-6, 11]

kpoint_matrix = [
    [(0.00,  0.50,  0.00),    (0.00,  0.00,  0.00)],
    [(0.00,  0.00,  0.00),    (0.67, -0.33,  0.00)],
    [(0.67, -0.33,  0.00),    (0.00,  0.50,  0.00)]
]
# =============================================================================================================================

A = load_lattice_vectors(win_file="wannier90.win")
G = reciprocal_lattice_vectors(A)

# create k-point mesh
if kmesh_2D is True:
    kpoints = get_2D_kpoint_mesh(G, limits=kmesh_2D_limits, Nk=kmesh_density)
else:
    kpoints, kpath = get_kpoint_path(kpoint_matrix, G, Nk=kmesh_density)

# load the eigenvalues and spin projection dictionaries
with open(hr_R_name_sym, 'rb') as fr:
    hr_R_dict = pickle.load(fr)

if wannSymm_hr is True:
    hr_R_dict = hr_wann90_to_dict(fin_wannier90 = wannSymm_hr_file)

# interpolate Hamiltonian
H_k_W = real_to_W_gauge(kpoints, hr_R_dict)
Eigs_k, U_mn_k = W_gauge_to_H_gauge(H_k_W, U_mn_k={}, hamiltonian=True)

# interpolate spin operator
if spn_present:
    with open(spn_R_name_sym, 'rb') as fr:
        spn_R_dict = pickle.load(fr)

    S_mn_R_x, S_mn_R_y, S_mn_R_z = split_spn_dict(spn_R_dict, spin_names=['x','y','z'])

    Sx_k_W = real_to_W_gauge(kpoints, S_mn_R_x)
    Sy_k_W = real_to_W_gauge(kpoints, S_mn_R_y)
    Sz_k_W = real_to_W_gauge(kpoints, S_mn_R_z)

    S_mn_k_H_x = W_gauge_to_H_gauge(Sx_k_W, U_mn_k=U_mn_k, hamiltonian=False)
    S_mn_k_H_y = W_gauge_to_H_gauge(Sy_k_W, U_mn_k=U_mn_k, hamiltonian=False)
    S_mn_k_H_z = W_gauge_to_H_gauge(Sz_k_W, U_mn_k=U_mn_k, hamiltonian=False)

# save_bands_and_spin_texture(kpoints, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, kmesh_2D=False, 
#                                 fout='bands_spin.pickle', fout2D='bands_spin_2D.pickle')

# plotting
if spn_present:
    plot_bands_spin_texture(kpoints, kpath, Eigs_k, S_mn_k_H_x, S_mn_k_H_y, S_mn_k_H_z, EF=EF, fout=fout.split('.jpg')[0]+"_Sxyz.jpg", fig_caption=fig_caption)

# if no spin information, put {} for S
plot_bands_spin_texture(kpoints, kpath, Eigs_k, {}, {}, {}, EF=EF, fout=fout, fig_caption=fig_caption)

