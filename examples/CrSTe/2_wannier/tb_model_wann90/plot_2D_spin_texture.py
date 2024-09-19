from functools import reduce
import pickle
from tabnanny import check
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import check_file_exists

# ==========  USER DEFINED  ===============
fin_1D = "bands_spin.pickle" #"bands_spin_model.pickle" #"./tb_model_wann90/bands_spin.pickle" #"bands_spin_model.pickle" #
fin_2D = "bands_spin_2D_100X100.pickle" #"bands_spin_2D_model.pickle" #"./tb_model_wann90/bands_spin_2D.pickle" #"bands_spin_2D_model.pickle"
E_F = float(np.loadtxt('../FERMI_ENERGY.in')) #-2.31797502 # for CrTe2 with EFIELD: -2.31797166
E_to_cut_2D = E_F #E_F # + 1.44
kmesh_limits = None #[-.5, .5] #None #   # unit 1/A; put 'None' if no limits should be applied
colorbar_Sx_lim = [-1, 1] #[-0.2, 0.2] #None # put None if they should be determined automatically
colorbar_Sy_lim = [-1, 1] #[-0.2, 0.2] #None # put None if they should be determined automatically
colorbar_Sz_lim = [-1, 1] #None # put None if they should be determined automatically

bands_ylim = [-6, 11]

band = 8 #160   # 1-indexed
E_thr = 0.040 #0.020  # eV
quiver_scale_magmom_OOP = 1 # quiver_scale for calculations with MAGMOM purely out-of-plane
quiver_scale_magmom_IP = 10   # quiver_scale for calculations with MAGMOM purely in-plane
reduce_by_factor = 1 # take each 'reduce_by_factor' point in the '_all_in_one.jpg' plot

scatter_size = 0.8

scatter_for_quiver = True
scatter_size_quiver = 0.1
# ========================================

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


def selected_band_plot(band=0):
    # -------------- SELECTED BAND plot ------------------

    fig, axes = plt.subplots(1, 3, figsize=[14,4])
    spin_name = ['Sx', 'Sy', 'Sz']

    ax = axes[0]
    ax.axhline(linestyle='--', color='k')
    ax.set_xlim([min(kpath1D), max(kpath1D)])
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_xlabel('k-path (1/$\mathrm{\AA}$)')
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
        axes[i].set_xlabel(r'$k_\mathrm{x}$ (1/$\mathrm{\AA}$)')
        axes[i].set_ylabel(r'$k_\mathrm{y}$ (1/$\mathrm{\AA}$)')
        axes[i].set_aspect('equal')
        if kmesh_limits:
            axes[i].set_xlim(kmesh_limits)
            axes[i].set_ylim(kmesh_limits)

    plt.suptitle(f"{os.getcwd()}\npyplot.quiver scale={quiver_scale}\nband {band}", fontsize=6)
    plt.tight_layout()
    fout = f"selected_band{band}_plot_E{bands_ylim[0]:.1f}-{bands_ylim[1]:.1f}eV.jpg"
    fout = check_file_exists(fout)
    plt.savefig(fout, dpi=400)
    # plt.show()
    plt.close()


def fermi_surface_spin_texture(kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, E=0, E_F=0, E_thr=0.01, fig_name=None, quiver_scale=1, \
                                scatter_for_quiver=True, scatter_size_quiver=1, scatter_size=0.8, reduce_by_factor=1, \
                                    kmesh_limits=kmesh_limits):
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
    plt.savefig(fig_name_all_one, dpi=400)
    # plt.show()
    plt.close()


    fig, axes = plt.subplots(1, 3, figsize=[12,4])

    sc1 = axes[0].scatter(kx, ky, c=Sx2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(sc1, cax=cax, orientation='vertical')
    cbar1.set_label(r'$S_\mathrm{x}$')
    axes[0].set_title(r'$S_\mathrm{x}$')
    if colorbar_Sx_lim:
        sc1.set_clim(vmin=colorbar_Sx_lim[0], vmax=colorbar_Sx_lim[1])

    sc2 = axes[1].scatter(kx, ky, c=Sy2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(sc2, cax=cax, orientation='vertical')
    cbar2.set_label(r'$S_\mathrm{y}$')
    axes[1].set_title(r'$S_\mathrm{y}$')
    if colorbar_Sy_lim:
        sc2.set_clim(vmin=colorbar_Sy_lim[0], vmax=colorbar_Sy_lim[1])

    sc3 = axes[2].scatter(kx, ky, c=Sz2D[include_kpoint,closest_band], cmap='seismic', s=scatter_size)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(sc3, cax=cax, orientation='vertical')
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
    plt.savefig(fig_name_Sxyz, dpi=400)
    # plt.show()
    plt.close()


def main(fig_name=None, E_to_cut=None):
    global kpoints2D, bands2D, Sx2D, Sy2D, Sz2D, kpoints1D, kpath1D, bands1D, Sx1D, Sy1D, Sz1D, quiver_scale

    magmom_OOP = magmom_OOP_or_IP('./INCAR')
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
                                E_thr=E_thr, \
                                fig_name=None, quiver_scale=quiver_scale, scatter_for_quiver=scatter_for_quiver, \
                                scatter_size_quiver=scatter_size_quiver, scatter_size=scatter_size, \
                                reduce_by_factor=reduce_by_factor, kmesh_limits=kmesh_limits
                               )


if __name__ == '__main__':
    main()

