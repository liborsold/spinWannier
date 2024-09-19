import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import uniform_real_space_grid, load_lattice_vectors

# ================ USER INPUT ==================
fin = "wannier90_hr.dat"
distance_in_fractional_coord = False        # if true, the distance in Angstrom will be divided by the length of the first lattice vector
consider_wannier_center_distances = True   # should the slight correction of the wannier center distances be added? (for accurate distance)
# ==============================================

def get_drxyz_mn(fin="wannier90_centres.xyz"):
    """Get the distance between centres of m and n wannier function."""
    with open(fin, 'r') as fr:
        Dat = []
        for i, line in enumerate(fr):
            if i >= 2:
                Dat.append(line.split()[1:])
        xyz = np.array(Dat[:-3][:], dtype=np.float32)

    # plot centers in 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    if consider_wannier_center_distances is True:
        plt.title('Wannier center positions')
    else:
        plt.title('Neighboring cell distance')
    ax.set_xlabel('x (A)')
    ax.set_ylabel('y (A)')
    ax.set_zlabel('z (A)')
    # plt.show()
    plt.close()

    return xyz


def main():
    dr_xyz_mn = get_drxyz_mn(fin="wannier90_centres.xyz")
    # load the 
    A = load_lattice_vectors(win_file="wannier90.win")

    with open(fin, 'r') as fr:
        for i, line in enumerate(fr):
            if i == 2:
                nkpoints = int(line.split()[0])
                skiprows = 3 + nkpoints//15
                if nkpoints%15 != 0: skiprows += 1
    Dat = np.loadtxt(fin, skiprows=skiprows)
    N = Dat.shape[0]
    
    T_abs = np.zeros((N,))
    Rxyz = Dat[:,:3] @ A

    for i, line in enumerate(Dat):
        # add the vector from center n to center m (dr_xyz_m - dr_xyz_n)
        if consider_wannier_center_distances is True:
            Rxyz[i] += dr_xyz_mn[int(line[3]-1)] - dr_xyz_mn[int(line[4]-1)]
        T_abs[i] = np.linalg.norm(np.array([line[5], line[6]]))

    # take the norm of the distance vector
    R = np.linalg.norm( Rxyz, axis=1)
    # divide by the norm of the first lattice param.
    R_over_a = R / np.linalg.norm(A[0,:])

    # calculate maximum interaction distance for different grids (smaller than the base)
    Rxyz_rec = Dat[:,:3]

    meshes_ijk = [(3,3,1), (5,5,1), (7,7,1)]
    max_distance_for_mesh = []
    for mesh in meshes_ijk:
        R_grid = uniform_real_space_grid(R_mesh_ijk = mesh)
        idx_in_mesh = [tuple(Rxyz_rec_) in R_grid for Rxyz_rec_ in Rxyz_rec]
        # get maximum of only those interactions where the vector is in the current 'mesh'
        if distance_in_fractional_coord is True:
            max_distance_for_mesh.append( np.max(R_over_a[idx_in_mesh]) )
        else:
            max_distance_for_mesh.append( np.max(R[idx_in_mesh]) )

    if distance_in_fractional_coord is True:
        x = R_over_a
        xlabel = r"$r_\mathrm{ij}/a$"
    else:
        x = R
        xlabel = r"$r_\mathrm{ij}$ (A)"

    fig, ax = plt.subplots(1,1, figsize=(4,3))
    plt.scatter(x, T_abs)
    ax.axhline(linestyle='--', color='k')
    plt.ylabel(r"|$t_\mathrm{ij}$|", fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    figname = 'hopping_vs_cell_distance' if distance_in_fractional_coord is True else 'hopping_vs_distance'
    plt.title(figname.split('.')[0].replace('_', ' '))

    # plot vertical lines at meshes
    for max_distance in max_distance_for_mesh:
        ax.axvline(x=max_distance, linestyle='--', color='k')
    # add labels to them
    text_references = []
    y_text_place = 0.7 * ( (ax.get_ylim()[1] - ax.get_ylim()[0])/2 + ax.get_ylim()[0])
    for mesh, mesh_dist in zip(meshes_ijk, max_distance_for_mesh):
        shift = 2/np.linalg.norm(A[0,:]) if distance_in_fractional_coord is True else 2
        text_ref = plt.text(mesh_dist - shift, y_text_place, str(mesh), rotation='vertical')
        text_references.append(text_ref)

    plt.tight_layout()

    # save figure
    plt.savefig(figname+'.png', dpi=400)

    # semilog scale
    ax.set_yscale('log')
    # ax.relim()
    # update ax.viewLim using the new dataLim
    # ax.autoscale_view()
       # clear text, add new one on different place
    for text_ref in text_references:
        text_ref.remove()
    y_text_place = 10**(0.7 * ((log10(ax.get_ylim()[1]) - log10(ax.get_ylim()[0]))/2 + log10(ax.get_ylim()[0])))
    for mesh, mesh_dist in zip(meshes_ijk, max_distance_for_mesh):
        shift = 2/np.linalg.norm(A[0,:]) if distance_in_fractional_coord is True else 2
        plt.text(mesh_dist - shift, y_text_place, str(mesh), rotation='vertical')
        text_references.append(text_ref)
    
    plt.tight_layout()
    plt.savefig(figname+'_semilog.png', dpi=400)
    plt.close()
    # plt.show()

if __name__ == '__main__':
    main()
