import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append('/home/lv268562/bin/python_scripts')
# from get_fermi import get_fermi

# =============== USER DEFINED ====================
ylim = [-6, 11]
sc_calc_folder = "../../../../sc_SOC/"
outfile = 'wannier90-bands_Sxyz.jpg'
# =================================================

E_fermi = -2.31797502 # get_fermi(path=sc_calc_folder)

fig, axes = plt.subplots(1, 3, figsize=[10,4])

for i, S in enumerate(['Sx', 'Sy', 'Sz']):
    ax = axes[i]
    data = np.loadtxt(f'wannier90-bands_{S}.dat')
    x=data[:,0]
    y=data[:,1] - E_fermi
    z=data[:,2]
    #ticks
    image = ax.scatter(x,y,c=z,marker='+',s=1,cmap=plt.cm.seismic, vmin=-1, vmax=1)
    ax.set_title(S)
    ax.set_xlim([0,max(x)])
    ax.set_ylim(ylim)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.axhline(y=0, color='k', linestyle='--')
    for n in range(1,len(tick_locs)):
        ax.plot([tick_locs[n],tick_locs[n]],[ylim[0],ylim[1]],color='gray',linestyle='-',linewidth=0.7)
    if i == 0:
        ax.set_ylabel(r'E - E$_\mathrm{F}$ (eV)')

    # add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(image, cax=cax, orientation='vertical')
    #ax.axes().set_aspect(aspect=0.65*max(x)/(max(y)-min(y)))
    #ax.colorbar(shrink=0.7)

plt.suptitle('Wannier band structure', fontsize='x-large')
fig.tight_layout()
plt.savefig(outfile,bbox_inches='tight', dpi=400)
# plt.show()
