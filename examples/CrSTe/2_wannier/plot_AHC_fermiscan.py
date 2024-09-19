"""Plot the AHC fermiscan for all the iterations."""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
import numpy as np

# ======================== user input ==================

Ef = float(np.loadtxt('FERMI_ENERGY.in')) # eV
seedname = "AHC_fermiscan"
n_adaptive = 30   # number of adaptive steps
ylim = [-1e3, 1e3]
#all_iterations = True
iters = [30] #range(0, 16, 3)  #   # adaptive iteration step
components = [0, 1, 2] #[2] #   # array of values; 0 stands for x, 1 for y, 2 for z

# ======================================================

component_names = ['x', 'y', 'z']
component_colormaps = ['Reds', 'Greens', 'Blues']

fig, ax = plt.subplots(1, 1, figsize=[8, 5])

#iters = range(n_adaptive) if all_iterations == True else [n_adaptive-1]

for j in components:
    for i in iters:

        with open(f"{seedname}-ahc_iter-{i:04d}.dat") as fin:
            data = np.loadtxt(fin, skiprows=2)
            plt.plot(data[:,0]-Ef, data[:,j+1]/100, "-", label=f"{component_names[j]}, ad. iter. {i:02d}", markerfacecolor="None")
    
    plt.set_cmap(matplotlib.cm.get_cmap(name=component_colormaps[j])) 

if ( len(iters) + len(components) ) > 1:
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.7))

plt.title("Intrinsic Anomalous Hall conductivity")
plt.plot([-100, 100], [0, 0], "k-", linewidth=1)
plt.plot([0,0], [-1e6, 1e6], "k--", linewidth=1)
plt.xlabel(r"$ E - E_\mathrm{F}$ (eV)")
plt.ylabel("$ \sigma^\mathrm{AH, int.}$ (S/cm)")
plt.xlim([min(data[:,0])-Ef, max(data[:,0])-Ef])
plt.ylim(ylim)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
plt.savefig("AHC_fermiscan.png", dpi=400)

