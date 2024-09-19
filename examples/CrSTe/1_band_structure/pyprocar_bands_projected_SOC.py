import pyprocar
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import copy
import numpy as np
from os.path import exists
import sys
sys.path.append('/home/lv268562/bin/python_scripts')
from get_fermi import get_fermi

# user parameters
# ====================================================

SOC_nsc_calc_folder = "../sc"

E_fermi = get_fermi(path=SOC_nsc_calc_folder) # eV; take from DOSCAR of SC calculation, because bandstructure calculation has unrepresentative number of kpoints

orbital_groups =  [[0], [1,2,3], [4,5,6,7,8]] #list(range(1,4)), list(range(4,9))]
orbital_names  =  ['s', 'p', 'd']
spins          =  [0]
spin_colormaps =  ['seismic']

PROCAR_sz = 'PROCAR_sz'

limits=[-6, 11]   # eV; yaxis energy limits for bandstructure

colorbar_min = -1
colorbar_max = 1
PROCAR_repaired = 'PROCAR-repaired'
# ====================================================
 # GET ELEMENT NAMES AND NUMBER AUTOMATICALLY

def parse_structural(POSCAR='POSCAR'):
    """From POSCAR parse the names of the elements and their numbers."""
    with open(POSCAR, 'r') as fr:
        for i, line in enumerate(fr):
            if i ==5:
                element_names = line.split()
            if i == 6:
                l_split = line.split()
                n_elements = [ int(n) for n in l_split ]
                return n_elements, element_names

n_elements, element_names = parse_structural(POSCAR='POSCAR')

min_range = 0
elements = []
for i in range(len(n_elements)):
    elements.append(list(range(min_range, min_range + n_elements[i])))
    min_range += n_elements[i]

# ====================================================

# first repair PROCAR
if not exists(PROCAR_repaired):
    pyprocar.repair('PROCAR',PROCAR_repaired)
else:
    print(f"{PROCAR_repaired} does exist")

# then FILTER SZ COMPONENT FROM PROCAR
if not exists(PROCAR_sz):
    pyprocar.filter(PROCAR_repaired, PROCAR_sz, spin=[3])  # spin in z direction; change to [2] for y and [1] for x
else:
    print('SZ does exist')


# pyprocar.dosplot(filename='vasprun.xml',
#                   mode='stack_orbitals',
#                   #elimit=[-4, 4],
#                   orientation='horizontal',
#                   labels=[r'$\uparrow$', r'$\downarrow$'],
#                   title=r'Total Density of bcc Fe', 
#                   plot_total=True, 
#                   orbitals=[0,1,2],
#                   savefig="DOS_Fe.png"
#                   )


# GENERATE KPOINTS for a given structure
#pyprocar.kpath('POSCAR')




# bandstructure with DOS
# pyprocar.bandsdosplot(  bands_file='PROCAR', 
#                         outcar='OUTCAR', 
#                         dos_file='vasprun.xml', 
#                         bands_mode='plain',
#                         dos_mode='plain', 
#                         dos_labels=[r'$\uparrow$',r'$\downarrow$'], 
#                         elimit=[-30,30], 
#                         kpointsfile='KPOINTS')


# MULTIPLOT of bandstructures
fig_height = 1 + 7*len(n_elements)
fig, axes = plt.subplots(len(elements),len(orbital_groups), figsize=[30,fig_height])

for i, element in enumerate(elements):
    for j, orbital_group in enumerate(orbital_groups):
        axis = axes[i, j] if len(elements) > 1 else axes[j]
        axis.set_title(f"{element_names[i]} {orbital_names[j]}")
        axis.yaxis.set_minor_locator(MultipleLocator(1))
        for k, spin in enumerate(spins):
            orb_group = copy.copy(orbital_group)  # need to copy because of sum bug where pyprocar increases the number of 'orb_group' every spin iteration, which would give wrong orbitals for spin down
            pyprocar.bandsplot(PROCAR_sz, repair=True, linewidth=3, outcar='OUTCAR',mode='parametric', orbitals=orb_group, atoms=element, cmap=spin_colormaps[k],kpointsfile='KPOINTS', code='vasp', ax=axis, elimit=limits, show=False, vmin=colorbar_min, vmax=colorbar_max, fermi=E_fermi)

plt.savefig(f"bands_projected_{limits[0]:.2f}-{limits[1]:.2f}eV.jpg", dpi=300)
plt.close()
