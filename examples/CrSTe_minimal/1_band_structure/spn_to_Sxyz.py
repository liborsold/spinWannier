"""Convert seedname.spn file into a dictionary with (k-point, spin) as keys.
     -   k-point is a tuple of three float numbers
     -   spin is a string 'x', 'y', 'z'
           -   e.g., for Gamma k-point and spin 'x' the key is the tuple ((0.0, 0.0, 0.0), 'x') 

See the wannierberri's 'vaspspn.py' script for format of the spn matrix.
The input seedname.spn file MUST BE 'formatted' !!!, i.e. human-readable, not binary.
    So it must be generated with the modified vaspspn.py script which outputs txt instead of binary file.
"""

from matplotlib.pyplot import text
import numpy as np
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import get_kpoint_names
import pickle
from os.path import exists
import shutil


def spn_to_Sxyz(fwin="wannier90.win", fin="wannier90.spn_formatted", fout="Sxyz_exp_values_from_spn_file.dat", text_file=False):
    """Convert wannier90.spn file to a dictionary object and save as pickle. 
    If 'text_file' == True, then save as a human-readable text file."""

    spin_names = ['x', 'y', 'z']

    # get number of bands and kpoints
    with open(fin, 'r') as fr:
        fr.readline()
        NB = int(float(fr.readline()))
        NK = int(float(fr.readline()))

    # get the spin-projection matrices
    with open(fin, 'r') as fr:
        Sskmn = np.loadtxt(fr, dtype=np.complex64, skiprows=3)

    # construct dictionary
    Skn_xyz = np.zeros((NK*NB, 3))

    print('Sskmn shape', Sskmn.shape)

    for ik in range(NK):
        # unflatten the upper-diagonal matrix and add hermitian conjugates
        for n in range(NB):
            for m in range(n+1):
                for s in range(3):
                    if m==n:
                        Skn_xyz[ik*NB+n,s] = float(Sskmn[ik, int(3*(n*(n+1)/2 + m) + s)])

    header = "from spn projections from WannierBerri! (not PROCAR, from which the spin projections are kind of wrong)\nfollowing the structure of PROCAR: all bands at 0th k-point, follows all bands at 1st k-point, ...\n Sx	Sy	Sz"

    if exists(fout) and not exists(fout+'_old'):
        shutil.move(fout, fout+'_old')

    np.savetxt(fout, Skn_xyz, header=header, fmt='%.12e')

def main():
    spn_to_Sxyz()


if __name__ == '__main__':
    main()


