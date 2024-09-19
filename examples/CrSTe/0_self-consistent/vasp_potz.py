#!/usr/bin/env python

from pymatgen.io.vasp.outputs import Locpot

import numpy as np
import matplotlib.pyplot as plt


def plot_locpot(folder, shift=0):
    pot = Locpot.from_file(folder+'/LOCPOT')
    grid = pot.get_axis_grid(ind=2)
    avg = pot.get_average_along_axis(ind=2)
    
    np.savetxt(folder+'/potz.dat',np.array([grid,avg-shift]).T)

    plt.plot(grid, avg-shift)
    plt.savefig('potz.png', dpi=400)

plot_locpot('./')






