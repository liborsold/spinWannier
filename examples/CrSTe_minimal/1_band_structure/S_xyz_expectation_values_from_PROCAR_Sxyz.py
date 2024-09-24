import regex as re
import numpy as np

fin = 'PROCAR_sx'

S = []
for s in ['x', 'y', 'z']:
    fin = f"PROCAR_s{s}"
    S_i = []
    with open(fin, 'r') as fr:
        for line in fr:
            tot_line = re.findall(r"^tot.*", line)
            if tot_line:
                S_i.append( float(tot_line[0].split()[-1]) )
    S.append(S_i)

np.savetxt("Sxyz_exp_values.dat", np.array(S).T, header="following the structure of PROCAR: all bands at 0th k-point, follows all bands at 1st k-point, ...\nSx\tSy\tSz")
