import pyprocar
from os.path import exists

# user parameters
# ====================================================

PROCAR_repaired = 'PROCAR-repaired'

# ====================================================

directions = ['x', 'y', 'z']
PROCAR_dirs = ["PROCAR_s" + direction for direction in directions]

# first repair PROCAR
if not exists(PROCAR_repaired):
    pyprocar.repair('PROCAR',PROCAR_repaired)
else:
    print(f"{PROCAR_repaired} does exist")

# then FILTER SX, SY, SZ COMPONENTs FROM PROCAR
for i, PROCAR_dir in enumerate(PROCAR_dirs):
    if not exists(PROCAR_dir):
        pyprocar.filter(PROCAR_repaired, PROCAR_dir, spin=[1+i])  # spin in z direction; change to [2] for y and [1] for x
    else:
        print(f'{PROCAR_dir} does exist')

