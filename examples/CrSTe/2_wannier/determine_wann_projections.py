import re
fin = 'INCAR'

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

n_elements, element_names = parse_structural(POSCAR=f"POSCAR")

proj_str = ""
for i, name in enumerate(element_names):
    if i == 0:
        # for first element (Cr) put d orbitals
        proj_str += name + ":d\n"
    else:
        # for all other elements put p orbitals
        proj_str += name + ":p\n"

# replace projections in the INCAR file

with open(fin, 'r') as fr:
    str_INCAR = fr.read()

proj_str_INCAR = re.findall('Begin Projections[\S\n ]+End Projections', str_INCAR)

str_INCAR = str_INCAR.replace(proj_str_INCAR[0], f"Begin Projections\n{proj_str}End Projections")

with open(fin, 'w') as fw:
    fw.write(str_INCAR)

