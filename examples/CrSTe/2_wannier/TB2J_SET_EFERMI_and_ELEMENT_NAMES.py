# in TB2J_SOC.job replace
   # EFERMI_PLACEHOLDER
   # ELEMENT_NAMES_PLACEHOLDER

import numpy as np

def get_element_names_from_POSCAR(POSCAR_file_name):
    """_summary_

    Args:
        POSCAR_file_name (str): _description_

    Returns:
        _type_: _description_
    """
    with open(POSCAR_file_name, 'r') as f:
        lines = f.readlines()
    element_names_nonrepeated = lines[5].split()
    element_numbers = lines[6].split()
    # repeat each element_name element_number times
    element_names = [element_name for element_name, element_number in zip(element_names_nonrepeated, element_numbers) for _ in range(int(element_number))]
    return element_names

# ===================== USER INPUTS =====================
Ef = float(np.loadtxt('./FERMI_ENERGY.in'))
element_names = get_element_names_from_POSCAR('./POSCAR')
# ========================================================

# in TB2J_SOC.job replace EFERMI_PLACEHOLDER and ELEMENT_NAMES_PLACEHOLDER by Ef and element_names
with open('TB2J_SOC.job', 'r') as fr:
    TB2J_job_file_lines = fr.readlines()

TB2J_job_file_lines_new = []
for line in TB2J_job_file_lines:
    if 'EFERMI_PLACEHOLDER' in line:
        line = line.replace('EFERMI_PLACEHOLDER', f'{Ef:.8f}')
    if 'ELEMENT_NAMES_PLACEHOLDER' in line:
        line = line.replace('ELEMENT_NAMES_PLACEHOLDER', ' '.join(element_names))
    TB2J_job_file_lines_new.append(line)

with open('TB2J_SOC.job', 'w') as fw:
    fw.writelines(TB2J_job_file_lines_new)



