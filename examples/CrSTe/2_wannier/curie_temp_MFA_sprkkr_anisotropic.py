"""Get the Curie temperature by Mean-field approximation (MFA) ala Sasioglu 2005.
    - the sublattice index mu (or nu) is in spr-kkr a double-index (site IQ, type IT)"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob

from sympy import fraction


code_info = {
        # code_name: {
        #               'labels':           <column labels>, 
        #               'index':            <column labels serving as an index (all except the values)>, 
        #               'levels':           <column numbers not being summed over, i.e., those indicating the sublattices>,
        #               'target_column':    <column label to be used for Tc calculation>,
        #               'const':            <constant to multiply after summation>
        # }

        'sprkkr': {
                    'labels':          ['IT', 'IQ', 'JT', 'JQ', 'N1', 'N2', 'N3', 'DRX', 'DRY', 'DRZ', 'DR', 'J_xx', 'J_yy', 'J_xy', 'J_yx'], 
                    'index':           ['IT', 'IQ', 'JT', 'JQ', 'N1', 'N2', 'N3', 'DRX', 'DRY', 'DRZ', 'DR'],
                    'levels':          [0,1,2,3], 
                    'target_column':   'J_xx', 
                    'const':            2 * 1.602/1.38 * 10  # meV / kB
                    },

        'tb2j_ucf': {
                    'labels':          ['IID', 'i', 'j', 'dx', 'dy', 'dz', 'Jxx', 'Jxy', 'Jxz', 'Jyx', 'Jyy', 'Jyz', 'Jzx', 'Jzy', 'Jzz'], 
                    'index':           ['IID', 'i', 'j', 'dx', 'dy', 'dz'],
                    'levels':          [1,2], 
                    'target_column':   'Jxx', 
                    'const':            1/1.38 * 1e23      # 1 / kB
                    },
       
       
       }

def get_n_atoms_UCF(fname):
    """Get number of atoms from UCF file."""
    with open(fname, 'r') as fr:
        for i, line in enumerate(fr):
            if i == 7:
                n_atoms = int(line.split()[0])
                return n_atoms


def get_rows_to_skip_Jij(fread):
    """Find how many lines to skip in SPR-KKR's Jij.dat file"""
    rows_to_skip = 0
    with open(fread, 'r') as fr:
        for line in fr:
            rows_to_skip += 1
            if "J_xx" in line:
                return rows_to_skip + 1


def get_rows_to_skip_UCF(fread):
    """Find how many lines to skip in UCF file."""
    n_atoms = get_n_atoms_UCF(fread)
    return n_atoms + 10


def mean_field_Tc(fname, code='sprkkr', rows_to_skip=0):
      # get how many rows to skip in the file:
    if code == 'sprkkr':
        rows_to_skip = get_rows_to_skip_Jij(fname)
    elif code == 'tb2j_ucf':
        rows_to_skip = get_rows_to_skip_UCF(fname)

    with open(Path(fname), 'r') as fr:
        df_orig = pd.read_csv(fname, skiprows=rows_to_skip, delim_whitespace=True, names= code_info[code]['labels'] )
        df_orig.set_index( code_info[code]['index'] , inplace=True)
        n_rows = df_orig.shape[0]
        if n_rows == 0:
            return 0.0

        # sum over unit cells, i.e., get J0^(mu,nu) 
        df = df_orig.groupby(level= code_info[code]['levels'] ).sum()
        print(f"\nSum of {df[code_info[code]['target_column'] ]} in meV:\n {df[ code_info[code]['target_column'] ]}\n")

        # # check if size of the data is correct
        # if not int(n_rows/n_sublatts) == n_rows/n_sublatts:
        #     raise ValueError('Number of rows is not divisible by number of atoms! Check if skiprows does not trim data: n_rows/n_sublatts should be an integer. Try df_orig.head() and df_orig')

        # factorize by a suitable factor
        df[ code_info[code]['target_column'] ] *= 1/3 * code_info[code]['const']

        if code == 'sprkkr':
            J0 = df[ code_info[code]['target_column'] ].to_numpy()
            n_sublatts = int(np.sqrt(len(J0)))
            J0 = np.reshape(J0, (n_sublatts, n_sublatts))
        elif code == 'tb2j_ucf':
            n_atoms = get_n_atoms_UCF(fname)
            J0 = np.zeros((n_atoms, n_atoms))
            for i, j in df[code_info[code]['target_column'] ].index.values:
                J0[i,j] = df[code_info[code]['target_column'] ][i][j]
            #print(f"J0: {J0}")

        #print(J0)
        E_arr, v_arr = np.linalg.eig(J0)
        print(f"Eigenvalue array:\n {E_arr}\n")
        
        Tc = E_arr[np.argmax(np.abs(E_arr))]
        # Tc = np.max(E_arr)
        print(f"Curie temperature: {Tc:.3f} K     from  {code}  code and file  {fname}.")
        return float(Tc)


def main():
    
    # -- SWEEP over cropped UCFs  (Cr2Te3 TB2J)
    # code = 'tb2j_ucf'
    # UCF_fnames = glob.glob('*UCF*', recursive=False)
    # Tc_all = []
    # fractions_all = []
    # for i, file in enumerate(UCF_fnames):
    #     if "png" not in file and ".py" not in file and not ".txt" in file and file != "vampire.UCF":
    #         print(file)
    #         print(get_rows_to_skip_UCF(file))        
    #         fname_in = file
    #         Tc = mean_field_Tc(fname_in, code, rows_to_skip = get_rows_to_skip_UCF(fname_in))
    #         Tc_all.append(Tc)
    #         fractions_all.append(float(file[-5:]))
    # dat_out = np.array([fractions_all, Tc_all]).T
    # # sort by fractions
    # dat_out = dat_out[dat_out[:,0].argsort()]
    # np.savetxt('Tc_all.txt', dat_out, header='\tT_C (K)', fmt="%s\t%s")

    # -- ORIGINAL Cr2Te3 SPR-KKR
    # rows_to_skip = 29
    # fname_in = "H:/2D/Cr2Te3/bulk/transport/PBE/Cr2Te3_JXC_XCPLTEN_Jij.dat"
    # code = 'sprkkr'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # -- ORIGINAL Cr2Te3 TB2J
    # rows_to_skip = 30
    # fname_in = 'H:/2D/Cr2Te3/bulk/Wannier/no_SOC/2_wann/TB2J_results/Vampire/vampire.UCF'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # -- cropped Cr2Te3 TB2J
    # rows_to_skip = 30
    # fname_in = 'C:/Users/lv268562/Documents/PhD work/Scripts/ucf_crop_small_values_vampire/vampire.UCF_cropped_1088_0.100meV'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # -- IMITATING EXPERIMENTAL SPR-KKR 
    # for folder in ['Gr', 'WSe2', 'Bi2Te3']:
    #     rows_to_skip = 29
    #     fname_in = f"H:/2D/Cr2Te3/bulk/sprkkr_imitating_experimental/{folder}/POSCAR_JXC_XCPLTEN_Jij.dat"
    #     code = 'sprkkr'
    #     Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # # -- Wannier bcc Fe SOC TB2J 
    # rows_to_skip = 12
    # fname_in = 'H:/Wannier/1_bcc_Fe/1_SOC/3_vampire/TB2J_results/Vampire/vampire.UCF'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)


    # # # -- Wannier bcc Fe 60eV TB2J
    # rows_to_skip = 12
    # fname_in = 'H:/Wannier/1_bcc_Fe/0_sc_less_kpoints_NCORE=1/0_sc_less_kpoints_NCORE=1/2_NW18_Fe_spd/0_spd_frozmax_4_winmax_60_dis_500_num-iter_400/TB2J_results/Vampire/vampire.UCF'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # # # -- Wannier bcc Fe 30eV TB2J
    # rows_to_skip = 12
    # fname_in = 'H:/Wannier/1_bcc_Fe/0_sc_less_kpoints_NCORE=1/0_sc_less_kpoints_NCORE=1/2_NW18_Fe_spd/1_spd_frozmax_4_winmax_30_dis_500_num-iter_400/TB2J_results/Vampire/vampire.UCF'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # # # -- SPR-KKR bcc Fe
    # rows_to_skip = 10
    # fname_in = 'H:/SPR-KKR/bcc_Fe/Fe_JXC_XCPLTEN_Jij.dat'
    # code = 'sprkkr'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # # # -- SPR-KKR bcc Fe
    # rows_to_skip = 11
    # fname_in = 'H:/SPR-KKR/bcc_Fe/vampire.UCF'
    # code = 'tb2j_ucf'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)


    # # # -- Pajda 2001 Fe
    # rows_to_skip = 10
    # fname_in = "H:/test_exchange_Fe_Co_Ni/Fe/NL4/Fe_JXC_XCPLTEN_Jij.dat"
    # code = 'sprkkr'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)


    # # # -- Pajda 2001 Co
    # rows_to_skip = 10
    # fname_in = "H:/test_exchange_Fe_Co_Ni/Co/a_lit/NL3/Co_JXC_XCPLTEN_Jij.dat"
    # code = 'sprkkr'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    # # -- Pajda 2001 Ni
    # rows_to_skip = 10
    # fname_in = "H:/test_exchange_Fe_Co_Ni/Ni/a_lit/NL3/Ni_JXC_XCPLTEN_Jij.dat"
    # code = 'sprkkr'
    # Tc = mean_field_Tc(fname_in, code, rows_to_skip = rows_to_skip)

    components = ['Jxx', 'Jyy', 'Jzz']
    fname_in = 'vampire.UCF'
    code = 'tb2j_ucf'
    with open(f'Tc_anisotropic.dat', 'a+') as fw:
        fw.write('component\tT_C (K)\n')
    for component in components:
        code_info['tb2j_ucf']['target_column'] = component
        Tc = mean_field_Tc(fname_in, code, rows_to_skip = get_rows_to_skip_UCF(fname_in))
        with open(f'Tc_anisotropic.dat', 'a+') as fw:
            fw.write(f"{component}\t{Tc:.6f} #K\n")


if __name__ == '__main__':
    main()
