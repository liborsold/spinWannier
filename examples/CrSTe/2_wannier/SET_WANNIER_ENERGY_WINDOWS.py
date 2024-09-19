import string
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import wannier_energy_windows, eigenval_for_kpoint, band_with_spin_projection_under_threshold_for_kpoint, magmom_direction, band_with_spin_projection_under_threshold_for_kpoint
sys.path.append('/home/lv268562/bin/python_scripts')
from get_fermi import get_fermi
import numpy as np


# =========== USER-DEFINED ====================

wann_bands_lims = [0, 21]

dis_win_min = None #-6.0	# eV; relative to E_F; put None to have the 'tight' value
dis_win_max = None #5.0	# eV; relative to E_F; put None to have the 'tight' value
dis_froz_min = None #-0.5	# eV; relative to E_F; put None to have the 'tight' value
dis_froz_max = None #0.5	# eV; relative to E_F; put None to have the 'tight' value

SOC_nsc_calc_folder = "../sc"

# =============================================


# get Fermi energy
E_F = get_fermi(path=SOC_nsc_calc_folder)

# get tight energy windows
dis_win_min_tight, dis_win_max_tight, dis_froz_min_tight, dis_froz_max_tight = wannier_energy_windows(wann_bands_lims=wann_bands_lims, eigenval_file="wannier90.eig")

# !! LIMIT THE MAX OF FROZEN WINDOW BY BAND NUMBER 20 (0-indexed) - i.e., the 2nd band from top !!
# specific to CrXY, sometimes it crosses the other bands at Gamma

#   not good
# spin_direction = magmom_direction('INCAR')
# print('spin_direction', spin_direction)
# # get the number of the band which has a low spin projection (below 'threshold'; in direction 'spin_direction' - same as the current calculation) at Gamma point (0.0, 0.0, 0.0); skip the 10 lowest bands which are not wannierized 
# band_with_low_spin_character = band_with_spin_projection_under_threshold_for_kpoint(kpoint=(0.0, 0.0, 0.0), spin_direction=spin_direction, threshold=0.250, PROCAR_file="../bands/PROCAR", n_ions=3, skip_lowest_bands=10)

band_with_low_Cr_d_XY_p_character = band_with_spin_projection_under_threshold_for_kpoint(kpoint=(0.0, 0.0, 0.0), orbital_characters_considered={0:[4,5,6,7,8], 1:[1,2,3], 2:[1,2,3]}, threshold=0.250, PROCAR_file="../bands/PROCAR", n_ions=3, skip_lowest_bands=10)
crossing_high_energy_band_at_Gamma = eigenval_for_kpoint(kpoint=(0.0, 0.0, 0.0), band=band_with_low_Cr_d_XY_p_character, eigenval_file="wannier90.eig", win_file="wannier90.win")
print('crossing high energy band at Gamma', crossing_high_energy_band_at_Gamma)
dis_froz_max_tight = min(dis_froz_max_tight, crossing_high_energy_band_at_Gamma)

# set the desired energy windows
 	 # make wannierization windows larger   
	 # make frozen windows tighter ! otherwise more bands than the number of wann. bands could fall within the frozen window, which results in error

# add small margin to values
margin = 0.200 # 0.3 # 0.2 # 0.1 # 0.05 # 0.03 # 0.01 # 0.001 # 0.0001 # 1e-4
margin_small = 0.0001 #1e-4

dis_froz_min = E_F + dis_froz_min if dis_froz_min is not None else dis_froz_min_tight + margin_small
dis_froz_max = E_F + dis_froz_max if dis_froz_max is not None else dis_froz_max_tight - margin
dis_win_min = E_F + dis_win_min if dis_win_min is not None else dis_win_min_tight - margin_small
dis_win_max = E_F + dis_win_max if dis_win_max is not None else dis_win_max_tight + margin_small

# files to change
files_to_change = ['wannier90.win', 'INCAR']

# replace the values
replace_what = ["dis_win_min=", "dis_win_max=", "dis_froz_min=", "dis_froz_max="]
replace_with = [f"dis_win_min={dis_win_min:.6f} #", f"dis_win_max={dis_win_max:.6f} #", f"dis_froz_min={dis_froz_min:.6f} #", f"dis_froz_max={dis_froz_max:.6f} #", ]

for file in files_to_change:
    with open(file, 'r') as fr:
        string_in = fr.read()

    for rep_what, rep_with in zip(replace_what, replace_with):
        string_in = string_in.replace(rep_what, rep_with)

    with open(file, 'w') as fw:
        fw.write(string_in)
