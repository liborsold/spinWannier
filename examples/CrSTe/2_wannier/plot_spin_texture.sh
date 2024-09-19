
# Sx
sed -i "s/spin_axis_polar = /spin_axis_polar = 90 #/g" wannier90.win
sed -i "s/spin_axis_azimuth = /spin_axis_azimuth = 0 #/g" wannier90.win
mpirun -np 32 postw90.x wannier90
mv wannier90-bands.dat wannier90-bands_Sx.dat

# Sy
sed -i "s/spin_axis_polar = /spin_axis_polar = 90 #/g" wannier90.win
sed -i "s/spin_axis_azimuth = /spin_axis_azimuth = 90 #/g" wannier90.win
mpirun -np 32 postw90.x wannier90
mv wannier90-bands.dat wannier90-bands_Sy.dat

# Sz
sed -i "s/spin_axis_polar = /spin_axis_polar = 0 #/g" wannier90.win
sed -i "s/spin_axis_azimuth = /spin_axis_azimuth = 0 #/g" wannier90.win
mpirun -np 32 postw90.x wannier90
mv wannier90-bands.dat wannier90-bands_Sz.dat

python wannier90-bands_Sxyz.py
