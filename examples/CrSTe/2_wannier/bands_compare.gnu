# ---- gnuplot script to compare DFT- and Wannier- calculated bandstructures ----

set xlabel "K-Path"
set ylabel "E - E_F (eV)"

# set the fermi energy (from DOSCAR)  AND   copy wannier90.dn_band.gnu here and replace "dots" with "lines"
# ==============================================================================================
FERMI = -2.31796034

set yrange [ -6.0 : 11.0]
set xrange [0: 3.68571]
set arrow from  0.99995,  -6 to  0.99995,   11 nohead
set arrow from  2.16041,  -6 to  2.16041,   11 nohead
set xtics ("M"  0.00000,"G"  0.99995,"K"  2.16041,"M"  3.68571)
# ===============================================================================================

plot "wannier90_band.dat" u 1:($2-FERMI) w l lw 2 lc rgb "black" title "wannier", "../bands/BAND.dat" u 1:2 lc rgb "green" pt 6 title "DFT"

