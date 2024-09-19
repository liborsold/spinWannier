set style data dots
set nokey
set xrange [0: 2.89129]
set yrange [ -9.73245 :  2.39556]
set arrow from  1.22200,  -9.73245 to  1.22200,   2.39556 nohead
set arrow from  2.28029,  -9.73245 to  2.28029,   2.39556 nohead
set xtics ("K"  0.00000,"G"  1.22200,"M"  2.28029,"K"  2.89129)
 plot "wannier90_band.dat"
