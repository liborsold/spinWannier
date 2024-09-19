set arrow from       1.22200133,     -9.73244978 to       1.22200133,      2.39555876 nohead
set arrow from       2.28028658,     -9.73244978 to       2.28028658,      2.39555876 nohead
unset key
set xrange [0:  2.89129]
set yrange [     -9.73244978 :      2.39555876]
set xtics (" K "  0.00000," G "  1.22200," M "  2.28029," K "  2.89129)
 set palette defined (-1 "blue", 0 "green", 1 "red")
 set pm3d map
 set zrange [-1:1]
 splot "wannier90-bands.dat" with dots palette
