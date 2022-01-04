set terminal png size 1280,720
set datafile separator ','
set output 'maxpol_coeff.png'

set key off
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 pi -1 ps 0.5
set pointintervalbox 1.0

plot 'maxpol_coeff.csv' using 1:2 with lp ls 1