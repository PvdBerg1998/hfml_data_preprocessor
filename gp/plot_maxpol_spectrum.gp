set terminal png size 1280,720
set datafile separator ','
set output 'maxpol_spectrum.png'

set title 'Operator spectra'
set key top left
set xrange [0:pi]
set grid xtics mxtics
set xtics pi/8
set format x '%.1Pπ'
set mxtics 2
unset ytics
set grid

plot 'maxpol_spectrum.csv' using 1:2 with lines lw 2 title 'MaxPol',\
    'dv_spectrum.csv' using 1:2 with lines lw 2 title 'Exact derivative'