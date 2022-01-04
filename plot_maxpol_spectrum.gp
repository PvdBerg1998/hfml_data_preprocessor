set terminal png size 1280,720
set datafile separator ','
set output 'maxpol_spectrum.png'

set xrange [0:pi]
set xtics ('i 0' 0, 'i π/4' pi/4, 'i π/2' pi/2, 'i 3π/4' 3*pi/4, 'i π' pi)
set title 'Operator spectra'

plot 'maxpol_spectrum.csv' using 1:2 with lines lw 2 title 'MaxPol',\
    'dv_spectrum.csv' using 1:2 with lines lw 2 title 'Exact derivative'