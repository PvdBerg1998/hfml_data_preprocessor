set terminal png size 1920,1080
set datafile separator ','
set output 'out.png'
set multiplot layout 3,1

set xrange [0.03:0.2]

plot 'raw.csv' using 1:2 with lines, 'full_fit.csv' using 1:2 with lines axes x1y1
plot 'mr_dv.csv' using 1:2 with lines, 'full_dv.csv' using 1:2 with lines axes x1y1
plot 'mr_residuals.csv' using 1:2 with lines, 'full_residuals.csv' using 1:2 with lines axes x1y1

unset multiplot
set terminal png size 7680,4320
set output 'detail.png'
plot 'raw.csv' using 1:2 with points, 'full_fit.csv' using 1:2 with lines axes x1y1
