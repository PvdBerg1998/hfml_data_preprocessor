set terminal png size 1920,1080
set datafile separator ','
set output 'out.png'
set multiplot layout 4,1

plot 'raw.csv' using 1:2 with lines, 'spline_fit.csv' using 1:2 with lines axes x1y1
plot 'maxpol_dv.csv' using 1:2 with lines, 'spline_dv.csv' using 1:2 with lines axes x1y2
plot 'spline_residuals.csv' using 1:2 with lines axes x1y1

set xrange [1:500]
plot 'maxpol_fft.csv' using 1 with lines, 'spline_fft.csv' using 1 with lines axes x1y2
