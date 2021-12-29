set terminal png size 1920,1080
set datafile separator ","
set output 'out.png'
set multiplot layout 3,1

set xrange [5:33]

plot 'out_interp.csv' using 1:2 with lines, 'out.csv' using 1:2 with lines
plot 'out_dv1.csv' using 1:2 with lines, 'out_dv2.csv' using 1:2 with lines axes x1y2
plot 'out_residuals.csv' using 1:2 with lines
