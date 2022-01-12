set terminal png size 1920,1080
set datafile separator ','
set output 'out.png'
plot 'Vxy.csv' using 1:2 with lines, 'processed.csv' using 1:2 with lines axes x1y1
