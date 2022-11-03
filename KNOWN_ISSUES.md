# GPU is not detected
On some machines, the FFT cannot be accelerated because the GPU is not detected. This is caused by the CUDA driver.

## Workaround
No known fix. Attempt updating your drivers or use the CPU FFT.

# Rare GPU FFT driver error
This error manifests itself as a `GPU Other(5)` error, i.e. a `CUFFT_INTERNAL_ERROR` from the driver. This is very rare and may be hardware / driver dependent.

## Workaround
No known fix. Rerun the program or use the CPU FFT.

# Data with invalid x values
The data is sorted and deduplicated to ensure a monotonic increase of the x variable. However, invalid x values, such as 0, will thus get sorted together in the same bin.

## Workaround
There is no way to distinguish invalid x values when also sorting the data. Typically, the bin at 0 does not matter that much. Otherwise, try impulse filtering as this can remove the invalid point.