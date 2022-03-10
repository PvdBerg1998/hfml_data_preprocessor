# HFML Data Preprocessor
This tool is meant to quickly extract, inspect, preprocess and Fourier transform data generated at the HFML, Radboud University Nijmegen. While designed for this specific purpose, the process is very general and can be adapted to other use cases.

## Buzzwords
- State-of-the-art parsers and formatters ([some](https://arxiv.org/abs/2101.11408) [literature](https://dl.acm.org/doi/10.1145/3192366.3192369))
- Optional multithreading and NVIDIA GPU support with automatic batching
- Full data and settings verification
- Autovectorisation through LLVM and rustc
- Insane runtime performance: ~500 ms per project

## Capabilities
### Preprocessing
- Data x/y pair extraction
- Header renaming
- Domain trimming
- Bad data masking
- Premultiplication
- Tuneable impulse filtering for "popcorn noise" removal
- x inversion for 1/B periodic processes
- Linear interpolation and Steffen (monotonic) spline interpolation with automatic length detection and coordination to ensure uniformity in your dataset
- First and second numerical derivative

### Realistic performance running a full analysis
For this measurement, I ran a full analysis on 5 samples measured in low field as well as one high field week. The dataset counts a total of 211 data files for a total of 170 MB of raw data. The test system was running Manjaro and consists of an Intel i5 6600K @ 4.4GHz, 16GB DDR4, GTX 1070, Samsung 980 Pro.

```
- 1 core   GPU disabled  : 44 sec (1.0x)
- 4 cores  GPU disabled  : 27 sec (1.6x)
- 1 core   GPU enabled   : 11 sec (4.0x)
- 4 cores  GPU enabled   : 6 sec  (7.3x)
```

These numbers vary strongly depending on your analysis parameters. Especially for more extreme settings, multithreading and GPU FFT will provide larger benefits. Data output and intermediate plot generation is a major performance bottleneck. At debug logging level, performance measurements are displayed.

### Output
- Data output at each intermediate step in either CSV or [MessagePack](https://msgpack.org/index.html) binary format
- Quick 'n dirty plotting for manual inspection without external tools
- Detailed metadata stored as JSON for easy scripted postprocessing
- Full "trace-level" log files for debugging and manual checks

MessagePack is recommended because it skips converting the floating point data back and forth to a base 10 representation. This saves processing power as well as potential accuracy loss. For usage in Python, I recommend [ormsgpack](https://pypi.org/project/ormsgpack/) to decode the data. An example is detailed further down.

### FFT
- DC component removal
- Zero padding
- Boxcar or Hann windowing
- Frequency domain trimming
- FTFT (Finite Time Fourier Transform), subdomain lower/upper/center uniform sweeping

If enabled, the FFT is accelerated using the first available NVIDIA GPU. For realistic datasets, the speedup is often around 5x.

# Known limitations
- Other filtering methods such as Savitzky-Golay are missing
    - Most commonly required for visualisation, can easily be handled during post processing with e.g. SciPy.
- No support for AMD GPU's
    - Could be implemented via OpenCL or VkFFT
- No check for wrong specialisation targets
    - No method known to do this without writing a full TOML parser

# Usage
### Settings format
The tool can be used by specifying required and optional settings via a copy of the `Settings.toml` file. The file format is called [TOML](https://toml.io/en/). The format allows for specialisation of settings at each of the following levels:
- Global
- Per variable
- Per file
- Per file, per variable

The `Settings.toml` template contains documentation explaining all possible options and details.
Note that while most settings are checked for validity, it is currently not feasible to check specialisation targets. This means that you should take care when changing your output destinations if you are using specialisations.

### Running
The tool is compiled from source, unlike Python scripts. This allows for much better runtime performance and optimisations. If one has obtained a compatible compiled binary, usage is simple:
    
    hfml_data_preprocessor <SETTINGS FILE>

The files listed in the given settings file are assumed to be present in the working directory. It is therefore recommended to add the tool to your path and execute it in the folder containing your data files.

Further settings are documented through the runtime help option:

    hfml_data_preprocessor --help

When compiled with [NVIDIA CUDA](https://developer.nvidia.com/cufft) support, the proprietary runtime libraries must be reachable on your device. If this fails, the tool will not start and may crash without an error message.

### Output structure

The tool generates nested output folders, separating the data generated at each step of the process. Your typical project folder will look something like this:

    output/<project name>/<variable>
        /raw
        /preprocessed
        /inverted
        /post_interpolation
        /fft
        /fft_sweep_<upper,lower,window>_<n>
        
    output/<project name>/metadata.json
    
    log/
        hfml_data_preprocessor_<unix time>.log
        ...

    my_settings.toml
    file.001.dat
    file.002.dat
    ...

The data is processed as follows and in the following order:
- `raw`: only sorted and deduplicated, such that the x data is monotonically increasing. This is often required for filtering algorithms. All steps after this will stay monotonic.
- `preprocessed`: after masking, filtering, premultiplication. Essentially the full preprocessing machinery except x inversion, as this makes visual comparison with the raw data near impossible.
- `inverted`: after x inversion.
- `post_interpolation`: after interpolation and derivative. For performance and simplicity, these two steps are implemented as a single mathematical operation. Linear interpolation therefore defines the second derivative to be zero everywhere. No filtering is applied, so keep in mind high frequency noise will be amplified by taking the derivative.
- `fft`: after the Fast Fourier Transform (full domain).
- `fft_sweep_<upper,lower,window>_<n>`: after the `n`'th FTFT window. These windows are distributed uniformly over the given data domain, after x inversion.

The raw and preprocessed data may include preliminary plot generation. This facilitates the quick iteration process of changing settings, rerunning and inspecting the result.

Files inside the output folders are named as you defined inside the settings. This may include subfolders.

Information about the generated file structure is stored in `metadata.json` in human readable `json`. This is meant to simplify further postprocessing by providing information about every generated file. Additional measurement metadata can be defined in `Settings.toml` per file. A rich amount of information is provided. For reasons of brevity the structure is undocumented, but should be straightforward. This may include:
- Dynamic interpolation length statistics
- Settings for each processing step
- Full output file paths
- FTFT subdomains
- User defined tags

An example usecase: imagine you want to plot the FFTs for each extracted variable measured at an angle of zero degrees. Simply load the `metadata.json` and filter it appropriately:

```python
import os
import json
import ormsgpack
import numpy as np
import pandas as pd
from functional import seq

sample = "MyMaterialStudy123"

# Loads a messagepack encoded output file
def load_msgpack(path):
    # Be sure to read the file as binary
    with open(os.path.join(f"../{sample}/", path), "rb") as f:
        xy = np.array(ormsgpack.unpackb(f.read()))
        xy = np.transpose(xy)
        # These column names are arbitrary,
        # but its useful to be consistent
        df = pd.DataFrame(data=xy, columns=["x", "y"])
        return df

# Loads the JSON metadata
def metadata(project):
    path = f"../{sample}/output/{project}/metadata.json"
    with open(path, "rt") as f:
        return json.loads(f.read())

#
# This was all you really need!
# The following code is just an example of metadata filtering.
#

# Extracts the output section from the metadata
def output_seq(project):
    return seq(metadata(project)["output"])

# Select only the FFTs with tag angle=0,
# then load the data bundled with the variable name.
data = output_seq("MyProject")\
    .filter(lambda out: out["stage"] == "fft")\
    .filter(lambda out: out["metadata"]["tags"]["angle"] == 0)\
    .map(lambda out: (out["variable"], load_msgpack(out["path"])))\
    .to_list()

# Generate plots through matplotlib
for variable, fft in data:
    # Limiting the frequencies to [0,1000]
    fft[fft.x.between(0, 1000)].plot(x="x", y="y", title=variable)
```

# Compiling
The tool is written in Rust. A working Rust compiler is therefore required. The stable toolchain is sufficient. For installation, follow [these steps](https://www.rust-lang.org/tools/install). This may require you to install additional build tools, documented [here](https://rust-lang.github.io/rustup/installation/windows.html).

The tool itself relies on two libraries:
- [GNU Scientific Library](https://www.gnu.org/software/gsl/) (GSL) for general purpose math
- [CUDA toolkit](https://developer.nvidia.com/cufft) for GPU FFT acceleration

The GSL is bundled and compiled automatically. This requires a C/C++ compiler to be present, as well as the build tool `cmake`. These compilers will most likely already be available once you have a working Rust installation. The NVIDIA CUDA libraries are proprietary and must therefore be installed manually.

## Requirements

Linux package managers make the compilation almost trivial and it is therefore recommended to use a Linux installation instead of Windows. The following commands should install all requirements:

Ubuntu:

```shell
sudo apt install build-essential cmake nvidia-cuda-toolkit pkg-config
```

Arch Linux:

```shell
sudo pacman -S base-devel cmake cuda pkg-config
```

For Windows these steps must be done manually through provided installers. It is recommended to install the MSVC flavour of Rust tooling. This does NOT require a full installation of Visual Studio.

Note that Windows 10 and Windows 11 support an embedded Linux installation named [WSL](https://docs.microsoft.com/en-us/windows/wsl/install). This provides the convenience of Linux without requiring a separate bootable installation. Running through WSL may require [additional steps](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to provide GPU support.

On Linux, the installation location of the CUDA libraries is determined through `pkg-config`. If this fails, the location is guessed to be `/usr/local/cuda`. On Windows, the `CUDA_PATH` environment variable must be set to the folder containing both `lib` and `include`.

## Cargo

If all is ready, the tool can then be compiled through Cargo, the build manager of Rust:

```shell
cd hfml_data_preprocessor
cargo build --release
```

If no NVIDIA hardware is present or if you do not manage to perform the installation, a compile time flag can be set to disable GPU capability:

```shell
cargo build --release --no-default-features
```

If compilation is successful, the binaries are placed inside the `target/release` folder.

# Licensing
This repository is licensed under the GNU General Public License version 3. This is required because the tool uses the GNU Scientific Library and makes sure everyone has free access to this source code.
