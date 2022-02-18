# HFML Data Preprocessor
This tool is meant to quickly extract, inspect, preprocess and Fourier transform data generated at the HFML, Radboud University Nijmegen.

## Buzzwords
- State-of-the-art parsers and formatters ([some](https://arxiv.org/abs/2101.11408) [literature](https://dl.acm.org/doi/10.1145/3192366.3192369))
- Optional multithreading and NVIDIA GPU support
- Full data and settings verification
- Autovectorisation through LLVM and rustc
- Insane runtime performance: ~500 ms for a full dataset

## Capabilities
### Preprocessing
- Data x/y pair extraction
- Header renaming
- Domain trimming and data masking
- Premultiplication
- Impulse filtering for "popcorn noise" removal
- x inversion for 1/B periodic processes
- Linear and Steffen (monotonic) spline interpolation
- First and second numerical derivative

### Output
- Data output at each intermediate step in either CSV or MessagePack binary format
- Quick 'n dirty plotting for manual inspection

### FFT
- DC component removal
- Zero padding
- Boxcar or Hann windowing
- Frequency domain trimming
- FTFT (Finite Time Fourier Transform), subdomain sweeping

# Usage
### Settings format
The tool can be used by specifying required and optional settings via a copy of the `settings.toml` file. The format allows for specialisation of settings at each of the following levels:
- Global
- Per variable
- Per file
- Per file, per variable

The `settings.toml` template contains documentation explaining all possible options and details.

### Running
The tool is compiled from source, unlike Python scripts. This allows for much better runtime performance and optimisations. If one has obtained a compatible compiled binary, usage is simple:
    
    hfml_data_preprocessor <SETTINGS FILE>

The files listed in the given settings file are assumed to be present in the working directory. It is therefore recommended to add the tool to your path and execute it in the folder containing your data files.

Further settings are documented through the runtime help option:

    hfml_data_preprocessor --help

When compiled with [NVIDIA CUDA](https://developer.nvidia.com/cufft) support, the proprietary runtime libraries must be reachable on your device. If this fails, the tool will not start and may crash without an error message.

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

## GSL wrapper

Make sure the [`gsl_rust`](https://github.com/PvdBerg1998/gsl_rust) wrapper library is present in the folder containing this repository. Example structure:

    GitHub/gsl_rust
    GitHub/hfml_data_preprocessor

The `gsl_rust` crate gets built automatically but requires you to initialize the `GSL` submodule through Git:

```shell
git clone git@github.com:PvdBerg1998/gsl_rust.git
cd gsl_rust
git submodule init
git submodule update
```

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