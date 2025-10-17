# Fast numerics minimal playbook!
A tutorial on SIMD (Single Instruction, Multiple Data) programming in C++ and Python, featuring Chebyshev polynomial interpolation as a practical example.

The python tutorial and examples are available on binder while the c++ code is on compiler explorer:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiamonDinoia/fwam2025/HEAD?urlpath=tree%2Fpython)
[![Compiler Explorer](https://img.shields.io/badge/Compiler%20Explorer-open-blue?style=flat-square)](https://marco.godbolt.org/z/cbvo9o7dT)

# For the FI crowd

The Python binder code is available at: https://sdsc-binder.flatironinstitute.org/~mbarbone/fwam2025


## Overview

This repository contains educational materials demonstrating SIMD optimization techniques for numerical computing. The tutorial covers:

- **SIMD fundamentals** and when to use them
- **Performance benchmarking** using nanobench
- **Portable SIMD programming** with xsimd
- **Practical example**: Chebyshev polynomial interpolation
- **Python notebooks** with interactive exercises

## Contents

### C++ Implementation
- `main.cpp` - Benchmarking and testing code for Chebyshev interpolation
- `include/cheb.h` - Chebyshev interpolation implementations (scalar and SIMD)
- `CMakeLists.txt` - Build configuration with xsimd and nanobench

### Python Notebooks
- `python/presentation.ipynb` - Tutorial presentation slides
- `python/exercise.ipynb` - Hands-on exercises
- `python/solution.ipynb` - Exercise solutions

## Prerequisites

### C++ Build
- CMake 3.10+
- C++17 compatible compiler
- CPU with SIMD support (AVX2, AVX512, or similar)

### Python
- Python 3.x
- Jupyter Notebook
- Dependencies in `requirements.txt`

## Installation

### Option 1: Using pip (if you have a modern C++ compiler)

If you already have a working modern C++ compiler:

```bash
# Create a virtual environment that inherits system packages
python3 -m venv --system-site-packages venv

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

### Option 2: Using Conda (recommended for complete environment)

#### 1. Install Miniconda

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts and restart your shell or run:
source ~/.bashrc
```

#### 2. Create the environment from the YAML file

```bash
# Create environment from local_environment.yml
conda env create -f local_environment.yml

# Activate the environment
conda activate fwam2025
```

#### 3. Launch Jupyter

```bash
# Launch JupyterLab
jupyter lab

# Or launch classic Notebook
jupyter notebook
```

## Building and Running

### C++ Version

```bash
# Create build directory
mkdir -p build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .

# Run
./simd101

### Python Version

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook python/
```

## Dependencies

The project automatically fetches:
- **xsimd**: Portable SIMD wrapper library
- **nanobench**: High-precision benchmarking library
