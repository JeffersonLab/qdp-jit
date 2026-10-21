# Download and build

The easiest way to build qdp-jit together with Chroma is the build package at
[github.com/fwinter/package](https://github.com/fwinter/package). Its scripts
download the sources of LLVM, QMP, qdp-jit and Chroma, and build and install
them in order, for one GPU architecture.

## Prerequisites

Before you start, make sure the following are available, on a cluster usually
by loading the corresponding modules:

- the GPU software stack: the CUDA toolkit (NVIDIA), ROCm (AMD) or oneAPI with
  Level Zero (Intel);
- an MPI installation with its compiler wrappers (`mpicxx`, `mpicc`);
- a C++ compiler, CMake, `git` and `wget`.

Get the package:

```bash
git clone https://github.com/fwinter/package.git
cd package
```

## 1. Choose the GPU architecture

Each architecture has its own download script and its own directory of build
scripts:

| GPU | Download script | Directory |
|---|---|---|
| NVIDIA | `download_sources_cuda.sh` | `cuda/` |
| AMD | `download_sources_rocm.sh` | `rocm/` |
| Intel | `download_sources_intel.sh` | `level-zero/` |

The examples below are for NVIDIA GPUs; replace `cuda` by the script and
directory of your architecture.

## 2. Download the sources

From the root of the package, run the download script:

```bash
./download_sources_cuda.sh
```

It clones QMP, qdp-jit and Chroma and downloads the LLVM sources, all into
`src/`. Unpacking LLVM takes a moment.

## 3. Configure the environment

Change to the directory of your architecture and edit `env.sh`:

```bash
cd cuda
```

`env.sh` is read by every build script. Set the compilers to your MPI
compiler wrappers:

```bash
export CXX=mpicxx
export CC=mpicc
```

and add anything else your system needs, such as `module load` lines or paths.
On some systems the wrappers have other names; on Cray systems, for example,
they are `CC` and `cc`.

The directory also holds `env_*.sh` files for particular machines (for example
`env_perlmutter.sh` in `cuda/`, `env_frontier.sh` in `rocm/`).
Use one of them as a starting point if your machine is among them.

The rest of `env.sh` sets where the build happens. Keep it unless you want
another location:

| Variable | Default | Contents |
|---|---|---|
| `SRCROOT` | `../src` | the sources downloaded in step 2 |
| `BUILDROOT` | `./build` | one build directory per package |
| `INSTALLROOT` | `./install` | one install directory per package |

`env.sh` also adds the LLVM libraries, `install/llvm-22/lib`, to
`LD_LIBRARY_PATH`. The package builds LLVM as shared libraries, so qdp-jit
programs such as `chroma` need this path at run time as well.

A job script can therefore source the same `env.sh` to set up everything:
the modules, the environment variables and the library search path. `env.sh`
takes its paths from the current directory, so change to the architecture
directory before sourcing it:

```bash
#!/bin/bash
#SBATCH ...

cd /path/to/package/cuda
source ./env.sh

cd /path/to/run
srun ${INSTALLROOT}/chroma/bin/chroma -i input.xml -geom 1 1 1 4
```

## 4. Check the host CPU architecture

Besides the GPU target, LLVM must be built for the CPU of the machine qdp-jit
will run on: qdp-jit compiles expressions on scalars for the host CPU (the
host JIT, on by default). Before building, check whether the compute nodes
have x86 or ARM CPUs, and set `LLVM_TARGETS_TO_BUILD` in `build_llvm-22.sh`
accordingly:

| GPU | x86 host | ARM host |
|---|---|---|
| NVIDIA | `"NVPTX;X86"` | `"NVPTX;AArch64"` |
| AMD | `"AMDGPU;X86"` | `"AMDGPU;AArch64"` |
| Intel | `"X86"` | `"AArch64"` |

The scripts are set up for x86 hosts. For a machine with ARM CPUs, such as
NVIDIA Grace, change `X86` to `AArch64`. Intel GPUs need no GPU target in
LLVM, because their kernels are translated to SPIR-V by a separate tool.

## 5. Build

```bash
./build_all.sh
```

`build_all.sh` runs the build scripts in order: LLVM, QMP, qdp-jit
(`build_qdpxx.sh`) and Chroma (`build_chroma.sh`); in `level-zero/` it also
builds the SPIR-V translator after LLVM. Building LLVM takes the
longest. When it has finished, `install/` holds:

| Directory | Contents |
|---|---|
| `install/llvm-*` | LLVM |
| `install/qmp` | QMP |
| `install/qdpxx` | qdp-jit |
| `install/chroma` | Chroma, with the executable `install/chroma/bin/chroma` |

## Rebuilding a part

Each package has its own build script, which can be run on its own, from the
architecture directory. A script removes its build directory first, so each
run is a clean build.

After a change to qdp-jit, rebuild qdp-jit and then Chroma, which is compiled
against qdp-jit's headers:

```bash
./build_qdpxx.sh
./build_chroma.sh
```

## Build options of qdp-jit

`build_qdpxx.sh` configures qdp-jit with CMake. To change an option, edit the
`cmake` command in that script. The most important options:

| Option | Values | Description |
|---|---|---|
| `QDP_ENABLE_BACKEND` | `CUDA`, `ROCM`, `L0` | the GPU backend |
| `QDP_PRECISION` | `single`, `double` (default) | the precision of `Real`, `LatticeFermion`, … |
| `QDP_LAYOUT` | `cb2` (default), `lexico`, `cb3d` | the site layout, see [Layouts](layouts.md#each-layout-must-fit-the-node-grid) |
| `QDP_ND`, `QDP_NC`, `QDP_NS` | 4, 3, 4 (default) | the number of dimensions, and the initial current shape (Ns, Nc), see [Shape](shape.md) |
| `QDP_ENABLE_HOST_COMPILER` | `ON` (default), `OFF` | also JIT-compile scalar expressions for the CPU; needs the host CPU target in LLVM, see [step 4](#4-check-the-host-cpu-architecture) |
| `LLVM_DIR` | a path | the LLVM installation to use; qdp-jit needs LLVM 22 or newer. With the ROCm backend it defaults to the LLVM that comes with ROCm |

## Using qdp-jit in your own program

qdp-jit installs a CMake package configuration. A CMake project needs C and
C++ enabled (QMP looks for MPI's C interface); linking `QDPXX::qdp` compiles it
as C++20:

```cmake
project(my_program C CXX)

find_package(QDPXX REQUIRED)
target_link_libraries(my_program QDPXX::qdp)
```

Configure it with the locations of qdp-jit, QMP and LLVM, as `build_chroma.sh`
does for Chroma:

```bash
cmake <source> \
    -DQDPXX_DIR=<package>/cuda/install/qdpxx/lib/cmake/QDPXX \
    -DQMP_DIR=<package>/cuda/install/qmp/lib/cmake/QMP \
    -DLLVM_DIR=<package>/cuda/install/llvm-<version>/lib/cmake/llvm
```

To run on several GPUs, start one MPI rank per GPU and give the grid of ranks
with `-geom`; see [Command-line options](options.md#ranks-and-devices).
