# qdp-jit

qdp-jit is an implementation of the [QDP++](https://github.com/usqcd-software/qdpxx)
data-parallel API for lattice QCD that runs on GPUs. Programs written for
QDP++, such as [Chroma](https://github.com/JeffersonLab/chroma), build against
qdp-jit. For every expression, qdp-jit generates a kernel at run time with
LLVM and runs it on the GPU.

- **GPU backends:** NVIDIA (CUDA), AMD (ROCm) and Intel (Level Zero).
- **Several lattices in one program:** a fine lattice and the coarse lattices
  of a multigrid hierarchy, made by blocking the fine one.
- **Spin and color extents at run time:** Ns and Nc are carried by each field,
  so coarse-grid fields with, say, 2 spins and 24 "colors" use the ordinary
  types (`LatticeFermion`, `LatticeColorMatrix`, …).
- **Half-precision storage:** fields can be stored in 16-bit floating point
  and used in ordinary expressions.
- **Multigrid operations:** `blockSum` for restriction, and expressions that
  mix fine and coarse fields for prolongation.
- **Host JIT:** expressions on scalars are compiled for the CPU (on by default).

## Documentation

The user documentation is at <https://jeffersonlab.github.io/qdp-jit/>. It assumes
you know QDP++ and describes what qdp-jit adds.


## Building qdp-jit and Chroma

The easiest way is the build package at
[github.com/fwinter/package](https://github.com/fwinter/package). It downloads
and builds LLVM, QMP, qdp-jit and Chroma for one GPU architecture:

```bash
git clone https://github.com/fwinter/package.git
cd package
./download_sources_cuda.sh       # or _rocm.sh, _intel.sh
cd cuda                          # or rocm, level-zero
# edit env.sh: compilers (mpicxx, mpicc), modules
# on ARM hosts: LLVM_TARGETS_TO_BUILD in build_llvm-22.sh, X86 -> AArch64
./build_all.sh
```

The download script fetches the LLVM sources, and LLVM is built for the GPU and
the host CPU (`NVPTX;X86` for CUDA, `AMDGPU;X86` for ROCm, `X86` for Level Zero).
LLVM is installed as shared libraries, and `env.sh` adds them to
`LD_LIBRARY_PATH`. A job script can therefore set up the run-time environment
by sourcing the same file, after changing to the architecture directory:

```bash
cd /path/to/package/cuda
source ./env.sh
cd /path/to/run
srun ${INSTALLROOT}/chroma/bin/chroma -i input.xml -geom 1 1 1 4
```

See [Download and build](https://jeffersonlab.github.io/qdp-jit/building/) for
the details.

## Building qdp-jit by hand

### Requirements

- CMake 3.17 or newer and a C++20 compiler, used through the MPI compiler
  wrappers
- [LLVM](https://llvm.org) 22 or newer (qdp-jit is tested with LLVM 22). With
  the ROCm backend, the LLVM that comes with ROCm is used by default
- [QMP](https://github.com/usqcd-software/qmp), built with MPI
- the GPU software: the CUDA toolkit, ROCm, or oneAPI with Level Zero
- libxml2

### Configure, build and install

```bash
mkdir build && cd build
CXX=mpicxx CC=mpicc cmake /path/to/qdp-jit \
    -DQDP_ENABLE_BACKEND=CUDA \
    -DQDP_PRECISION=double \
    -DQMP_DIR=/path/to/qmp/lib/cmake/QMP \
    -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
    -DCMAKE_INSTALL_PREFIX=/path/to/install/qdp-jit
make -j 8
make install
```

### CMake options

| Option | Values | Description |
|---|---|---|
| `QDP_ENABLE_BACKEND` | `CUDA` (default), `ROCM`, `L0` | the GPU backend |
| `QDP_PRECISION` | `single`, `double` (default) | the precision of `Real`, `LatticeFermion`, … |
| `QDP_LAYOUT` | `cb2` (default), `lexico`, `cb3d` | the site layout |
| `QDP_ND`, `QDP_NC`, `QDP_NS` | 4, 3, 4 (default) | the number of dimensions, and the initial Nc and Ns |
| `QDP_ENABLE_HOST_COMPILER` | `ON` (default), `OFF` | also JIT-compile scalar expressions for the CPU; LLVM must include the host CPU target (`X86` or `AArch64`) |
| `QDP_BUILD_EXAMPLES` | `ON` (default), `OFF` | build the programs in `examples/` |
| `QDP_USE_HDF5` | `ON`, `OFF` (default) | HDF5 I/O |
| `LLVM_DIR` | a path | the LLVM installation to use |

## Using qdp-jit in your own program

qdp-jit installs a CMake package. A program needs C and C++ enabled (QMP
looks for MPI's C interface); linking `QDPXX::qdp` compiles it as C++20:

```cmake
cmake_minimum_required(VERSION 3.17)
project(hello C CXX)

find_package(QDPXX REQUIRED)

add_executable(hello hello.cc)
target_link_libraries(hello QDPXX::qdp)
```

Configure it with the locations of qdp-jit, QMP and LLVM:

```bash
CXX=mpicxx CC=mpicc cmake /path/to/hello \
    -DQDPXX_DIR=/path/to/install/qdp-jit/lib/cmake/QDPXX \
    -DQMP_DIR=/path/to/qmp/lib/cmake/QMP \
    -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

A minimal program:

```c++
#include "qdp.h"

using namespace QDP;

int main(int argc, char** argv)
{
  QDP_initialize(&argc, &argv);

  multi1d<int> nrow(Nd);
  nrow = 8;
  Layout::setLattSize(nrow);
  Layout::create();

  LatticeFermion psi, chi;
  LatticeColorMatrix u;
  gaussian(psi);
  gaussian(u);

  chi = u * psi;        // one JIT-compiled kernel on the GPU
  QDPIO::cout << "norm2(chi) = " << norm2(chi) << std::endl;

  QDP_finalize();
}
```

## Running

Start one MPI rank per GPU and give the grid of ranks with `-geom`:

```bash
mpirun -np 4 ./my_program -geom 1 1 2 2
```

At start-up qdp-jit allocates 92% of the free GPU memory as a memory pool.
When several ranks share one GPU, give each a smaller pool with `-poolsize`,
for example `-poolsize 5g`.

With a GPU-aware MPI, add `-gpudirect`: MPI then sends and receives directly
from device memory, without copies through host memory. The communication
buffers are allocated on the device outside the memory pool, so by default 10%
of the free GPU memory is kept outside the pool for them. `-pool-reserve SIZE`
sets that amount instead, for example `-pool-reserve 2g`. A `-poolsize` that
would leave less than that is reduced.

For correct times with `StopWatch`, add `-timing-run`. All options are listed
in
[Command-line options](https://jeffersonlab.github.io/qdp-jit/options/).

