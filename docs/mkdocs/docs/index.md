# qdp-jit

qdp-jit is an implementation of the [QDP++](https://github.com/usqcd-software/qdpxx)
data-parallel API for lattice QCD. Programs written for QDP++, such as Chroma,
build against qdp-jit. The difference is in how expressions are evaluated: qdp-jit generates
a kernel for each expression at run time with LLVM and runs it on a GPU.

- **GPU backends:** CUDA, ROCm and Intel Level Zero. The backend is chosen when
  qdp-jit is configured (`QDP_ENABLE_BACKEND=CUDA|ROCM|L0`).
- **Host JIT:** when built with `QDP_ENABLE_HOST_COMPILER=ON`, expressions on
  scalar (`OScalar`) values can also be JIT-compiled for the host CPU.

These pages assume you know QDP++. They describe what qdp-jit adds to QDP++
and to earlier qdp-jit versions, and point out where it behaves differently.
To build qdp-jit and Chroma, see [Download and build](building.md). Then
start with [Concepts](concepts.md), which defines the terms the other pages use.

## What is new

### Several lattices in one program

QDP++ has one lattice, fixed at `QDP_initialize`. qdp-jit can hold any number
of lattice layouts at the same time, typically a fine lattice and the coarse
lattices of a multigrid hierarchy. Coarse layouts are made by blocking an
existing one, and a stack of current layouts decides where new fields live.
See [Layouts](layouts.md) and [Scopes](scopes.md).

### Spin and color extents at run time

In QDP++ the number of spin components Ns and of colors Nc are compile-time
constants. In qdp-jit they are run-time values, carried by each field as its
[shape](shape.md). A coarse-grid field with, say, 2 spin components and 24
"colors" uses the same types (`LatticeFermion`, `LatticeColorMatrix`, …) as a
fine-grid field.

### Data format and half-precision storage

Each field carries a [data format](data-format.md): the order of its data in
memory and how its floating-point numbers are stored. Floating-point fields can
be stored in 16-bit half precision and still be used in ordinary expressions,
together with fields stored at full precision.

### Clear rules for metadata

A field's layout, shape and data format are set when it is constructed.
[Construction and copying](construction.md) says, for each constructor and for
assignment, where they come from.

### Block sums

`blockSum(expr, blocking)` sums an expression over each block of sites of a
fine lattice and returns a field on the coarse lattice, one site per block.
This is the restriction step of a multigrid solver. See [blockSum](block-sum.md).

### Mixed-layout expressions

Fields on different layouts can appear in the same expression, for example
`fine = fine * coarse`. Each coarse site's value is used for all fine sites in
its block. This is the prolongation step of a multigrid solver. See
[Mixed-layout expressions](mixed-layouts.md).

### Timing

Kernel launches are asynchronous, so a host-side timer measures only the
launch. [Timing](timing.md) shows how to get correct times with `StopWatch`.
All command-line options are listed in [Command-line options](options.md).
