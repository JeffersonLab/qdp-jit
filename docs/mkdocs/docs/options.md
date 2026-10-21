# Command-line options

`QDP_initialize(&argc, &argv)` reads the options below from the command line.
They are not removed from `argv`, so the program sees them too. Options that
qdp-jit does not know are ignored, and QMP reads its own options as well.

```
mpirun -np 4 ./my_program -geom 1 1 2 2 -poolsize 20g -timing-run
```

At start-up the program prints the resulting settings: the memory pool size,
the code generation options and the launch configuration.

An option without its value, for example `-blocksize` as the last argument,
stops the program with `QDP_initialize: -blocksize needs 1 value`.

Memory sizes are a number with an optional suffix `k`, `m`, `g` or `t`
(powers of 1024): `5000m`, `20g`, `1.5g`. Anything else stops the program with
`Invalid memory size: … (expected a number with an optional k, m, g or t suffix)`.

## Ranks and devices

| Option | Description |
|---|---|
| `-geom Px Py Pz Pt` | the logical grid of ranks, one number per dimension. Without it, QMP chooses the grid when the first layout is created. See [Concepts](concepts.md#machine-geometry) |
| `-iogeom Px Py Pz Pt` | the grid of I/O nodes, as in QDP++ |
| `-use-gpu N` | use device `N`. Without it, each rank uses the device `local rank % number of devices`, with the local rank taken from the environment variables of the MPI launcher (`PMI_RANK`, `MV2_COMM_WORLD_LOCAL_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK` or `SLURM_LOCALID`). If none is set, device 0 is used and a warning is printed |
| `-V` | verbose output from QMP |
| `-debug-layout` | print the lattice and configuration summary each time a layout is created; without it, the summary is printed only for the first layout. Also print the site tables built for [mixed-layout expressions](mixed-layouts.md) and [`blockSum`](block-sum.md), and their cleanup at the end of the run |

## Device memory

qdp-jit allocates one large memory pool on the device at start-up and places
all fields in it.

| Option | Description |
|---|---|
| `-poolsize SIZE` | the size of the memory pool. Default: 92% of the free device memory |
| `-pool-reserve SIZE` | keep `SIZE` of device memory outside the pool, for libraries that allocate their own memory, e.g. the comm buffers for GPU-aware MPI calls (see `-gpudirect`). A `-poolsize` that leaves less than that is reduced |
| `-pool-alignment N` | the alignment of allocations in the pool, in bytes; a power of 2. Default: 256 |
| `-poolmemset V` | fill the whole pool with the byte value `V` at start-up; for debugging |
| `-pool-stats` | count the allocations by size and print the counts at the end of the run |
| `-gpudirect` | pass device buffers directly to MPI for communication (requires a GPU-aware MPI). Unless `-pool-reserve` is given, 10% of the free device memory is kept outside the pool for the communication buffers |
| `-ringbuffersize N` | the number of scalar values kept on the device for use as kernel arguments. Default: 100 |

## Fields

| Option | Description |
|---|---|
| `-default-data-order-site-slow` | new fields store their data with the site index running slowest (`SiteSlow`). See [Data format](data-format.md#the-default-data-format) |

## Kernels

| Option | Description |
|---|---|
| `-blocksize N` | threads per block for kernel launches. Default: 128 |
| `-blocksum N` | threads per coarse site in [`blockSum`](block-sum.md#how-it-works); a power of 2. Default: 0, chosen automatically |
| `-enable-hostjit`, `-disable-hostjit` | compile expressions that involve only scalars (`Real`, `Complex`, …) for the CPU and run them on the host (on by default), or run them on the device. Only in builds with `QDP_ENABLE_HOST_COMPILER` |
| `-gpu-sync` | wait after every kernel launch and check for errors, naming the kernel that failed; for debugging (CUDA) |

## Code generation

| Option | Description |
|---|---|
| `-opt N` | the LLVM optimization level, 0 to 3. Default: 3 (CUDA), 1 (ROCm), 2 (L0). `-opt -1` runs no LLVM passes. On L0, `-opt 0` also turns off the optimizations of the Intel graphics compiler (IGC), see [Level Zero](#level-zero) |
| `-cuda-ftz N` | 1: flush denormal numbers to zero in kernels, 0: don't. Default: 0 (CUDA) |
| `-libdevice-path DIR` | the directory of the device math library, if it is not found in the default locations |
| `-libdevice-name NAME` | the file name of the device math library |
| `-codegen-verbose` | print the code of every kernel when it is compiled |
| `-cache-verbose` | print every copy of a field between host and device memory |

## Timing

| Option | Description |
|---|---|
| `-timing-run` | `StopWatch` waits for the device at `start()` and `stop()`, so the times are correct. See [Timing](timing.md#stopwatch) |
| `-stats` | time every kernel and print a table of all kernels at the end of the run. See [Timing](timing.md#kernel-statistics) |

## ROCm

These options exist only in builds with the ROCm backend.

| Option | Description |
|---|---|
| `-workgroup-bound`, `-no-workgroup-bound` | compile kernels for at most `-blocksize` threads per block (on by default), or for any block size |
| `-opt-prepass`, `-no-opt-prepass` | run qdp-jit's own optimization pre-pass instead of the generic pass pipeline (on by default) |
| `-temp-files-path DIR` | the directory for the temporary files of the code generation. Default: the working directory |
| `-libdevice-extra FILE` | link an extra device library into every kernel; can be given several times |
| `-keep-files` | keep the temporary files of the code generation (also for the L0 backend) |

## Level Zero

These options exist only in builds with the L0 backend.

The L0 backend translates each kernel to SPIR-V in memory and hands it to the
Intel graphics compiler (IGC) through `zeModuleCreate`; no files are written
unless `-keep-files` is given. IGC compiles the SPIR-V for the device and always
optimizes, it has no numeric optimization level. `-opt N` therefore selects the
LLVM passes that run before the SPIR-V translation; `-opt 0` additionally
passes `-ze-opt-disable` to IGC, which compiles about 10 times faster but gives
kernels several times slower.

The defaults, `-opt 2` with IGC's automatic large register file, gave the
fastest kernels on Aurora (PVC) for most expressions. `-opt 3` unrolls more and
raises register pressure. Fields with half-precision storage in `SiteFast`
order can run faster with `-opt -1`.

| Option | Description |
|---|---|
| `-l0-build-flags "FLAGS"` | build flags passed to IGC for every kernel; they replace the default. Default: `-ze-intel-enable-auto-large-GRF-mode` (use 256 registers per thread when it avoids register spills). `-ze-opt-large-register-file` always uses 256 registers per thread; `-l0-build-flags ""` passes no flags (128 registers per thread) |

The IGC build log, which also reports informational messages such as
`[RetryManager] Start recompilation of the kernel`, is printed only when a
kernel fails to compile, or with `-codegen-verbose`.

## Options of special builds

| Option | Build | Description |
|---|---|---|
| `-p LEVEL` | `QDP_ENABLE_PROFILING` | the profiling level |
| `-deep-log-create FILE`, `-deep-log-compare FILE`, `-deep-log-tolerance X`, `-deep-log-fuzzfactor X`, `-deep-log-events N`, `-deep-log-memset V` | `QDP_ENABLE_DEEP_LOG` | record the result of every kernel to `FILE`, or compare a run with a recorded one; for debugging qdp-jit itself |

