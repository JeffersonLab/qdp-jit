# Timing

qdp-jit runs each expression as a kernel on the GPU. Kernel launches are
asynchronous: `chi = u * psi` returns as soon as the kernel is queued, before
it has run. A host timer started and stopped around a few lines of code
therefore measures how long it took to queue the kernels, not how long they
ran, and the work that was queued shows up in whatever is timed next.

## StopWatch

`StopWatch` is the QDP++ timer:

```c++
StopWatch sw;
sw.reset();
sw.start();
for (int i = 0; i < n; ++i)
  chi = u * psi;
sw.stop();
QDPIO::cout << "time: " << sw.getTimeInSeconds() << " s" << std::endl;
```

When the program runs with the command-line option

```
-timing-run
```

`start()` and `stop()` first wait until the GPU has finished all work queued
so far. The time between them is then the time the GPU took for the work
queued in between. Without `-timing-run`, `start()` and `stop()` do not wait,
and the times are wrong.

On one GPU, 2000 launches of `chi = u * psi` on a `16^4` lattice:

| | Measured |
|---|---|
| without `-timing-run` | 11 ms, the time to queue 2000 kernels; the kernels run later |
| with `-timing-run` | 23 ms, the time the kernels took |

The program prints whether `-timing-run` is on at start-up, in the line
`Accurate timing : 1`.

Waiting for the GPU costs time, and it stops the host from queueing work
ahead. Use `-timing-run` for timing runs, not for production runs.

`StopWatch(false)` makes a stopwatch that never waits, even with
`-timing-run`. Use it to measure host time only, for example the time spent
queueing kernels.

Any operation that brings a result back to the host waits for the GPU too:
`norm2`, `sum`, `innerProduct`, `peekSite` and the other reductions return a
value on the host, so they include all work queued before them. A timed block
that ends with a reduction is therefore timed correctly even without
`-timing-run`, but it then includes the reduction.

## The first call compiles the kernel

The first time an expression is evaluated, qdp-jit generates and compiles a
kernel for it, which takes several milliseconds. Later evaluations reuse the
kernel:

| Evaluation | Time |
|---|---|
| first `chi = u * psi` | 11 ms (compilation) |
| second `chi = u * psi` | 0.03 ms |

A kernel is reused for the same expression on other fields of the same types,
on another subset and on another layout. A new kernel is compiled for a new
expression, and also for the same expression on fields with a different shape
or data format, or when the destination also appears on the right-hand side
(`chi = u * chi` after `chi = u * psi`).

To time the work and not the compilation, run the code once before timing it:

```c++
chi = u * psi;                 // warm-up: compiles the kernel

StopWatch sw;
sw.start();
for (int i = 0; i < n; ++i)
  chi = u * psi;
sw.stop();
```

At the end of the run, `QDP_finalize` prints how many kernels were compiled
and how long the compilation took in total:

```
Code generator
  functions jit-compiled (GPU)             27
  total time for IR builder:               0.008144 s
  total time for libm IR linking:          0.003707 s
  total time for IR passes:                0.002814 s
  total time for code generation:          0.137968 s
  total time for dynamic loading:          0.002633 s
```

A number of compiled kernels that grows with the length of the run means that
new kernels keep being made, for example for fields with ever new shapes.

## Kernel statistics

The command-line option

```
-stats
```

times every kernel launch on the GPU. At the end of the run, `QDP_finalize`
prints a table of all kernels, sorted by their total time:

```
#   calls  stack  regs  cmem  sum(ms)     mean      stddev    name
1   8005   0      64    0     102.313667  0.012781  0.000528  eval5
8   3      0      44    0     0.078720    0.026240  0.003231  reduce_expr0
...
```

| Column | Meaning |
|---|---|
| `calls` | number of launches |
| `stack` | local memory per thread (stack and register spills), in bytes |
| `regs` | registers per thread |
| `cmem` | constant memory, in bytes |
| `sum(ms)`, `mean`, `stddev` | total, mean and standard deviation of the kernel's run time, in milliseconds |
| `name` | the kernel's name |

On AMD GPUs, the columns `stack`, `regs` and `cmem` are replaced by the
register and memory counts of that platform.

The same table is written to the file `qdp_jit_stats.txt` in the working
directory, with two extra columns for register spills, which are filled in only
on some platforms.
After the table, the file lists each kernel's name with the C++ types of the
expression it was generated for, which tells which kernel belongs to which
line of code.

`-stats` waits for each kernel to finish before the next one is launched, so
the program runs slower: the 2000 launches above take 41 ms instead of
23 ms. Use it to find which kernels take the time, and `StopWatch` with
`-timing-run` to measure the time of a piece of code. The statistics are
collected on the primary rank only.

## Reference

| | Description |
|---|---|
| `StopWatch(bool device_sync = true)` | a stopwatch; with `device_sync` false it never waits for the GPU |
| `void reset()` | clear the stopwatch |
| `void start()` | start; with `-timing-run`, first wait for the GPU |
| `void stop()` | stop; with `-timing-run`, first wait for the GPU. Aborts if the stopwatch was not started |
| `double getTimeInSeconds()`, `double getTimeInMicroseconds()` | the time between `start()` and `stop()`; aborts if the stopwatch was not started and stopped |
| `-timing-run` | command-line option: `StopWatch` waits for the GPU |
| `-stats` | command-line option: time every kernel and print the table at the end |
