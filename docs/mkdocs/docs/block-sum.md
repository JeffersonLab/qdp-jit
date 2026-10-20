# blockSum

`blockSum(expr, blocking)` divides the lattice into blocks of sites and sums
an expression over each block. The result is a field on the coarse layout,
with one site per block. In a multigrid solver this is the restriction step:
the projection of a fine vector onto the coarse grid.

```c++
multi1d<int> blocking(Nd);
blocking = 4;                                   // 4^4 sites per block

LatticeComplexD a = blockSum(localInnerProduct(v, psi), blocking);
```

## What it computes

The blocking gives the block size in each dimension. The coarse site `y`
collects the fine sites `x` with `x[mu] / blocking[mu] == y[mu]` in every
dimension `mu`, that is the sites from `y[mu] * blocking[mu]` to
`(y[mu] + 1) * blocking[mu] - 1`. Its value is the sum of the expression over
those sites:

```c++
multi1d<int> blocking(Nd);
blocking = 2;

LatticeReal delta = zero;
pokeSite(delta, Real(1.0), x);         // x = (3, 1, 0, 2)
LatticeRealD b = blockSum(delta, blocking);
// b is 1 at the coarse site (1, 0, 0, 1) and 0 elsewhere

LatticeRealD n = blockSum(LatticeReal(1.0), blocking);
// n is 16, the number of sites in a block, at every coarse site
```

Every fine site belongs to exactly one block, so the total is unchanged:
`sum(blockSum(f, blocking))` equals `sum(f)`.

The sum runs over all sites; there is no version that takes a subset.

## The result

| | The result of `blockSum(expr, blocking)` |
|---|---|
| Type | the type of `sum(expr)` as a lattice field: a single-precision type becomes double precision (`LatticeComplexF` gives `LatticeComplexD`), other types stay the same |
| Layout | the layout of `expr`, blocked by `blocking`: `expr_layout.coarsen(blocking)` |
| Shape | the shape of `expr` |
| Data format | the default data format |

The sums are computed at the precision of the result type, so a
single-precision expression is summed in double precision.

The layout of the result does not depend on the current layout. If the
expression has leaves on several layouts, its layout is the finest one (see
[Mixed layouts](mixed-layouts.md)). Since
[there is one layout per lattice size](layouts.md#what-a-layout-is), the result
is on the same layout as fields made in a scope of `fine.coarsen(blocking)`
and can be combined with them:

```c++
LatticeLayout coarse = fine.coarsen(blocking);

auto r = blockSum(psi, blocking);      // r.layout() == coarse
{
  LayoutScope s(coarse);
  LatticeFermionD c = zero;
  c += r;                              // the same layout
}
```

Like every field computed from an expression, the result gets the default
data format, not the format of the expression's leaves. Inside a
`JitDataFormatScope` with half storage it is stored in half precision.

## The blocking

`blocking` has one entry per dimension (`Nd` entries) and must divide the
lattice size of the expression's layout in every dimension. Otherwise the
program aborts:

```
LatticeLayout::coarsen: blocking 3 1 1 1 does not divide the lattice 4 4 4 4
```

The coarse layout must fit the node grid, like every layout (see
[Layouts](layouts.md#each-layout-must-fit-the-node-grid)). Because of this,
each block lies entirely on one rank, and `blockSum` needs no communication
between ranks.

## Expressions as the argument

The argument can be any lattice expression. It is evaluated site by site
inside the summing kernel, without a temporary field for the expression:

```c++
LatticeComplexD a = blockSum(localInnerProduct(v, psi), blocking);
LatticeRealD    n = blockSum(localNorm2(v), blocking);
LatticeColorMatrixD m = blockSum(u * adj(w), blocking);
```

An expression that contains a `shift` is first evaluated into a temporary
field on the expression's layout, and the temporary is summed. The result is
the same, but it costs the memory and the time of one extra fine field.

## Repeated blocking

The argument can be on a coarse layout too, so blockings can be applied one
after the other. Two blockings in a row give the same field, on the same
layout, as one blocking by their product:

```c++
auto r1 = blockSum(blockSum(z, b1), b2);
auto r2 = blockSum(z, b1 * b2);        // per dimension
// r1.layout() == r2.layout(), and r1 equals r2
```

## blockSum and broadcasting

Assigning a coarse field to a fine one copies each coarse value to every fine
site of its block (see [Mixed layouts](mixed-layouts.md)). `blockSum` is the
adjoint of that operation: for a coarse field `c` and a fine field `f`,

```c++
LatticeComplex pc = c;                 // pc is fine: c broadcast to its blocks
innerProduct(c, blockSum(f, blocking)) == innerProduct(pc, f);   // up to rounding
```

So the restriction `blockSum(localInnerProduct(v, psi), blocking)` and the
prolongation `psi += v * c` with a coarse `c` form an adjoint pair, as a
multigrid solver needs. [Scopes](scopes.md#the-typical-multigrid-pattern) shows a
complete restriction onto a coarse vector with several components.

## How it works

`blockSum` runs one kernel, generated for the expression, the shapes and the
data formats of its leaves, and cached like other kernels.

Each coarse site is handled by a group of threads. Each thread in the group
reads a share of the block's fine sites, evaluates the expression there and
adds up the values. The group then combines its partial sums in shared memory
and one thread writes the result to the coarse site. With a group size of 1,
one thread sums the whole block and no shared memory is used.

The group size is a power of 2, at most 32 and no larger than the block size
rounded up to a power of 2. qdp-jit
chooses the smallest group size that gives enough threads to fill the GPU:
large blocks on a small coarse lattice get large groups, small blocks on a
large coarse lattice get one thread each. The command-line option

```
-blocksum N
```

sets the group size to `N` for every `blockSum`; `N` must be a power of 2,
otherwise the program aborts with `blockSum: -blocksum N is not a power of 2`.
The result does not depend on the group size apart from rounding. Use the
option only to compare timings.

## Reference

| Function | Description |
|---|---|
| `blockSum(const QDPType<T,C>& f, const multi1d<int>& blocking)` | the block sums of a field |
| `blockSum(const QDPExpr<RHS,OLattice<T>>& expr, const multi1d<int>& blocking)` | the block sums of an expression |
| `LatticeLayout::coarsen(const multi1d<int>& blocking)` | the layout of the result, see [Layouts](layouts.md#creating-coarse-layouts) |
| `-blocksum N` | command-line option: `N` threads per coarse site, a power of 2; 0 (the default) chooses automatically |
| `int qdp_jit_config_get_blocksum()`, `void qdp_jit_config_set_blocksum(int n)` | the same setting in code |
