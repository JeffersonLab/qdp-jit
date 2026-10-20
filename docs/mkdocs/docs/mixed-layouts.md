# Mixed-layout expressions

An expression can combine fields on different layouts, as long as each
field's lattice divides the lattice the expression is evaluated on. The
typical case is a coarse field in a fine expression:

```c++
LatticeFermion psi;                 // fine
LatticeComplex c_coarse;            // made on the coarse layout
// ...
psi += v * c_coarse;                // v fine: the prolongation step
```

Each coarse value is used for every fine site of its block. No temporary
fine copy of the coarse field is made: the kernel reads the coarse field
directly.

## Which coarse site a fine site reads

The expression is evaluated on one layout, the *expression layout* (see
[below](#the-expression-layout)). For a field on another layout, the blocking
is the ratio of the lattice sizes, `b[mu] = L_expr[mu] / L_field[mu]`, and the
fine site `x` reads the site `y` with

```
y[mu] = x[mu] / b[mu]          (integer division, in every dimension mu)
```

These are the same blocks that [`blockSum`](block-sum.md) sums over. With a
fine lattice `8 8 8 8` and a coarse field on `fine.coarsen(b)` with
`b = 1 2 2 4`, the fine site `(7, 5, 6, 7)` reads the coarse site
`(7, 2, 3, 1)`.

The blocking is worked out from the lattice sizes alone, so it does not
matter how the coarse layout was made: with `coarsen`, with repeated
`coarsen` calls, or with `Layout::find_or_create` (there is only one layout per
lattice size, see [Layouts](layouts.md#what-a-layout-is)). Each block lies on
one rank, so reading a coarse field needs no communication.

## The expression layout

| Evaluation | Expression layout |
|---|---|
| `dest = expr`, `dest += expr`, … | the layout of `dest` |
| `T f = expr`, `T f(expr)` | the finest leaf of `expr`, the one with the largest volume |
| `sum(expr)`, `norm2(expr)`, `innerProduct(a, b)`, `blockSum(expr, b)`, … | the finest leaf |

Assignment never changes the destination's layout (see
[Construction and copying](construction.md#assignment)). A field constructed
from an expression gets the finest leaf's layout:

```c++
LatticeReal g = f * a2 + a4;        // f fine, a2 and a4 on two coarser layouts:
                                    // g is fine
LatticeReal h = a2 * a4;            // h is on a2's layout
f = a2 * a4;                        // still fine: both broadcast to f
```

Scalars (`Real`, `Complex`, …) can appear in any expression and are the same
at every site.

## Restrictions

Every field in the expression must be on a lattice that divides the
expression layout's lattice in every dimension. The fine-to-coarse direction
is therefore an error:

```c++
coarse = fine;
// mixed-layout expression: the destination lattice 4 4 4 4 is not a multiple
// of the operand lattice 8 8 8 8
```

To go from fine to coarse, use [`blockSum`](block-sum.md).

The same error happens between two coarsenings of which neither divides the
other. With a fine lattice `8 8 8 8`, `ax` on `fine.coarsen(2 1 1 1)` (lattice
`4 8 8 8`) and `ay` on `fine.coarsen(1 2 1 1)` (lattice `8 4 8 8`):

| Expression | Result |
|---|---|
| `f = ax + ay` with `f` fine | works: both divide `8 8 8 8` |
| `ax = ay` | aborts: `4 8 8 8` is not a multiple of `8 4 8 8` |
| `LatticeReal e = ax + ay` | aborts: both have the same volume, and one of them becomes the expression layout |

The shapes and the data formats of the leaves follow the usual rules (see
[Shape](shape.md#how-expressions-combine-shapes) and
[Data format](data-format.md#mixing-formats-in-an-expression)), independently
of their layouts.

## Subsets

A subset on the left-hand side is a subset of the destination's layout:

```c++
f[rb[0]]  = c_coarse;               // the even sites of f get c_coarse
f[rb[1]] += c_coarse * f;
```

A reduction over a subset, `sum(f * c_coarse, rb[1])`, also uses the subset of
the expression layout.

## Shifts

A shift moves by one site of the expression layout, so the field inside a
shift must be on the expression layout:

```c++
f2 = shift(f, FORWARD, mu) * c_coarse;   // works: f is fine
f2 = f * shift(c_coarse, FORWARD, mu);   // aborts
```

The second line stops with

```
shift in a mixed-layout expression: the shifted field is on the lattice 4 4 4 4,
the expression is evaluated on the lattice 8 8 8 8. Shift the field on its own
layout first, or assign it to a field on the expression's layout and shift that
```

There are two different things that line could mean, and each can be written
out:

```c++
// the neighbouring coarse site: shift on the coarse layout first
LatticeComplex cs(c_coarse.layout(), c_coarse.shape());
cs = shift(c_coarse, FORWARD, mu);
f2 = f * cs;

// the neighbouring fine site: broadcast first, then shift on the fine layout
LatticeComplex cb;                  // fine
cb = c_coarse;
f2 = f * shift(cb, FORWARD, mu);
```

The two give different results: the first moves by a whole block, the second
by one fine site, which crosses into the next block only at the block's edge.

## Reductions

`sum`, `sumMulti`, `norm2`, `innerProduct` and the other reductions accept
mixed-layout expressions. They run over the sites of the finest leaf's layout,
and give the same result as with the coarse fields broadcast to that layout
first:

```c++
sum(f * c_coarse)                   // a sum over all fine sites
innerProduct(f, c_coarse)           // the same as innerProduct(f, cb), cb = c_coarse
sum(f * c_coarse, rb[1])            // over the odd fine sites
```

`blockSum` of a mixed-layout expression blocks the finest leaf's layout, see
[blockSum](block-sum.md#the-result).
