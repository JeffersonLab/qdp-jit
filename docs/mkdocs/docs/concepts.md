# Concepts

This page defines the terms used on the other pages. Each term is defined once
here; the pages that follow go into detail.

## Lattice size

The global extent of a lattice in each of the `Nd` dimensions, for example
`8 8 8 16`. As in QDP++, the lattice size of the first lattice is set with
`Layout::setLattSize(nrow)` before `Layout::create()`.

```c++
const multi1d<int>& nrow = Layout::lattSize();   // of the current layout
size_t volume            = Layout::vol();        // product of nrow
```

In qdp-jit there can be several lattices, each with its own lattice size. The
functions in the `Layout` namespace (`Layout::lattSize()`, `Layout::vol()`,
`Layout::sitesOnNode()`, …) return the values of the
[current layout](#current-layout). The same functions are members of a
`LatticeLayout` object, for example `layout.lattSize()`.

## Machine geometry

The logical grid of MPI ranks (nodes), one extent per dimension, returned by
`Layout::logicalSize()`. Set it on the command line with

```
-geom X Y Z T
```

Without `-geom`, QMP chooses the geometry. The geometry is fixed when the first
layout is created and is the same for every layout created later.

## Subgrid

The part of a lattice that lives on one rank: the lattice size divided by the
machine geometry in each dimension.

```c++
const multi1d<int>& sub = Layout::subgridLattSize();  // lattice size / geometry
int sites               = Layout::sitesOnNode();      // product of sub
```

Every lattice size must therefore be divisible by the machine geometry in every
dimension. This holds for coarse lattices too, which limits how far a lattice
can be coarsened on a given number of ranks. See [Layouts](layouts.md).

## Layout

A `LatticeLayout` is a lattice size together with its site tables: which rank
holds a site, the site's linear index on that rank, and the reverse map from
rank and index to coordinates. The first layout is created as in QDP++, with
`Layout::setLattSize(nrow)` and `Layout::create()` after `QDP_initialize`.
Further layouts, typically coarser ones, are created from it.
A `LatticeLayout` is a small handle; two handles are equal when they refer to
the same layout. See [Layouts](layouts.md).

### Current layout

qdp-jit keeps a stack of layouts. Its top, `Layout::current()`, is the layout
that default-constructed fields are created on and that the `Layout::`
functions refer to. At start-up the stack holds the first layout. See
[Layouts](layouts.md#the-layout-stack) and [Scopes](scopes.md).

## Shape

A field's `Shape` is its pair of extents (Ns, Nc): the number of spin and
color components. In QDP++ these are compile-time constants; in qdp-jit they
are run-time values, so fields of the same C++ type can have different shapes.
A field's shape is `f.shape()`. Like layouts, shapes have a stack, and its top,
`Shape::current()`, is used for new fields. At start-up it holds the `Ns` and
`Nc` that qdp-jit was configured with. See [Shape](shape.md).

## Data format

A field's `JitDataFormat` describes how its data is stored in memory: the order
of the site and component indices (`JitDataOrder`), and whether floating-point
numbers are stored at their native precision or as 16-bit halves
(`JitStorage`). A field's data format is `f.data_format()`. See
[Data format](data-format.md).

## Field

A lattice field (`OLattice`, for example `LatticeFermion`) or a scalar
(`OScalar`, for example `Real` or `Complex`). Besides its data, every lattice
field carries three pieces of metadata:

| Metadata    | Accessor           |
|-------------|--------------------|
| layout      | `f.layout()`       |
| shape       | `f.shape()`        |
| data format | `f.data_format()`  |

Where these come from when a field is constructed, copied or assigned is
described in [Construction and copying](construction.md).

## Expression and leaf

An expression is a QDP++ expression such as `u * shift(psi, FORWARD, mu) + chi`.
qdp-jit compiles each expression into a kernel the first time it is evaluated
and reuses the kernel when the same expression is evaluated again.
The leaves of an expression are its operands: the fields and scalars it reads
(`u`, `psi` and `chi` above). In qdp-jit the leaves of one expression can have
different layouts, shapes and data formats; the rules are on the pages for
[shapes](shape.md), [data formats](data-format.md) and
[mixed layouts](mixed-layouts.md).
