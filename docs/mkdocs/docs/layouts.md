# Layouts

QDP++ has a single lattice. qdp-jit can hold several lattice layouts at once,
typically the fine lattice and the coarse lattices of a multigrid hierarchy.
This page covers what a layout is, how coarse layouts are made, the layout
stack, and which parts of QDP++ follow the current layout.

## What a layout is

A `LatticeLayout` is a lattice size together with its site tables: for every
site, which rank holds it and its linear index on that rank, and the reverse
map from rank and index back to coordinates (see [Concepts](concepts.md#layout)).

The first layout is set up exactly as in QDP++:

```c++
QDP_initialize(&argc, &argv);

multi1d<int> nrow(Nd);
nrow = 16;
nrow[3] = 32;
Layout::setLattSize(nrow);
Layout::create();          // creates the first (fine) layout

LatticeLayout fine = Layout::current();
```

A `LatticeLayout` is a small value-type handle; copy it freely. Two handles
compare equal when they refer to the same layout, and since there is only one
layout per lattice size (see [below](#creating-coarse-layouts)), that is the
same as having the same lattice size. Handles have `<` and `std::hash`, so they
can be keys of `std::map` and `std::unordered_map`. A default-constructed
`LatticeLayout()` refers to no layout; its `index()` is `-1`.

Layouts are never destroyed before `QDP_finalize`. A handle stays valid for the
whole program from its point of creation.

## Creating coarse layouts

There are two ways to get another layout. Both return the layout without
changing the [current layout](#the-layout-stack).

**By blocking** an existing layout. `layout.coarsen(blocking)` divides the
lattice size by `blocking` in each dimension:

```c++
multi1d<int> blocking(Nd);
blocking = 2;                                       // 2^4 blocks
LatticeLayout coarse = fine.coarsen(blocking);      // 8 8 8 16
```

The blocking must divide the lattice size in every dimension, otherwise the
program stops with

```
LatticeLayout::coarsen: blocking 3 1 1 1 does not divide the lattice 4 4 4 4
```

A blocking of 1 in every dimension returns the layout itself.

**By lattice size.** `Layout::find_or_create(nrow)` returns the layout with
lattice size `nrow`, creating it if there is none yet:

```c++
multi1d<int> nrow(Nd);
nrow = 4;
LatticeLayout l = Layout::find_or_create(nrow);
```

`coarsen` is a shorthand for `find_or_create(lattSize() / blocking)`. There is
never more than one layout for a given lattice size, so asking for the same
size again, by either function, returns the same layout.

### Repeated blocking

Because layouts are identified by their lattice size, blocking in steps gives
the same layout as blocking in one go:

```c++
multi1d<int> bt(Nd), bs(Nd), b(Nd);
bt = 1; bt[3] = 2;               // 1 1 1 2
bs = 2; bs[3] = 1;               // 2 2 2 1
b  = 2;                          // 2 2 2 2

assert(fine.coarsen(bt).coarsen(bs) == fine.coarsen(b));
```

A multigrid hierarchy can therefore be built level by level, each level by
coarsening the one above it, and fields on a level can be compared or combined
with fields made from a different chain of blockings, as long as the lattice
sizes agree.

## Each layout must fit the node grid

All layouts share the machine geometry: the logical grid of ranks, set with
`-geom` or chosen by QMP when the first layout is created (see
[Concepts](concepts.md#machine-geometry)). Every rank holds part of every
layout. A lattice size therefore fits only if, in every dimension:

- it is divisible by the number of ranks in that dimension, and
- the resulting subgrid extent is a multiple of the minimal extent of the site
  layout qdp-jit was built with (`QDP_LAYOUT`):

| `QDP_LAYOUT`    | minimal subgrid extent      |
|-----------------|-----------------------------|
| `lexico`        | 1 in every dimension        |
| `cb2` (default) | 2 in x, 1 otherwise         |
| `cb3d`          | 2 in x, 1 otherwise         |

This applies to coarse layouts too, and limits how far a lattice can be
coarsened on a given number of ranks. If a layout does not fit, creating it
stops the program, whether through `coarsen` or `find_or_create`:

```
LayoutData::create - problem size 4 4 4 1 does not fit the node grid 1 1 1 2
with a subgrid that is a multiple of 2 1 1 1
```

This is with the lattice `4 4 4 4` on 2 ranks (`-geom 1 1 1 2`), blocked by 4
in time: the coarse lattice has 1 site in time, which cannot be split over 2
ranks. With `-geom 2 1 1 1` and the `cb2` site layout, blocking by 2 in x fails
too: each rank would get 1 site in x, and `cb2` needs an even number.

Two earlier checks report the same problem with other messages: a lattice
whose volume is not divisible by the number of ranks stops with
`LayoutData::create - problem size not divisible by number of processors`,
and one whose size is not a multiple of the minimal subgrid extent (for
example a coarse lattice with 1 site in x on one rank with `cb2`) stops with
`LayoutData::create - Error: problem size not multiple of smallest size allowed
for this type of layout`.

`Layout::fits_node_grid(nrow)` returns whether a lattice size fits, without
creating a layout. Use it to choose a blocking that works on the current number
of ranks.

```c++
multi1d<int> nrow = fine.lattSize() / blocking;
if (!Layout::fits_node_grid(nrow))
  QDPIO::cerr << "blocking too coarse for this node grid\n";
```

## The layout stack

qdp-jit keeps a stack of layouts. The layout on top is the **current layout**,
`Layout::current()`. After `Layout::create()` the stack holds the first layout.

```c++
Layout::push(coarse);        // coarse is now current
LatticeFermion c;            // a field on the coarse layout
Layout::pop();               // fine is current again
```

`push` and `pop` must be balanced. `Layout::push(LatticeLayout())` stops the
program (`Layout::push called with invalid layout -1`), and so does a call of
`Layout::current()` once the stack has been popped empty. Rather than calling
`push` and `pop` by hand, use a [`LayoutScope`](scopes.md), which pops when it
goes out of scope, also when an exception is thrown.

Layouts can be pushed in any order and any number of times. Pushing a layout
that is already on the stack is allowed.

### A field keeps its layout

A field's layout is fixed when the field is constructed (see
[Construction and copying](construction.md) for where it comes from). Popping
the stack does not change the layout of fields created while the layout was
current:

```c++
{
  LayoutScope scope(coarse);
  LatticeReal t = zero;      // on the coarse layout
  ...
}                            // fine is current again; t is gone

LayoutScope scope(coarse);
LatticeReal d = zero;        // on the coarse layout
scope.close();               // pop now, d stays alive

assert(d.layout() == coarse);
d[rb[0]] = Real(1.0);        // still evaluated on the coarse layout
Double s = sum(d);           // coarse volume / 2
```

### Creating a field on a given layout

A field can also be created on a layout without touching the stack. The
constructor `OLattice(LatticeLayout, Shape)` makes a field on the layout and
with the shape you pass, whichever layout and shape are current:

```c++
LayoutShapeScope coarse_grid(coarse, 2, 24);   // coarse layout, Ns=2, Nc=24

LatticeFermion c;                               // coarse, Ns=2, Nc=24
LatticeFermion f(fine, Shape(Ns, Nc));          // fine, the configured Ns and Nc
LatticeColorMatrix u(fine, Shape(Ns, Nc));      // fine, shape (1, Nc)
```

There is no constructor that takes only a layout, on purpose: inside a scope
like the one above, the current shape belongs to the coarse grid, and a fine
field would silently get it. Extents that the type
does not have are set to 1: `u` above has the shape (1, Nc), since a color
matrix has no spin index. The data format is the current default, as for a
default-constructed field (see [Data format](data-format.md)).

This is the simplest way to make a field on another layout in the middle of
code that runs on the current one, for example a fine-grid temporary inside a
coarse-grid scope. The fields are ordinary fields: `f = psi;` with a fine
`psi` works as usual.

### Operations use the field's layout

Operations on a field, such as assignment, `shift`, subsets, `random` and
reductions, use the field's own layout, not the current one. The current layout
matters only where no field says which layout to use; the
[next section](#what-the-current-layout-affects) lists these places.

!!! warning "Assignment does not move a field to another layout"
    `c = zero;` inside a `LayoutScope(coarse)` does not put `c` on the coarse
    layout if `c` was constructed on the fine one. Assignment keeps the
    destination's layout. Construct the field while the layout you want is
    current, or pass the layout to the constructor:
    `LatticeReal c(coarse, Shape());`.

## What the current layout affects

| What                                            | Uses                                         |
|-------------------------------------------------|----------------------------------------------|
| default-constructed fields                      | the current layout (and shape)               |
| fields made with `OLattice(layout, shape)`      | the layout passed                            |
| the `Layout::` query functions                  | the current layout                           |
| `Layout::latticeCoordinate(mu)`                 | the current layout                           |
| `RNG::setrn`, `RNG::savern`                     | the current layout's seed                    |
| `random(x)` for a scalar `x` (`Real`, …)        | the current layout's seed                    |
| `random(f)` for a lattice field `f`             | `f`'s layout and its seed                    |
| `Set::make`, `Map::make`                        | the current layout (see [below](#subsets-and-maps)) |
| `subset.siteTable()`, `numSiteTable()`, …       | the current layout; pass a layout to choose another |
| expressions, `shift`, subsets in `f[s] = …`     | the layout of the fields involved            |

### Lattice coordinates

`Layout::latticeCoordinate(mu)` returns a `LatticeInteger` on the current
layout, holding each site's coordinate in direction `mu`. To get coordinates
that match a given field, make the field's layout current first:

```c++
LatticeFermion mask_boundary(const LatticeFermion& in, const multi1d<int>& block, int mu)
{
  LayoutScope scope(in.layout());
  LatticeReal one = 1.0, nil = zero;
  LatticeReal m = where((Layout::latticeCoordinate(mu) + 1) % block[mu] == 0, one, nil);
  return in * m;
}
```

### Random numbers

Each layout has its own random number generator state. When a layout is
created, its seed is set to the same default value as the first layout's;
seeding one layout does not seed the others. To seed a coarse layout, make it
current:

```c++
RNG::setrn(seed);                      // seeds the fine layout only
{
  LayoutScope scope(coarse);
  RNG::setrn(coarse_seed);             // seeds the coarse layout
}
```

`random(f)` and `gaussian(f)` for a lattice field draw from the generator of
`f`'s layout, whichever layout is current. Drawing on a coarse layout leaves
the fine layout's stream unchanged. `random(x)` for a scalar draws from the
current layout's generator.

### Subsets and maps

The predefined sets and subsets (`all`, `even`, `odd`, `rb`, `rb3`, `mcb`) and
the nearest-neighbour shifts (`shift(f, FORWARD, mu)`, …) are set up on every
layout when it is created. They work on fields of any layout, with no further
action.

A `Set` or `Map` you make yourself is different: `Set::make(fn)` and
`Map::make(fn)` compute their tables for the current layout only, and the
function `fn` sees that layout's coordinates (for example through
`Layout::lattSize()`). To use one on another layout, call `make` again with
that layout current. The tables for the other layouts are kept:

```c++
Set blocks;
blocks.make(MyBlockFunc());            // tables for the fine layout
{
  LayoutScope scope(coarse);
  blocks.make(MyBlockFunc());          // and for the coarse layout
}
// blocks[i] now works on fine and on coarse fields
```

Using a user-made `Set` or `Map` on a layout it was not made for stops the
program:

```
Subset numSiteTable: no site tables for layout 1
Map make_lazy: map not made for layout 1
```

The site tables of a subset are queried for a given layout with
`s.siteTable(layout)`, `s.numSiteTable(layout)`, `s.isElement(layout, i)` and
so on. The versions without a layout argument use the current layout.

## Reference

### The `Layout` namespace

Functions marked *current* act on the current layout.

**Setting up the first layout**

| Function | Description |
|---|---|
| `void setLattSize(const multi1d<int>& nrow)` | set the lattice size, before `create()` |
| `void create()` | create the layout: site tables, default subsets and shifts, RNG |
| `void init()` | internal, called by `QDP_initialize` |
| `void finalize()` | internal, called by `QDP_finalize` |

**The layout stack and creating layouts**

| Function | Description |
|---|---|
| `LatticeLayout current()` | the layout on top of the stack |
| `void push(LatticeLayout l)` | make `l` current |
| `void pop()` | return to the previous layout |
| `LatticeLayout find_or_create(const multi1d<int>& nrow)` | the layout with lattice size `nrow`, created if needed; the stack is unchanged |
| `bool fits_node_grid(const multi1d<int>& nrow)` | whether a lattice of size `nrow` fits the node grid |
| `LatticeLayout from_index(int index)` | the layout with the given `index()` |
| `LayoutData& data(int index)` | internal: the layout's data |

**Lattice and node geometry** (*current*)

| Function | Description |
|---|---|
| `const multi1d<int>& lattSize()` | lattice size |
| `size_t vol()` | lattice volume |
| `const multi1d<int>& subgridLattSize()` | subgrid size on each rank |
| `int sitesOnNode()` | subgrid volume |
| `const multi1d<int>& logicalSize()` | machine geometry |
| `int numNodes()` | number of ranks |
| `int nodeNumber()` | this rank |
| `bool primaryNode()` | whether this is rank 0 |
| `const multi1d<int>& nodeCoord()` | this rank's coordinates in the node grid |
| `LatticeInteger latticeCoordinate(int mu)` | each site's coordinate in direction `mu` |

**Site maps** (*current*)

| Function | Description |
|---|---|
| `int nodeNumber(const multi1d<int>& coord)` | rank that holds the site `coord` |
| `int linearSiteIndex(const multi1d<int>& coord)` | linear index of `coord` on its rank |
| `int linearSiteIndex(int lexicosite)` | linear index of a lexicographic site number |
| `multi1d<int> siteCoords(int node, int index)` | coordinates of the site at `index` on `node` |
| `multi1d<int> nodeCoord(const multi1d<int>& coord)` | node-grid coordinates of the rank that holds `coord` |
| `int getNodeNumberFrom(const multi1d<int>& node_coord)` | rank at the given node-grid coordinates |
| `multi1d<int> getLogicalCoordFrom(int node)` | node-grid coordinates of a rank |

**I/O node grid** (*current*)

| Function | Description |
|---|---|
| `bool isIOGridDefined()` | whether an I/O grid is set |
| `void setIONodeGrid(const multi1d<int>& io_grid)` | set the I/O grid |
| `int numIONodeGrid()` | number of I/O nodes |
| `const multi1d<int>& getIONodeGrid()` | the I/O grid |

### `LatticeLayout`

The members of a `LatticeLayout` give the same information for that layout,
whichever layout is current.

| Member | Description |
|---|---|
| `int index() const` | the layout's number; `0` for the first layout, `-1` for none |
| `const multi1d<int>& lattSize() const` | lattice size |
| `size_t vol() const` | lattice volume |
| `const multi1d<int>& subgridLattSize() const` | subgrid size |
| `int sitesOnNode() const` | subgrid volume |
| `const multi1d<int>& logicalSize() const` | machine geometry |
| `int numNodes() const` | number of ranks |
| `int nodeNumber() const` | this rank |
| `bool primaryNode() const` | whether this is rank 0 |
| `int nodeNumber(const multi1d<int>& coord) const` | rank that holds `coord` |
| `int linearSiteIndex(const multi1d<int>& coord) const` | linear index of `coord` on its rank |
| `multi1d<int> siteCoords(int node, int index) const` | coordinates of a site |
| `LatticeLayout coarsen(const multi1d<int>& blocking) const` | the layout blocked by `blocking` |
| `==`, `!=`, `<`, `std::hash` | comparison, for use as a key |

### Fields

| Constructor or member | Description |
|---|---|
| `OLattice()` | a field on the current layout, with the current shape |
| `OLattice(LatticeLayout layout, const Shape& shape)` | a field on `layout` with `shape`, whichever layout and shape are current |
| `LatticeLayout layout() const` | the field's layout |

The other constructors, and where they take a field's layout from, are on
[Construction and copying](construction.md).
