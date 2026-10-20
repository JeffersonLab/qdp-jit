# Shape (Ns, Nc)

In QDP++ the number of spin components `Ns` and color components `Nc` are
compile-time constants and template arguments of the primitive types
(`PSpinVector<T,Ns>`, `PColorMatrix<T,Nc>`). In qdp-jit the primitive types
have no extent argument (`PSpinVector<T>`, `PColorMatrix<T>`), and the extents
are a run-time property of each field, its `Shape`. Fields of the same C++
type can therefore have different extents: a `LatticeFermion` with Ns = 4,
Nc = 3 on the fine grid and one with Ns = 2, Nc = 24 on a coarse grid.

The constants `Ns` and `Nc` still exist. They are the values qdp-jit was
configured with, and they make up the initial current shape: at start-up
`Shape::current()` is `Shape(Ns, Nc)`. Code that never changes the shape
behaves as in QDP++.

## The Shape class

```c++
Shape s(2, 24);          // Ns = 2, Nc = 24
s.ns();                  // 2
s.nc();                  // 24
std::cout << s;          // Shape(ns=2, nc=24)
```

A field's shape is `f.shape()`, for lattice fields (`OLattice`) and for
scalars (`OScalar`) alike.

## Which types use which extent

A field uses only the extents its type has. The other extent is stored as 1.
Inside `ShapeScope s(Shape(2, 5))`, default-constructed fields get:

| Type | Shape |
|---|---|
| `LatticeReal`, `LatticeComplex`, `Complex` | (1, 1) |
| `LatticeColorVector`, `LatticeColorMatrix`, `ColorMatrix` | (1, 5) |
| `LatticeSpinVector`, `LatticeSpinMatrix` | (2, 1) |
| `LatticeFermion`, `LatticePropagator`, `Fermion` | (2, 5) |

Some named types fix one extent and take the other from the current shape:

| Type | Ns | Nc | Inside `ShapeScope(Shape(2, 5))` |
|---|---|---|---|
| `LatticeDiracFermion`, `LatticeDiracPropagator` | 4 | current | (4, 5) |
| `LatticeHalfFermion` | current Ns / 2 | current | (1, 5) |
| `LatticeStaggeredFermion` | 1 | current | (1, 5) |
| `LatticeFermionF3`, `LatticeFermionD3` | current | 3 | (2, 3) |
| `LatticeColorMatrixF3`, `LatticeColorMatrixD3` | (none) | 3 | (1, 3) |
| `LatticeDiracFermionF3`, `LatticeDiracFermionD3` | 4 | 3 | (4, 3) |

The table is for default construction. When one of these types is made from a
plain field, for example `LatticeDiracFermion d(f)`, it keeps `f`'s extents on
the axes it does not fix, and `f` must have the fixed extent (Ns = 4 here);
otherwise the program aborts with `OLattice: Incompatible tensor dimensions`.

## The current shape

The shape of a default-constructed field comes from the top of the shape
stack, `Shape::current()`. Change it with a [`ShapeScope`](scopes.md#shapescope)
or, together with the layout, with a
[`LayoutShapeScope`](scopes.md#layoutshapescope):

```c++
{
  ShapeScope s(Shape(2, 24));
  LatticeFermion psi;          // (2, 24)
  LatticeColorMatrix u;        // (1, 24)
}
LatticeFermion chi;            // (Ns, Nc) again
```

`Shape::push` and `Shape::pop` change the stack directly; prefer the scopes,
which pop automatically. `Shape::pop` or `Shape::current` on an empty stack
aborts with `Shape::pop: Shape stack is empty` or
`Shape::current: Shape stack is empty`.

A field keeps its shape for its whole life. Changing the current shape does not
change existing fields.

## Creating a field with a given shape

Instead of changing the current shape, pass the shape to the constructor:

```c++
LatticeFermion psi(Shape(2, 24));          // on the current layout
LatticeFermion chi(Shape(2, 24), zero);    // and set to zero
LatticeFermion eta(coarse, Shape(2, 24));  // on a given layout, see Layouts
Fermion        f(Shape(2, 24));            // scalars too
```

Extents the type lacks are dropped: `LatticeColorMatrix u(Shape(4, 3))` has
shape (1, 3). A field made by a copy or from an expression takes the shape of
the source; an index the source's type does not have takes its extent from the
current shape, so `LatticeColorMatrix m = r + r` with `r` a `LatticeReal` has
Nc from `Shape::current()`. See also [Construction and copying](construction.md).

## How expressions combine shapes

The shape of an expression is worked out from its leaves, operation by
operation:

1. The shapes of the operands are merged extent by extent. An operand whose
   type has no spin index (or no color index) takes no part in that extent,
   so a `LatticeComplex` times a `LatticeFermion` (2, 5) gives (2, 5). Operands
   that have the index must agree on its extent, even when it is 1.
2. A few operations change an extent (see below).
3. Extents the result type lacks are dropped.

| Expression (with `psi`, `prop` of shape (2, 5)) | Result shape |
|---|---|
| `u * psi`, `u` a `LatticeColorMatrix` (1, 5) | (2, 5) |
| `c * psi`, `c` a `LatticeComplex` | (2, 5) |
| `shift(psi, FORWARD, mu) + psi` | (2, 5) |
| `traceSpin(prop)`, `traceSpin(outerProduct(psi, psi))` | (1, 5) |
| `traceColor(prop)` | (2, 1) |
| `trace(prop)`, `localInnerProduct(psi, psi)` | (1, 1) |
| `peekSpin(psi, s)` | (1, 5) |
| `peekColor(peekSpin(psi, s), c)` | (1, 1) |
| `sum(psi)` | a `FermionD` of shape (2, 5) |
| `innerProduct(psi, psi)`, `norm2(psi)` | a scalar, (1, 1) |

Operations that change an extent:

| Operation | Requires | Result |
|---|---|---|
| `spinProjectDir*` | Ns = 4 | Ns = 2 |
| `spinReconstructDir*` | Ns = 2 | Ns = 4 |
| `Gamma(n) * x` | Ns = 4 | Ns = 4 |

Operands that have an index but disagree on its extent cannot be combined.
The program aborts when the expression is evaluated:

| Cause | Message |
|---|---|
| `u * psi` with `u` (1, 3), `psi` (2, 5) | `Cannot combine incompatible extents 3 and 5` |
| `cv1 + cv5`, color vectors with Nc = 1 and Nc = 5 | `Cannot combine incompatible extents 1 and 5` |
| `Gamma(n) * psi` with Ns = 2 | `Cannot combine incompatible extents 4 and 2` |
| `spinProjectDir0Minus(psi)` with Ns = 2 | `Spin projection requires Ns=4; got Ns=2` |
| `spinReconstructDir0Minus(h)` with Ns ≠ 2 | `Spin reconstruction requires Ns=2; got Ns=…` |

## Assignment

Assigning to a lattice field keeps the field's shape. The right-hand side must
have the same extents, except for an index its type does not have: a number or
a `LatticeReal` can be assigned to a color matrix of any Nc (`u = 1.0` sets
every color matrix to the unit matrix). Otherwise the program aborts, also for
`+=` and the other compound assignments:

```c++
LatticeFermion psi;                 // (4, 3)
LatticeFermion chi(Shape(2, 5));
psi = chi;
// OLattice: Incompatible tensor dimensions: lhs=(Nc=3,Ns=4), rhs=(Nc=5,Ns=2)
```

Scalars (`OScalar`) behave differently: when the right-hand side's shape
does not fit, the scalar takes the new shape.

```c++
Fermion x;                          // (4, 3)
Fermion c(Shape(2, 12));
x = c;                              // x is now (2, 12)
```

This lets a scalar hold the result of `sum` or `peekSite` on any grid without
knowing the shape in advance.

## Reference

### `Shape` members

| Function | Description |
|---|---|
| `Shape()` | Ns = 1, Nc = 1 |
| `explicit Shape(int ns, int nc)` | the given extents |
| `int ns() const` | the spin extent |
| `int nc() const` | the color extent |
| `int extent(Shape::Axis axis) const` | `ns()` for `Shape::Axis::Spin`, `nc()` for `Shape::Axis::Color` |
| `void set_ns(int n)` | set the spin extent |
| `void set_nc(int n)` | set the color extent |

### The shape stack

| Function | Description |
|---|---|
| `static void Shape::push(const Shape& shape)` | make `shape` current |
| `static void Shape::pop()` | restore the previous shape; aborts on an empty stack |
| `static Shape Shape::current()` | the current shape; aborts on an empty stack |
| `ShapeScope(const Shape& shape)` | push for the lifetime of the scope, see [Scopes](scopes.md#shapescope) |

### Operators

| Function | Description |
|---|---|
| `bool operator==(const Shape&, const Shape&)` | both extents equal |
| `bool operator!=(const Shape&, const Shape&)` | |
| `std::ostream& operator<<(std::ostream&, const Shape&)` | prints `Shape(ns=…, nc=…)` |

### Fields

| Function | Description |
|---|---|
| `f.shape()` | the field's shape |
| `OLattice(const Shape& shape)` | a lattice field of `shape` on the current layout |
| `OLattice(const Shape& shape, const Zero&)` | the same, set to zero |
| `OLattice(LatticeLayout layout, const Shape& shape)` | on `layout`, see [Layouts](layouts.md#creating-a-field-on-a-given-layout) |
| `OScalar(const Shape& shape)` | a scalar of `shape` |
| `OScalar(const Shape& shape, const Zero&)` | the same, set to zero |
