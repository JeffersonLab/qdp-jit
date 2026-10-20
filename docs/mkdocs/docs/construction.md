# Construction and copying

Every lattice field carries a [layout](layouts.md), a [shape](shape.md) and a
[data format](data-format.md). All three are set when the field is constructed
and never change afterwards. This page says where each constructor takes them
from, and what assignment does with them.

There are three sources:

- the **ambient** settings: the current layout `Layout::current()`, the
  current shape `Shape::current()` and the default data format, all set by
  [scopes](scopes.md);
- a **source** field the new field is copied from;
- an **expression** the new field is computed from.

## Summary

| Construction | Layout | Shape | Data format |
|---|---|---|---|
| `T f;` `T f = zero;` `T f = 1.5;` | current | current | default |
| `T f(shape)` `T f(shape, zero)` | current | `shape` | default |
| `T f(layout, shape)` | `layout` | `shape` | default |
| `T f(format)` | current | current | `format` |
| `T f(a)` `T f = a` (copy) | `a`'s | `a`'s | `a`'s |
| `T f(a)` with `a` of another type, e.g. `LatticeFermionD f(aF)` | `a`'s | `a`'s | `a`'s |
| `T f(shape, a)` | `a`'s | `shape` | `a`'s |
| `T f(expr)` `T f = expr` | the finest leaf's | the expression's | default |
| `T f(expr)` with only scalar leaves, `T f(s)` with `s` a scalar | current | the expression's | default |
| `T f(std::move(a))`, a field returned by value | `a`'s | `a`'s | `a`'s |

"Current" and "default" mean the values in effect where the constructor runs.
In every row, extents the type does not have are dropped (a
`LatticeColorMatrix` has Ns = 1), and an index the source or the expression
does not have takes its extent from the current shape (a `LatticeColorMatrix`
made from a `LatticeReal` expression gets the current Nc). See
[Shape](shape.md#creating-a-field-with-a-given-shape).

## Default construction

A field constructed without a source takes everything from the ambient
settings:

```c++
{
  LayoutShapeScope coarse_grid(coarse, 2, 24);
  JitDataFormatScope half(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
  LatticeFermion psi;          // coarse, (2, 24), half
  LatticeFermion chi = zero;   // the same, set to zero
}
```

Constructors with a `Shape`, a `LatticeLayout` and `Shape`, or a
`JitDataFormat` argument override the corresponding ambient setting and take
the others from the ambient settings. See
[Layouts](layouts.md#creating-a-field-on-a-given-layout),
[Shape](shape.md#creating-a-field-with-a-given-shape) and
[Data format](data-format.md#creating-a-field-with-a-given-format).

## Copies

A copy takes all three from its source, whatever the ambient settings are:

```c++
LatticeFermion psi_c = make_coarse_half_fermion();  // coarse, (2, 24), half

LatticeFermion  a(psi_c);     // coarse, (2, 24), half
LatticeFermion  b = psi_c;    // the same
LatticeFermionD d(psi_c);     // the same, at double precision
```

This also holds for a conversion to another precision (`LatticeFermionD` from
a `LatticeFermionF`), except that an integer or boolean field is always stored
natively.

`T f(shape, a)` takes the layout and the format from `a` and the shape from
the argument. `shape` must have the same extents as `a` (after dropping the
extents `T` does not have); otherwise the program aborts with
`OLattice: Incompatible tensor dimensions: …` (`OScalar: …` for a scalar).

A [named type that fixes an extent](shape.md#which-types-use-which-extent),
made from a plain field, keeps the source's other extent and requires the
fixed one to match:

```c++
LatticeFermion f(Shape(4, 5));
LatticeDiracFermion d(f);        // (4, 5), whatever the current Nc
LatticeFermion h(Shape(2, 5));
LatticeDiracFermion e(h);        // aborts: Ns = 4 is required
```

## Fields computed from an expression

A field constructed from an expression is computed on the finest layout among
the expression's leaves (the one with the largest volume). Its shape is the
shape of the expression, and its data format is the default, whatever the
formats of the leaves:

```c++
LatticeFermion e1 = psi_c + psi_c;               // coarse, (2, 24), default format
LatticeFermion e2(shift(psi_c, FORWARD, 0));     // the same
LatticeComplex e3 = c_coarse * localInnerProduct(psi, psi);   // fine: psi is finer
```

The ambient layout plays no part as long as the expression has a lattice leaf.
An expression of scalars only, such as `LatticeReal r = Real(1) + Real(2)`,
and a field made from a scalar, `LatticeFermion f(s)`, are on the current
layout.

Mixing layouts in one expression is described in
[Mixed layouts](mixed-layouts.md).

## Moves and return values

A move takes over the source's storage together with its layout, shape and
format. Returning a field by value is therefore cheap and keeps the field's
metadata, even when the function opened scopes:

```c++
LatticeFermion make_coarse_half_fermion()
{
  LayoutShapeScope coarse_grid(coarse, 2, 24);
  JitDataFormatScope half(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
  LatticeFermion c;
  gaussian(c);
  return c;                    // the caller gets a coarse, (2, 24), half field
}
```

## Assignment

Assignment never changes the destination's layout, shape or data format. It
converts the right-hand side to them:

| | Rule |
|---|---|
| Shape | The right-hand side must have the destination's extents, except for an index its type does not have. Otherwise the program aborts with `OLattice: Incompatible tensor dimensions: …`. See [Shape](shape.md#assignment). |
| Data format | The values are converted to the destination's format: assigning to a half-precision field rounds them. See [Data format](data-format.md#where-a-fields-format-comes-from). |
| Layout | The right-hand side may live on a coarser layout whose blocks fit the destination's: each coarse value is broadcast to its block of fine sites. A right-hand side on a finer or incompatible layout aborts with `mixed-layout expression: the destination lattice … is not a multiple of the operand lattice …`. See [Mixed layouts](mixed-layouts.md). |

```c++
LatticeFermion h(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
h = psi;                       // still half: psi's values, rounded

LatticeFermion f(fine, Shape(2, 24));
f = psi_c;                     // still fine: psi_c broadcast to the fine sites
```

To get a field with the right-hand side's metadata, construct a new one
instead of assigning to an existing one.

### Move assignment

`a = std::move(b)` takes over `b`'s storage if `a` and `b` have the same
layout, shape and data format and `b` owns its storage. `b` then holds `a`'s
old storage and remains a valid field that can be assigned to. In all other
cases move assignment is a copy assignment, following the rules above, and `b`
is unchanged.

## Arrays of fields

A `multi1d` of fields constructs its elements by default construction, so they
take the ambient settings at the point where the array is sized:

```c++
multi1d<LatticeColorMatrix> u_h;
{
  JitDataFormatScope half(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
  u_h.resize(Nd);              // Nd half-precision fields
}
```

Copying an array copies each element with its metadata:
`multi1d<LatticeFermion> b(a)`, and `c = a` with `c` empty, give elements with
the layouts, shapes and formats of `a`'s elements. Assigning to an array that
already has the same size, `d = a`, assigns element by element and keeps `d`'s
metadata.

## Scalars

Scalars (`OScalar`, for example `Complex` or `ColorMatrix`) have a shape but
no layout or data format.

| Construction | Shape |
|---|---|
| `T s;` `T s = zero;` | current |
| `T s(shape)` | `shape` |
| `T s(t)`, `T s(expr)` | `t`'s or the expression's; an index the source lacks gets the current extent |

Unlike a lattice field, a scalar takes the right-hand side's shape on
assignment when it does not fit, so `s = sum(psi)` works for a `psi` of any
shape. See [Shape](shape.md#assignment).
