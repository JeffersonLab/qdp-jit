# Scopes

New fields take their layout, shape and data format from the current
settings (see [Concepts](concepts.md)). Scopes change these settings for a
block of code and restore them at the end of the block. They are the usual way
to work on a coarse grid.

| Scope | Sets | Restores at the end |
|---|---|---|
| `LayoutScope(layout)` | the current layout | the previous layout |
| `ShapeScope(shape)` | the current shape | the previous shape |
| `LayoutShapeScope(layout, ns, nc)` | layout and shape | both |
| `LayoutShapeScope(field)` | `field`'s layout and shape | both |
| `JitDataFormatScope(format)` | the default data format | the previous format |

All scopes are RAII objects: the constructor sets, the destructor restores,
also when the block is left by an exception. They cannot be copied or moved.

A scope changes only what *new* fields get. Fields that already exist keep
their layout, shape and data format, and inside a scope they can be used as
before:

```c++
LatticeFermion psi;                      // fine, Ns=4, Nc=3
{
  LayoutShapeScope coarse_grid(coarse, 2, 24);
  LatticeFermion chi = psi;              // a copy: fine, Ns=4, Nc=3, like psi
  LatticeFermion c;                      // coarse, Ns=2, Nc=24
}
```

## LayoutScope

```c++
explicit LayoutScope(LatticeLayout layout);
void close();
```

Pushes `layout` onto the [layout stack](layouts.md#the-layout-stack) and pops
it at the end of the scope. `close()` pops it early; calling `close()` again,
or the destructor after `close()`, does nothing.

`LayoutScope` changes only the layout. The shape stays what it was, so a field
whose shape differs between the grids (a coarse fermion, say) needs a
`LayoutShapeScope` instead. `LayoutScope` alone is useful for code that
depends on the layout but not on the shape, such as `Layout::latticeCoordinate`,
the `Layout::` query functions, or seeding a layout's random number generator:

```c++
LatticeFermion mask(const LatticeFermion& in, int mu)
{
  LayoutScope scope(in.layout());
  LatticeReal x = Layout::latticeCoordinate(mu);   // on in's layout
  return in * x;
}
```

## ShapeScope

```c++
explicit ShapeScope(const Shape& shape);
void close();
```

Pushes `shape` onto the shape stack and pops it at the end. The layout is not
changed. `close()` works as for `LayoutScope`.

```c++
{
  ShapeScope s(Shape(2, 7));
  LatticeFermion psi;         // Ns=2, Nc=7, on the current layout
  LatticeColorMatrix u;       // Nc=7 (a color matrix has no spin index)
  LatticeReal r;              // no spin or color index
}
```

Each field takes from the shape only the extents its type has. See
[Shape](shape.md).

## LayoutShapeScope

```c++
LayoutShapeScope(LatticeLayout layout, int ns, int nc);
template<class T> explicit LayoutShapeScope(const OLattice<T>& field);
void close();
```

Sets the layout and the shape together, the common case on a coarse grid. The
first form takes them as arguments; the second takes `field.layout()` and
`field.shape()`, so new fields match an existing one:

```c++
LatticeFermion make_like(const LatticeFermion& x)
{
  LayoutShapeScope scope(x);
  LatticeFermion y = zero;    // same layout and shape as x
  return y;
}
```

`close()` restores both.

!!! warning "Take the scope from a field with spin and color"
    `LayoutShapeScope(field)` makes `field.shape()` current as it is. For a
    field without a spin index, such as a `LatticeColorMatrix`, that shape has
    Ns = 1, and a `LatticeFermion` created inside the scope then gets Ns = 1
    as well. Use a fermion or propagator as the model, or pass the extents
    explicitly with `LayoutShapeScope(field.layout(), ns, nc)`.

## JitDataFormatScope

```c++
explicit JitDataFormatScope(JitDataFormat format);
```

Sets the default data format for new fields and restores the previous one at
the end. It has no `close()`; end it by ending the block.

```c++
JitDataFormat half(JitDataOrder::SiteFast, JitStorage::Half);
{
  JitDataFormatScope scope(half);
  LatticeColorMatrix u_half;     // stored in half precision
  LatticeInteger     n;          // integer fields are always stored natively
}
```

It does not change the layout or the shape, and it applies on every layout.
Integer and boolean fields always use native storage. See
[Data format](data-format.md).

## The typical multigrid pattern

A coarse-grid operation usually needs coarse fields and fine fields at the same
time. Open a scope to declare the coarse fields, close it, and do the work with
the fine grid current:

```c++
LatticeColorVector restrict_to_coarse(const LatticeFermion& psi,
                                      const multi1d<LatticeFermion>& vecs,
                                      const multi1d<int>& blocking)
{
  const int nvec = vecs.size();

  LayoutShapeScope coarse_grid(Layout::current().coarsen(blocking), 1, nvec);
  LatticeColorVector psi_c;        // coarse, Nc=nvec
  coarse_grid.close();             // the fine grid is current again

  for (int i = 0; i < nvec; ++i)
    {
      LatticeComplex a = blockSum(localInnerProduct(vecs[i], psi), blocking);
      pokeColor(psi_c, a, i);      // a and psi_c are on the coarse layout
    }
  return psi_c;
}
```

The coarse fields stay on the coarse layout after `close()`, so they can be
used next to fine fields: [`blockSum`](block-sum.md) takes a fine expression
and returns a coarse field, and a
[mixed-layout expression](mixed-layouts.md) such as `fine = fine * coarse`
combines the two.

The opposite case, a fine temporary inside code that runs on the coarse grid,
does not need a scope: construct it with
[`OLattice(layout, shape)`](layouts.md#creating-a-field-on-a-given-layout).

## Nesting

Scopes can be nested to any depth, and the same layout can be opened again
inside itself. At the end of each block the settings of the enclosing block
are back.

!!! warning "Close scopes in reverse order"
    The layout and shape stacks are plain stacks: `close()` pops the top
    entry, whichever scope pushed it. If an outer scope is closed while an
    inner one is still open, the outer `close()` pops the *inner* layout and
    the outer one stays current:

    ```c++
    LayoutScope a(coarse);
    LayoutScope b(coarser);
    a.close();                 // pops coarser: coarse is still current
    ```

    Scopes that end with their blocks are always closed in the right order.
    Call `close()` only on the innermost open scope, and don't mix scopes with
    hand-written `Layout::push`/`pop` or `Shape::push`/`pop` that cross a scope
    boundary.
