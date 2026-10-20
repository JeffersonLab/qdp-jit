# Data format

Every lattice field has a data format, `f.data_format()`, which says how the
field's data is laid out in device memory and how floating-point numbers are
stored there. A data format has two parts:

```c++
struct JitDataFormat
{
  JitDataOrder order   = JitDataOrder::SiteFast;   // or SiteSlow
  JitStorage   storage = JitStorage::Native;       // or Half
};
```

The data format is a property of the field's data, not of its type: a
`LatticeFermion` stored in half precision is still a `LatticeFermion` and is
used in expressions like any other. Scalars (`OScalar`) have no data format.

## Order

`JitDataOrder` gives the order of the site index relative to the spin, color
and complex indices on the device:

| Order | Device layout |
|---|---|
| `SiteFast` (default) | the site index runs fastest: for each component, the values of all sites are contiguous ("structure of arrays") |
| `SiteSlow` | the site index runs slowest: all components of a site are contiguous ("array of structures") |

`SiteFast` gives coalesced memory access on GPUs and is the default.

## Storage

`JitStorage` says how floating-point numbers are stored on the device:

| Storage | Device storage |
|---|---|
| `Native` (default) | at the precision of the field's type: `float` for `LatticeFermionF`, `double` for `LatticeFermionD` |
| `Half` | as 16-bit IEEE half-precision numbers |

A half-precision field takes half the device memory of a single-precision
field and a quarter of a double-precision one, and moving it costs
correspondingly less memory bandwidth. Kernels read the stored half values,
convert them to the precision of the field's type, and compute at that
precision. When a kernel writes to a half-precision field, it rounds the result
to the nearest half value (ties to even).

Half precision has an 11-bit significand, about 3 decimal digits, and a
limited range:

| Value stored | Read back |
|---|---|
| `1.0/3.0` | `0.333251953125` |
| `1000.3` | `1000.5` |
| `65504` (largest half value) | `65504` |
| `70000` | `inf` |
| `1e-6` | `1.0132789611816406e-06` (subnormal) |
| `1e-8` | `0` |

Values above 65504 become infinite. Scale data such as gauge links or
normalized vectors so that it stays within range; `isfinite(f)` tells whether a
field holds only finite values.

Integer and boolean fields (`LatticeInteger`, `LatticeBoolean`, …) are always
stored natively, see [below](#integer-and-boolean-fields).

## The default data format

New fields get the default data format unless the constructor takes the format
from somewhere else (see [Construction and copying](construction.md)). At
start-up the default is `SiteFast` and `Native`.

The command-line option

```
-default-data-order-site-slow
```

changes the default order to `SiteSlow`. There is no command-line option for
half storage; use a scope or a constructor argument.

Within a block, set the default with a
[`JitDataFormatScope`](scopes.md#jitdataformatscope):

```c++
JitDataFormat half(JitDataOrder::SiteFast, JitStorage::Half);
{
  JitDataFormatScope scope(half);
  multi1d<LatticeColorMatrix> u_half(Nd);   // all stored in half precision
}
```

The current default is `qdp_jit_config_get_default_data_format()`.

## Creating a field with a given format

Pass the format to the constructor:

```c++
LatticeFermion psi_h(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
```

This ignores the default. Note that `JitDataFormat(order)` alone means native
storage, and `JitDataFormat()` means `SiteFast`, `Native` whatever the default
is.

## Where a field's format comes from

| Construction | Data format |
|---|---|
| `T f;`, `T f = zero;`, `T f(shape);` | the default |
| `T f(format)` | `format` |
| `T f(g)`, `T f = g` (copy) | `g`'s format |
| `LatticeFermionD f(g)` with `g` a `LatticeFermionF` | `g`'s format |
| `T f(expr)`, `T f = expr` (expression) | the default, whatever the formats of the leaves |

Assignment never changes the destination's format: `f = g` converts `g`'s
values to `f`'s format. Assigning a half-precision field to a native one gives
the rounded values at native precision; assigning a native field to a
half-precision one rounds them.

For `multi1d` of fields, copying keeps each element's format:
`multi1d<LatticeFermion> b(a)` and assignment to an empty array `c = a` give
elements with the formats of `a`'s elements. Assigning to an array that already
has elements, `d = a` with `d` of size 3, assigns element by element and keeps
`d`'s formats.

The full rules, together with those for the layout and the shape, are in
[Construction and copying](construction.md).

## Mixing formats in an expression

The leaves of an expression can have different data formats. Each leaf is read
in its own format and the result is written in the destination's format:

```c++
LatticeColorMatrix u_h(JitDataFormat(JitDataOrder::SiteSlow, JitStorage::Half));
LatticeFermion     psi_h(JitDataFormat(JitDataOrder::SiteFast, JitStorage::Half));
u_h   = u;
psi_h = psi;

LatticeFermion chi = u_h * psi_h + shift(psi_h, FORWARD, 0);   // native result
```

The result is exactly what the same expression gives with native fields that
hold the rounded values of `u_h` and `psi_h`. The same holds for shifts in
both directions, for subsets (`chi[rb[0]] += …`), for coarse-grid fields and
for reductions: `norm2`, `innerProduct` and `globalMax` of a half-precision
field equal those of a native field holding the same rounded values.

Host access works as for native fields: `peekSite` returns, and `pokeSite`
stores, values at the precision of the field's type, rounded to half on the
way in.

## Integer and boolean fields

Half storage applies only to fields with floating-point words. Under a
`JitDataFormatScope` with half storage, integer and boolean fields get the
scope's order with native storage. Passing a half format to the constructor
of an integer or boolean field explicitly aborts:

```
OLattice: half storage requires a floating-point word type
```

## Reference

### `JitDataFormat`

| Member | Description |
|---|---|
| `JitDataOrder order` | `JitDataOrder::SiteFast` or `JitDataOrder::SiteSlow` |
| `JitStorage storage` | `JitStorage::Native` or `JitStorage::Half` |
| `JitDataFormat()` | `SiteFast`, `Native` |
| `JitDataFormat(JitDataOrder o, JitStorage s = JitStorage::Native)` | the given order and storage |
| `bool half() const` | `storage == JitStorage::Half` |
| `operator==`, `operator!=` | both parts equal |

### Default and scope

| Function | Description |
|---|---|
| `JitDataFormat qdp_jit_config_get_default_data_format()` | the default format for new fields |
| `void qdp_jit_config_set_default_data_format(JitDataFormat fmt)` | set it; prefer `JitDataFormatScope` |
| `JitDataOrder qdp_jit_config_get_default_data_order()` | the order part of the default |
| `void qdp_jit_config_set_default_data_order(JitDataOrder o)` | set only the order part |
| `JitDataFormatScope(JitDataFormat fmt)` | set the default for the lifetime of the scope, see [Scopes](scopes.md#jitdataformatscope) |
| `-default-data-order-site-slow` | command-line option: default order `SiteSlow` |

### Fields

| Function | Description |
|---|---|
| `f.data_format()` | the field's data format |
| `explicit OLattice(JitDataFormat fmt)` | a field stored in `fmt`, on the current layout with the current shape |
