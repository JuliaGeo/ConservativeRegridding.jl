# Regridding Interface

```@meta
CurrentModule = ConservativeRegridding
```

[`regrid!`](@ref ConservativeRegridding.regrid!)`(dst_field, regridder, src_field)`
is one sparse matrix-vector product between two fields whose layouts the package
may not know: an extension's field type, a view, an N-dimensional array with a
spatial axis. The call decomposes into five seams, each dispatched on the type
that knows the answer, so a field type plugs in by extending only the seams it
needs.

| step | seam | dispatches on | does |
|---|---|---|---|
| 1 | `extract_source_arraylike(src_field, regridder; kwargs...)` | the source type | returns what `mul!` reads from: the field itself when it is a `DenseVector` or a contiguous view, `regridder.src_temp` otherwise |
| 2 | `extract_dest_arraylike(dst_field, regridder; kwargs...)` | the destination type | returns what `mul!` writes to, likewise with `regridder.dst_temp` |
| 3 | `initialize_regridding!(regridder, src_field, src_arraylike; kwargs...)` | the source type | loads `src_field` into `src_arraylike` when they differ |
| 4 | `perform_regridding!(dst_arraylike, regridder, src_arraylike; kwargs...)` | both arraylike types | the sparse `mul!` |
| 5 | `finalize_regridding!(dst_field, regridder, dst_arraylike; kwargs...)` | the destination type | loads `dst_arraylike` back into `dst_field` and divides by the destination areas (`normalize = true`) |

Steps 1 to 3 and 5 see the field types; step 4 sees only the arraylikes, so
the fast path (a dense source into a dense destination: one `mul!`, no copies)
is selected by the arraylikes alone, and a field type never has to know what
the other side is. [Developer notes](../devnotes.md) explain why the seams are
cut here.

## N-dimensional fields

A `StridedArray` with more than one dimension is regridded along the axis
`dims` (default 1), the other axes iterated: the extractors wrap it in an
[`NDSliceLoop`](@ref), an [`AbstractDimensionalSlicer`](@ref), and
`perform_regridding!` runs each pair of 1-D slices through the pipeline above.
An extension with its own layout subtypes `AbstractDimensionalSlicer` and
returns it from its extractors.

```@docs
ConservativeRegridding.regrid
ConservativeRegridding.AbstractDimensionalSlicer
ConservativeRegridding.slice_views
ConservativeRegridding.NDSliceLoop
```
