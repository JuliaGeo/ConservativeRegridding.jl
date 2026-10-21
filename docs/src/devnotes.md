# Developer notes

```@meta
CurrentModule = ConservativeRegridding
```

Why some of the package's designs look the way they do.

## Why `regrid!` has five seams

A `regrid!` is one sparse `mul!`, but `src` and `dst` may each be dense (`mul!`
reads or writes it directly) or not (a non-contiguous view, an extension's field
type), and the right sequence depends on both sides at once:

| `src` | `dst` | sequence |
|---|---|---|
| dense | dense | `mul!(dst, A, src)` |
| dense | other | `mul!(regridder.dst_temp, A, src)`, then load `dst_temp` into `dst` |
| other | dense | load `src` into `regridder.src_temp`, then `mul!(dst, A, src_temp)` |
| other | other | both loads around `mul!(dst_temp, A, src_temp)` |

A single `regrid!` method per field type cannot pick the row: it dispatches on
one side. The [five seams](interfaces/regrid.md) split the decision so that
every method sees exactly what it needs:

- `extract_source_arraylike` / `extract_dest_arraylike` return the arraylike
  `mul!` works on, each dispatched on its own field type;
- `perform_regridding!` dispatches on the two arraylikes, where the dense-dense
  fast path is visible;
- `initialize_regridding!` / `finalize_regridding!` do the loads, each
  dispatched on its own field type.

Multidimensional arrays ride the same seams: the extractor returns an
[`AbstractDimensionalSlicer`](@ref), and `perform_regridding!` dispatches on it
to run the 1-D pipeline per slice.
