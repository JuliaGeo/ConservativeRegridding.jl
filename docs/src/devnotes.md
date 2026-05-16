# Developer notes

This file contains some notes written during the development of this package, and is intended to help clarify why certain design decisions were made.

## Why the regrid! interface is complicated

Initially, we had one function called `regrid!`. That proved to be too much power in a single function, making dispatch complicated.

To solve this, we initially proposed to split the function into three: `initialize_regrid!`, `perform_regridding!`, and `finalize_regrid!`.

I was wrong, when I thought that it was sufficient for `initialize_regridding!` to take only the source field and regridder, and `finalize_regridding!` to take only the destination field and regridder.

The case that breaks this is when you have one vector be a non-`DenseArray` and then the other vector be a `DenseArray`.  Consider the following:

* `src::DenseVector, dst::NotDenseVector`: you should do `mul!(regridder.dst_temp, regridder.intersections, src)` and then load `regridder.dst_temp` into `dst`.
* `src::NotDenseVector, dst::DenseVector`: you should load `src` into `regridder.src_temp`, then call `mul!(dst, regridder, src_temp)`.  Then you are done.
* `src::DenseVector, dst::DenseVector`: there should be no loading of anything, just a straight `mul!`.

Enabling the fast path for `::DenseVector, ::DenseVector` then requires that you are aware of the natures of both `src` and `dst`.

And this is not even getting started on what happens when you have multidimensional arrays.

### The new API that solves this

The solution to this is to have additional extractor functions that extract the vectors you want to do the regridding on.  Those can then be passed to regrid.  The new api surface is then:

```julia
extract_source_vector(src, regridder) -> something
extract_dest_vector(dst, regridder) -> something
```

where `something` is anything that works with `mul!`.  Then you can do:

```julia
function regrid!(dst, regridder, src)
    src_arraylike = extract_source_vector(src, regridder)
    dst_arraylike = extract_dest_vector(dst, regridder)
    initialize_regrid!(regridder, src, src_arraylike)
    perform_regridding!(dst_arraylike, regridder, src_arraylike)
    finalize_regrid!(dst, regridder, dst_arraylike)
end
```

This way, `perform_regridding!` can be assured what it's working on, and `finalize_regrid!` then has to know what the destination is.
In this case I'm not sure what the use of `initialize_regrid!` is.  But it's probably fine to keep it.  Perhaps the split is that `extract_source_vector` only gets you the pointer in some sense, and `initialize_regrid!` actually loads data into it - if necessary.

In this case the dispatch limitations are, then, the following (where a type annotation `::T` refers to something that you can dispatch on):

```julia
extract_source_arraylike(src::T, regridder)::AbstractArray
extract_dest_arraylike(dst::T, regridder)::AbstractArray
initialize_regrid!(regridder, src::T, src_arraylike) # - you should be assured what your src_arraylike is, but **must** always define `T`
finalize_regrid!(dst::T, regridder, dst_arraylike) # - you should be assured what your dst_arraylike is, but **must** always define `T`
perform_regridding!(dst_arraylike::T1, regridder, src_arraylike::T2) # - diagonal dispatch only
```
