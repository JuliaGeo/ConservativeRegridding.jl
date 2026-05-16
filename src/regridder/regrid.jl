# # Regridding
# This file defines the regridding pipeline and its behaviour.

"""$(TYPEDSIGNATURES)
Regrid data on `src_field` onto `dst_field` conservatively (mean-preserving) using the `regridder` matrix.

## Mathematics

If ``A`` are the intersection areas between the respective grids of the fields ``d`` (dst) and ``s`` (src),
and ``aˢ`` and ``aᵈ`` are the areas of the source and destination grid cells, then ``d`` is computed via

```math
d = (A s) / aˢ
```

Note that by construction,

```julia
aᵈ = sum(A; dims=2)
aˢ = sum(A; dims=1)

# Extended help

## Pipeline

`regrid!` decomposes into five steps:

1. `extract_source_arraylike(src_field, regridder)` — return an `AbstractArray` that `mul!` can
   read from. Dispatch on the source type.
2. `extract_dest_arraylike(dst_field, regridder)` — return an `AbstractArray` that `mul!` can
   write to. Dispatch on the destination type.
3. `initialize_regridding!(regridder, src_field, src_arraylike)` — load data from `src_field` into
   `src_arraylike` if needed. Dispatch on the source type.
4. `perform_regridding!(dst_arraylike, regridder, src_arraylike)` — perform the sparse matvec.
   Diagonal dispatch only (on the arraylike types).
5. `finalize_regridding!(dst_field, regridder, dst_arraylike)` — load `dst_arraylike` back into
   `dst_field`, applying normalization. Dispatch on the destination type.
```
"""
function regrid!(dst_field::D, regridder::R, src_field::S; kwargs...) where {D, R, S}
    src_arraylike = extract_source_arraylike(src_field, regridder; kwargs...)
    dst_arraylike = extract_dest_arraylike(dst_field, regridder; kwargs...)
    initialize_regridding!(regridder, src_field, src_arraylike; kwargs...)
    perform_regridding!(dst_arraylike, regridder, src_arraylike; kwargs...)
    finalize_regridding!(dst_field, regridder, dst_arraylike; kwargs...)
    return dst_field
end

# ## Extractors
# Return an arraylike (something `mul!` accepts) for the source/destination.

extract_source_arraylike(src_field::AbstractVector, regridder; kwargs...) = regridder.src_temp
extract_source_arraylike(src_field::DenseVector, regridder; kwargs...) = src_field

extract_dest_arraylike(dst_field::AbstractVector, regridder; kwargs...) = regridder.dst_temp
extract_dest_arraylike(dst_field::DenseVector, regridder; kwargs...) = dst_field

# ## Initialize
# Load source data into the arraylike. For `DenseVector` the arraylike is the field itself,
# so this is a no-op.

function initialize_regridding!(regridder, src_field::AbstractVector, src_arraylike; kwargs...)
    src_arraylike .= src_field
    return regridder
end

initialize_regridding!(regridder, src_field::DenseVector, src_arraylike; kwargs...) = regridder

# ## Perform regridding
# Diagonal dispatch only. The arraylikes are guaranteed to be `mul!`-compatible.
# In general, you should avoid dispatching on this function unless absolutely necessary.
function perform_regridding!(dst_arraylike::AbstractVector, regridder, src_arraylike::AbstractVector; kwargs...)
    LinearAlgebra.mul!(dst_arraylike, regridder.intersections, src_arraylike)
    return dst_arraylike
end

# ## Finalize
# Load the arraylike back into the destination field, applying normalization.
# For `DenseVector` the arraylike *is* the destination, so we just normalize in place.

Base.@constprop :aggressive function finalize_regridding!(dst_field::AbstractVector, regridder, dst_arraylike; normalize = true, kwargs...)
    if normalize
        @. dst_field = dst_arraylike / regridder.dst_areas
    else
        dst_field .= dst_arraylike
    end
    return dst_field
end

Base.@constprop :aggressive function finalize_regridding!(dst_field::DenseVector, regridder, dst_arraylike; normalize = true, kwargs...)
    if normalize
        dst_field ./= regridder.dst_areas
    end
    return dst_field
end

# ## Worked example: contiguous SubArray fast path
#
# `view(A, :, j)` of a column-major array is contiguous in memory, but a `SubArray`
# is never `<: DenseVector` in Julia's type hierarchy — so without these overrides
# the contiguous slice would be routed through `regridder.src_temp`/`dst_temp` and
# pay an unnecessary copy each call. We dispatch on `Base.FastContiguousSubArray`,
# whose `L=true` type parameter is Julia's compile-time flag for fast linear
# (i.e. contiguous) indexing, and treat it like a `DenseVector`: the SubArray is
# its own arraylike, initialization is a no-op, and finalization just normalizes
# in place. Strided-but-non-contiguous SubArrays (e.g. `view(A, 1:2:end)`) keep
# falling through to the generic `AbstractVector` path.
#
# This pair of sections also serves as a template: to plug a new field type into
# the pipeline, define the corresponding four methods (extract / initialize /
# finalize), routing through `regridder.src_temp`/`dst_temp` if and only if the
# field can't be handed to `mul!` directly.

extract_source_arraylike(src_field::Base.FastContiguousSubArray{<:Any,1}, regridder; kwargs...) = src_field
extract_dest_arraylike(dst_field::Base.FastContiguousSubArray{<:Any,1}, regridder; kwargs...) = dst_field

initialize_regridding!(regridder, src_field::Base.FastContiguousSubArray{<:Any,1}, src_arraylike; kwargs...) = regridder

Base.@constprop :aggressive function finalize_regridding!(dst_field::Base.FastContiguousSubArray{<:Any,1}, regridder, dst_arraylike; normalize = true, kwargs...)
    if normalize
        dst_field ./= regridder.dst_areas
    end
    return dst_field
end

# ## N-dimensional arrays
# Iterate over every slice along `dims` (default 1) and run the scalar `regrid!`
# on each. Each `view` is a `SubArray`: contiguous ones (the common `dims=1`
# case for column-major arrays) take the `FastContiguousSubArray` fast path
# defined above; strided or non-contiguous ones fall through to the
# `AbstractVector` path via the regridder's temp buffers.

# User-facing kwarg API. `@constprop :aggressive` lets the compiler propagate the
# `dims::Int` literal through `Val(dims)` so the workhorse method below is
# specialized on `dims` at the type level.
Base.@constprop :aggressive regrid!(dst_field::AbstractArray, regridder::Regridder, src_field::AbstractArray;
        dims::Int = 1, kwargs...) =
    regrid!(dst_field, regridder, src_field, Val(dims); kwargs...)

# Workhorse: `dims` is a `Val` type parameter, so it's known at compile time. This
# is critical for keeping the loop allocation-free for `N ≥ 3` — when `dims` is a
# runtime `Int` the `ntuple(d -> d == dims ? ...)` closures box for higher dims.
function regrid!(dst_field::AbstractArray{T,N}, regridder::Regridder, src_field::AbstractArray{S,N},
                 ::Val{dims}; kwargs...) where {T,S,N,dims}
    if N == 1
        # Delegate to the generic single-vector `regrid!` without recursing through
        # this AbstractArray method (which would otherwise match Vector too).
        return Base.invoke(regrid!, Tuple{Any,Any,Any}, dst_field, regridder, src_field; kwargs...)
    end
    @assert 1 <= dims <= N "dims=$dims is out of range for a $N-dimensional array"
    other_axes = ntuple(i -> axes(src_field, i < dims ? i : i + 1), Val(N - 1))
    for I in CartesianIndices(other_axes)
        idx = ntuple(d -> d == dims ? Colon() : I[d < dims ? d : d - 1], Val(N))
        regrid!(view(dst_field, idx...), regridder, view(src_field, idx...); kwargs...)
    end
    return dst_field
end

"""$(TYPEDSIGNATURES)
Regrid a vector `src_field` using `regridder`. Allocates a fresh destination vector."""
function regrid(
    regridder::AbstractMatrix,
    src_field::AbstractVector;
    kwargs...,
)
    n_out, n_in = size(regridder)
    dst_field = zeros(eltype(src_field), n_out)
    return regrid!(dst_field, regridder, src_field; kwargs...)
end
