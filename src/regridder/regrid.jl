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

# ## N-dimensional arrays
# TODO: re-enable once the extractor interface is settled.
#
# function regrid!(dst_field::AbstractArray, regridder::Regridder, src_field::AbstractArray; dims::Int=1)
#     if ndims(src_field) == 1
#         return regrid!(vec(dst_field), regridder, vec(src_field))
#     end
#     N = ndims(src_field)
#     @assert 1 <= dims <= N "dims=$dims is out of range for a $(N)-dimensional array"
#     other_axes = ntuple(i -> axes(src_field, i < dims ? i : i + 1), N - 1)
#     for I in CartesianIndices(other_axes)
#         idx = ntuple(N) do d
#             if d == dims
#                 Colon()
#             elseif d < dims
#                 I[d]
#             else
#                 I[d - 1]
#             end
#         end
#         src_slice = view(src_field, idx...)
#         dst_slice = view(dst_field, idx...)
#         regrid!(dst_slice, regridder, src_slice)
#     end
#     return dst_field
# end

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
