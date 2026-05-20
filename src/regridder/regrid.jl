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
# `dims` is lifted to an explicit kwarg here (rather than ridden inside `kwargs...`)
# and `@constprop :aggressive` so the literal value reaches `NDSliceLoop{dims}` at
# the extract callsite. Without this, `dims` is boxed through the kwarg `Pairs`
# plumbing and the `Dim` type parameter of the slicer becomes dynamic — every
# downstream call (`slice_views`, `_slice_index`, the per-slice 1-D dispatch)
# then runs through runtime dispatch, costing megabytes per N-D call.
Base.@constprop :aggressive function regrid!(dst_field::D, regridder::R, src_field::S; dims::Int = 1, kwargs...) where {D, R, S}
    src_arraylike = extract_source_arraylike(src_field, regridder; dims, kwargs...)
    dst_arraylike = extract_dest_arraylike(dst_field, regridder; dims, kwargs...)
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

function initialize_regridding!(regridder, src_field::AbstractVector, src_arraylike::AbstractVector; kwargs...)
    src_arraylike .= src_field
    return regridder
end

initialize_regridding!(regridder, src_field::DenseVector, src_arraylike::AbstractVector; kwargs...) = regridder

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

Base.@constprop :aggressive function finalize_regridding!(dst_field::AbstractVector, regridder, dst_arraylike::AbstractVector; normalize = true, kwargs...)
    if normalize
        @. dst_field = dst_arraylike / regridder.dst_areas
    else
        dst_field .= dst_arraylike
    end
    return dst_field
end

Base.@constprop :aggressive function finalize_regridding!(dst_field::DenseVector, regridder, dst_arraylike::AbstractVector; normalize = true, kwargs...)
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

initialize_regridding!(regridder, src_field::Base.FastContiguousSubArray{<:Any,1}, src_arraylike::AbstractVector; kwargs...) = regridder

Base.@constprop :aggressive function finalize_regridding!(dst_field::Base.FastContiguousSubArray{<:Any,1}, regridder, dst_arraylike::AbstractVector; normalize = true, kwargs...)
    if normalize
        dst_field ./= regridder.dst_areas
    end
    return dst_field
end

# ## N-dimensional regridding via the AbstractDimensionalSlicer interface
#
# N-D handling is *not* a `regrid!` dispatch — it routes through the existing
# 5-step pipeline (extract / initialize / perform / finalize). When the input
# is multi-dimensional, `extract_*_arraylike` returns an `AbstractDimensionalSlicer`,
# and `perform_regridding!` dispatches on that to iterate slices through the
# 1-D pipeline.

"""
    AbstractDimensionalSlicer

A marker for the regridding pipeline that the field should be processed in
1-D pieces. `perform_regridding!` dispatches on this type and iterates
`slice_views(slicer)`, running the standard pipeline on each (source,
destination) slice pair.

# Interface

A concrete subtype `MySlicer <: AbstractDimensionalSlicer` must implement:

| method | returns | semantics |
|---|---|---|
| `Base.parent(::MySlicer)` | `AbstractArray` | the underlying field data |
| `slice_views(::MySlicer)` | iterator of 1-D arraylikes | each element is something `mul!` can read from / write into |

# Contract

- Source and destination slicers in a single `perform_regridding!` call must produce
  iterators of equal length, paired by position.
- Each yielded slice must be a 1-D `AbstractVector` subject to the existing 1-D dispatch
  (`DenseVector` / `FastContiguousSubArray` fast paths, `AbstractVector` slow path).
- The slicer owns the data; `initialize_regridding!` and `finalize_regridding!` for
  slicers are no-ops because each slice is a view into the underlying field.
- Built-in slicers must fail loudly when the source/destination slice counts or
  non-spatial axes differ. Do not rely on `zip` truncation semantics.

# Built-in implementations

- [`NDSliceLoop`](@ref) — slices an N-D `StridedArray` along a chosen dimension.

# Defining a custom slicer

```julia
struct MyMatrixSlicer{A<:AbstractMatrix} <: ConservativeRegridding.AbstractDimensionalSlicer
    array::A
end

Base.parent(s::MyMatrixSlicer) = s.array
ConservativeRegridding.slice_views(s::MyMatrixSlicer) = (vec(parent(s)),)

ConservativeRegridding.extract_source_arraylike(src::MyMatrixField, r; kwargs...) =
    MyMatrixSlicer(rawdata(src))
ConservativeRegridding.extract_dest_arraylike(dst::MyMatrixField, r; kwargs...) =
    MyMatrixSlicer(rawdata(dst))
```
"""
abstract type AbstractDimensionalSlicer end

"""
    slice_views(slicer::AbstractDimensionalSlicer)

Return an iterator yielding 1-D arraylike views into the slicer's data.
Each view must be acceptable as the source/destination of a 1-D `regrid!` pass.
Required by the [`AbstractDimensionalSlicer`](@ref) interface.
"""
function slice_views end

"""
    NDSliceLoop{Dim,N,A<:StridedArray{<:Any,N}}

Built-in [`AbstractDimensionalSlicer`](@ref) for N-D strided arrays. Iterates 1-D
views along all axes except `Dim`, in column-major order over the other axes.

Constructed automatically by `extract_source_arraylike` / `extract_dest_arraylike`
when the input is a `StridedArray{T,N}` with `N ≥ 2`; not intended for direct user use.
"""
struct NDSliceLoop{Dim,N,A<:StridedArray{<:Any,N}} <: AbstractDimensionalSlicer
    array::A
end

function NDSliceLoop{Dim}(arr::StridedArray{<:Any,N}) where {Dim,N}
    _check_valid_dim(Val(Dim), Val(N))
    return NDSliceLoop{Dim,N,typeof(arr)}(arr)
end

Base.parent(s::NDSliceLoop) = s.array

function slice_views(s::NDSliceLoop{Dim,N}) where {Dim,N}
    arr = parent(s)
    other_axes = _nonspatial_axes(s)
    return (view(arr, _slice_index(Val(Dim), Val(N), I)...) for I in CartesianIndices(other_axes))
end

@inline function _check_valid_dim(::Val{Dim}, ::Val{N}) where {Dim,N}
    (Dim isa Integer && 1 <= Dim <= N) ||
        throw(ArgumentError("dims=$Dim is out of range for a $N-dimensional array"))
    return nothing
end

@inline _nonspatial_axes(s::NDSliceLoop{Dim,N}) where {Dim,N} =
    ntuple(i -> axes(parent(s), i < Dim ? i : i + 1), Val(N - 1))

@inline _slice_index(::Val{Dim}, ::Val{N}, I::CartesianIndex) where {Dim,N} =
    ntuple(d -> d == Dim ? Colon() : I[d < Dim ? d : d - 1], Val(N))

# ### Pipeline integration

# N-D StridedArrays wrap into an NDSliceLoop. `StridedArray` (not `AbstractArray`)
# is the critical type: package field types (Oceananigans.Field, ClimaCore.Fields.Field,
# Healpix.HealpixMap) subtype AbstractArray but NOT StridedArray, so they continue
# through their own extension-defined extract methods.
extract_source_arraylike(src::StridedArray{T,N}, regridder; dims::Int=1, kwargs...) where {T,N} =
    NDSliceLoop{dims}(src)
extract_dest_arraylike(dst::StridedArray{T,N}, regridder; dims::Int=1, kwargs...) where {T,N} =
    NDSliceLoop{dims}(dst)

# 1-D StridedVector guard. Without this, the StridedArray{T,N} method above would
# hijack 1-D non-contiguous SubArrays (e.g. `view(A, 1:2:end)`), which match
# StridedVector but neither DenseVector nor FastContiguousSubArray. Route them
# through src_temp/dst_temp like the existing AbstractVector path.
extract_source_arraylike(src::StridedVector, regridder; kwargs...) = regridder.src_temp
extract_dest_arraylike(  dst::StridedVector, regridder; kwargs...) = regridder.dst_temp

# Slicer pipeline: initialize and finalize are no-ops because the data lives
# in `parent(slicer)`; slices are views into it.
initialize_regridding!(regridder, src_field, src_arraylike::AbstractDimensionalSlicer; kwargs...) = regridder
finalize_regridding!(dst_field, regridder, dst_arraylike::AbstractDimensionalSlicer; kwargs...) = dst_field

# Generic custom slicers are responsible for satisfying the interface contract.
# Built-in NDSliceLoop can check its non-spatial axes cheaply before entering the loop.
_validate_slice_compatibility!(dst::AbstractDimensionalSlicer, src::AbstractDimensionalSlicer) = nothing

function _validate_slice_compatibility!(
    dst::NDSliceLoop{DstDim,DstN},
    src::NDSliceLoop{SrcDim,SrcN},
) where {DstDim,DstN,SrcDim,SrcN}
    DstN == SrcN ||
        throw(DimensionMismatch("source and destination ranks must match; got source rank $SrcN and destination rank $DstN"))
    DstDim == SrcDim ||
        throw(DimensionMismatch("source and destination regrid dimensions must match; got source dims=$SrcDim and destination dims=$DstDim"))
    src_axes = _nonspatial_axes(src)
    dst_axes = _nonspatial_axes(dst)
    src_axes == dst_axes ||
        throw(DimensionMismatch("source and destination non-spatial axes must match; got source axes $src_axes and destination axes $dst_axes"))
    return nothing
end

# perform_regridding! on the slicer: iterate source and destination slice views in
# lockstep, running the standard pipeline per slice. The per-slice work re-enters
# the existing 1-D dispatch (DenseVector / FastContiguousSubArray / AbstractVector).
function perform_regridding!(
    dst_arraylike::AbstractDimensionalSlicer,
    regridder,
    src_arraylike::AbstractDimensionalSlicer;
    kwargs...,
)
    _validate_slice_compatibility!(dst_arraylike, src_arraylike)
    for (src_slice, dst_slice) in zip(slice_views(src_arraylike), slice_views(dst_arraylike))
        s_src = extract_source_arraylike(src_slice, regridder; kwargs...)
        s_dst = extract_dest_arraylike(dst_slice, regridder; kwargs...)
        initialize_regridding!(regridder, src_slice, s_src; kwargs...)
        perform_regridding!(s_dst, regridder, s_src; kwargs...)
        finalize_regridding!(dst_slice, regridder, s_dst; kwargs...)
    end
    return dst_arraylike
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
