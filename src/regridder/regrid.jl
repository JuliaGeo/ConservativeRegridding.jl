"""$(TYPEDSIGNATURES)
Default forward regrid: `dst = weight_matrix · src`. Used for any mapping that
does not need post-`mul!` normalization (currently `FVtoSE`, where the inverse
mass matrix is already baked into the weight matrix).
"""
function regrid!(dst_field::DenseVector, regridder::Regridder, src_field::DenseVector)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    return dst_field
end

"""$(TYPEDSIGNATURES)
Normalizing forward regrid: `dst = (weight_matrix · src) ./ mapping.dst_areas`.
Applies to any [`AbstractNormalizingMapping`](@ref) (currently `FVtoFV` and `SEtoFV`).

Mathematics: if `A` are the intersection-area / `B`-integral entries of the
weight matrix between source `s` and destination `d`, and `aᵈ` are the
destination cell areas, then

```math
d = (A s) / aᵈ
```

For `FVtoFV`, by construction `aᵈ = sum(A; dims=2)` and `aˢ = sum(A; dims=1)`.
"""
function regrid!(
    dst_field::DenseVector,
    regridder::Regridder{<:Any, <:AbstractNormalizingMapping},
    src_field::DenseVector,
)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    dst_field ./= regridder.mapping.dst_areas
    return dst_field
end

# For vectors non-continuous in memory we use the temporary dense arrays provided by the `Regridder`
# This ensures the sparse matrix-vector multiplication uses the correct (optimal) methods
function regrid!(dst_field::AbstractVector, regridder::Regridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(regridder.dst_temp, regridder, regridder.src_temp)
    dst_field .= regridder.dst_temp
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::Regridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::Regridder, src_field::DenseVector)
    regrid!(regridder.dst_temp, regridder, src_field)
    dst_field .= regridder.dst_temp
    return dst_field
end

# For n-dimensional arrays, iterate over slices along dimension `dims` (default: 1).
# Like `eachslice`, `dims` specifies the spatial dimension that the regridder operates on;
# all other dimensions are iterated over.
function regrid!(dst_field::AbstractArray, regridder::Regridder, src_field::AbstractArray; dims::Int=1)
    if ndims(src_field) == 1
        return regrid!(vec(dst_field), regridder, vec(src_field))
    end
    N = ndims(src_field)
    @assert 1 <= dims <= N "dims=$dims is out of range for a $(N)-dimensional array"
    # Collect axes for all dimensions except `dims`
    other_axes = ntuple(i -> axes(src_field, i < dims ? i : i + 1), N - 1)
    for I in CartesianIndices(other_axes)
        # Build full index tuple: Colon() at position `dims`, indices elsewhere
        idx = ntuple(N) do d
            if d == dims
                Colon()
            elseif d < dims
                I[d]
            else
                I[d - 1]
            end
        end
        src_slice = view(src_field, idx...)
        dst_slice = view(dst_field, idx...)
        regrid!(dst_slice, regridder, src_slice)
    end
    return dst_field
end

"""$(TYPEDSIGNATURES)
Regrid a vector `src_field` using `regridder`. Area vector for the output grid can
be passed on as optional argument to prevent recalculating it from the regridder."""
function regrid(
    regridder::AbstractMatrix,
    src_field::AbstractVector,
)
    n_out, n_in = size(regridder)

    # default Julia vector for now, type of destination field not known here
    dst_field = zeros(eltype(src_field), n_out)     
    return regrid!(dst_field, regridder, src_field)
end
