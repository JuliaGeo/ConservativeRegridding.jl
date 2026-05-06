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
