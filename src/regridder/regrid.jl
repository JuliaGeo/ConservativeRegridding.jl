"""$(TYPEDSIGNATURES)
Regrid data on `src_field` onto `dst_field` conservatively (mean-preserving) using the `regridder` matrix.
`dst_area` is the area of each grid cell in `dst_field` and is used to normalize the result,
if not provided, will recompute this from `regridder`. `src_field` and `dst_field` can be any n-dimensional array,
in which case the regridding of the 1st dimension is broadcast to additional dimensions.

Mathematics of regridding: if A are the intersection areas between the respective grids of the fields d (dst) and s (src),
and aˢ and aᵈ are the areas of the source and destination grid cells, then ``d`` is computed via

```math
d = (A s) / aˢ 
```

Note that by construction,

```julia
aᵈ = sum(A; dims=2)
aˢ = sum(A; dims=1)
```
"""
function regrid!(dst_field::DenseVector, regridder::Regridder, src_field::DenseVector)
    areas = regridder.dst_areas # area of each grid cell
    LinearAlgebra.mul!(dst_field, regridder.intersections, src_field) # units of src_field times area of grid cell
    dst_field ./= areas # normalize by area of each grid cell
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

# For n-dimensional arrays, iterate over slices of the first dimension
function regrid!(dst_field::AbstractArray, regridder::Regridder, src_field::AbstractArray)
    if ndims(src_field) == 1
        return regrid!(vec(dst_field), regridder, vec(src_field))
    end
    for I in CartesianIndices(axes(src_field)[2:end])
        src_slice = view(src_field, :, I)
        dst_slice = view(dst_field, :, I)
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
