"""$(TYPEDSIGNATURES)
Regrid data on `src_field` onto `dst_field` conservativly (mean-preserving) using the `regridder` matrix.
`dst_area` is the area of each grid cell in `dst_field` and is used to normalize the result,
if not provided, will recompute this from `regridder`."""
function regrid!(dst_field::AbstractVector, regridder::Regridder, src_field::AbstractVector)

    # Mathematics of regridding: if A are the inersection areas between
    # the respective grids of the fields d (dst) and s (src),
    # and aˢ and aᵈ are the areas of the source and destination grid cells,
    # 
    # d = (A * s) ./ aˢ # regrid from s to d
    #
    # and
    #
    # s = (Aᵀ * d) ./ aᵈ # regrid from d to s
    #
    # Note that by construction,
    #
    # aᵈ = sum(A, 2)
    # aˢ = sum(A, 1)

    areas = regridder.dst_areas # area of each grid cell
    LinearAlgebra.mul!(dst_field, regridder.intersections, src_field) # units of src_field times area of grid cell
    dst_field ./= areas # normalize by area of each grid cell
    return dst_field
end

"""$(TYPEDSIGNATURES)
regrid a vector `src_field` using `regridder`. Area vector for the output grid can
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