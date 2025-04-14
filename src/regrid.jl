"""$(TYPEDSIGNATURES)
Regrid data on `src_field` onto `dst_field` conservativly (mean-preserving) using the `regridder` matrix.
`dst_area` is the area of each grid cell in `dst_field` and is used to normalize the result,
if not provided, will recompute this from `regridder`."""
function regrid!(field2::AbstractVector, regridder::Regridder, field1::AbstractVector)

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

    N2, N1 = size(regridder.intersections)
    forward = N2 == length(field2) && N1 == length(field1)

    if forward
        A = regridder.intersections # transpose the regridder
        areas = regridder.src_areas # area of each grid cell
        LinearAlgebra.mul!(field1, A, field2) # units of src_field times area of grid cell
    else
        A = transpose(regridder.intersections) # transpose the regridder
        areas = regridder.dst_areas # area of each grid cell
        LinearAlgebra.mul!(field1, A, field2) # units of src_field times area of grid cell
    end

    field1 ./= areas # normalize by area of each grid cell

    return field1
end

"""$(TYPEDSIGNATURES)
regrid a vector `src_field` using `regridder`. Area vector for the output grid can
be passed on as optional argument to prevent recalculating it from the regridder."""
function regrid(
    src_field::AbstractVector,
    regridder::AbstractMatrix,
    args...
)
    n_out, n_in = size(regridder)

    @boundscheck if n_in != length(src_field)
        @warn "Regridder of size $(size(regridder)) does not match input grid of length $(length(src_field))."
    end

    @boundscheck if n_out == length(src_field)
        @warn "Regridder of size $(size(regridder)) matches input grid of length $(length(src_field)) if transposed." *
            " You may want to regrid with `transpose(regridder)` instead."
    end

    dst_field = zeros(eltype(src_field), n_out) # default Julia vector for now
    return regrid!(dst_field, regridder, src_field, args...)
end