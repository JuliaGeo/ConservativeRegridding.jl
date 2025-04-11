"""$(TYPEDSIGNATURES)
Regrid data on `src_field` onto `dst_field` conservativly (mean-preserving) using the `regridder` matrix.
`dst_area` is the area of each grid cell in `dst_field` and is used to normalize the result,
if not provided, will recompute this from `regridder`."""
Base.@propagate_inbounds function regrid!(
    dst_field::AbstractVector,
    regridder::AbstractMatrix,
    src_field::AbstractVector,
    dst_area::AbstractVector,
)
    @boundscheck (size(regridder, 2) == length(dst_field) && size(regridder, 1) == length(src_field)) &&
        @warn "regridder of size $(size(regridder)) matches input grid $(length(src_field)) and output grid $(length(dst_field)) when transposed."*
                " You may want to regrid with `transpose(regridder)` instead."

    @boundscheck if length(dst_field) != length(dst_area)
        @warn "Area vector of length $(length(dst_area)) is incompatible with output grid vector of length $(length(dst_field))."
    end
    
    # the actual regridding is a matrix-vector multiplication with the regridder, do in-place
    LinearAlgebra.mul!(dst_field, regridder, src_field) # units of src_field times area of grid cell

    return dst_field ./= dst_area # normalize by area of each grid cell
end

"""$(TYPEDSIGNATURES)
regrid a vector `src_field` using `regridder` and writes the result in `dst_field`.
Recomputes the area vector for the output grid from `regridder`. Pass on as optional argument to avoid this."""
function regrid!(
    dst_field::AbstractVector,
    regridder::AbstractMatrix,
    src_field::AbstractVector,
)
    dst_area = cell_area(regridder, :out)
    regrid!(dst_field, src_field, regridder, dst_area)
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
    return regrid!(dst_field, src_field, regridder, args...)
end