"""$(TYPEDSIGNATURES)
Regrid data on `field_in` onto `field_out` conservativly (mean-preserving) using the `regridder` matrix.
`area_out` is the area of each grid cell in `field_out` and is used to normalize the result,
if not provided, will recompute this from `regridder`."""
Base.@propagate_inbounds function regrid!(
    field_out::AbstractVector,
    field_in::AbstractVector,
    regridder::AbstractMatrix,
    area_out::AbstractVector,
)
    @boundscheck (size(regridder, 2) == length(field_out) && size(regridder, 1) == length(field_in)) &&
        @warn "regridder of size $(size(regridder)) matches input grid $(length(field_in)) and output grid $(length(field_out)) when transposed."*
                " You may want to regrid with `transpose(regridder)` instead."

    @boundscheck if length(field_out) != length(area_out)
        @warn "Area vector of length $(length(area_out)) is incompatible with output grid vector of length $(length(field_out))."
    end
    
    # the actual regridding is a matrix-vector multiplication with the regridder, do in-place
    LinearAlgebra.mul!(field_out, regridder, field_in)    # units of field_in times area of grid cell
    field_out ./= area_out                               # normalize by area of each grid cell
end

"""$(TYPEDSIGNATURES)
regrid a vector `field_in` using `regridder` and writes the result in `field_out`.
Recomputes the area vector for the output grid from `regridder`. Pass on as optional argument to avoid this."""
function regrid!(
    field_out::AbstractVector,
    field_in::AbstractVector,
    regridder::AbstractMatrix,
)
    area_out = cell_area(regridder, :out)
    regrid!(field_out, field_in, regridder, area_out)
end

"""$(TYPEDSIGNATURES)
regrid a vector `field_in` using `regridder`. Area vector for the output grid can
be passed on as optional argument to prevent recalculating it from the regridder."""
function regrid(
    field_in::AbstractVector,
    regridder::AbstractMatrix,
    args...
)
    n_out, n_in = size(regridder)
    @boundscheck if n_in != length(field_in)
        @warn "Regridder of size $(size(regridder)) does not match input grid of length $(length(field_in))."
    end

    @boundscheck if n_out == length(field_in)
        @warn "Regridder of size $(size(regridder)) matches input grid of length $(length(field_in)) if transposed." *
            " You may want to regrid with `transpose(regridder)` instead."
    end

    field_out = zeros(eltype(field_in), n_out)    # default Julia vector for now
    regrid!(field_out, field_in, regridder, args...)
end