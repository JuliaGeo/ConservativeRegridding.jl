Base.@propagate_inbounds function regrid!(
    grid_out::AbstractVector,
    grid_in::AbstractVector,
    regridder::AbstractMatrix,
    area_out::AbstractVector,
)
    @boundscheck (size(regridder, 2) == length(grid_out) && size(regridder, 1) == length(grid_in)) &&
        @warn "regridder of size $(size(regridder)) matches input grid $(length(grid_in)) and output grid $(length(grid_out)) when transposed."*
                " You may want to regrid with `transpose(regridder)` instead."

    @boundscheck if size(grid_out, 1) != length(area_out)
        @warn "Area vector of length $(length(area_out)) is incompatible with output grid vector of length $(length(grid_out))." *
            " Recomputing the area vector from the regridder."
        regrid!(grid_out, grid_in, regridder)
    end
    
    LinearAlgebra.mul!(grid_out, regridder, grid_in)    # units of grid_in times area of grid cell
    grid_out ./= area_out                               # normalize by area of each grid cell
end

"""$(TYPEDSIGNATURES)
regrid a vector `gridin` using the regridder `regridder` and writes the result in `gridout`.
Recomputes the area vector for the output grid can be passed on as optional argument to prevent recalculating it from
the regridder."""
function regrid!(
    grid_out::AbstractVector,
    grid_in::AbstractVector,
    regridder::AbstractMatrix,
)
    area_out = area(regridder, dims=:out)
    regrid!(grid_out, grid_in, regridder, area_out)
end

"""$(TYPEDSIGNATURES)
regrid a vector `gridin` using the regridder `regridder`. Area vector for the output grid can
be passed on as optional argument to prevent recalculating it from the regridder."""
function regrid(
    grid_in::AbstractVector,
    regridder::AbstractMatrix,
    args...
)
    n_out, n_in = size(regridder)
    grid_out = typeof(grid_in)(undef, n_out)
    regrid!(grid_out, grid_in, regridder, args...)
end