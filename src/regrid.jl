"""$(TYPEDSIGNATURES)
Regrid data on `grid_in` onto `grid_out` conservativly (mean-preserving) using the `regridder` matrix.
`area_out` is the area of each grid cell in `grid_out` and is used to normalize the result,
if not provided, will recompute this from `regridder`."""
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
    
    # the actual regridding is a matrix-vector multiplication with the regridder, do in-place
    LinearAlgebra.mul!(grid_out, regridder, grid_in)    # units of grid_in times area of grid cell
    grid_out ./= area_out                               # normalize by area of each grid cell
end

"""$(TYPEDSIGNATURES)
regrid a vector `grid_in` using `regridder` and writes the result in `grid_out`.
Recomputes the area vector for the output grid from `regridder`. Pass on as optional argument to avoid this."""
function regrid!(
    grid_out::AbstractVector,
    grid_in::AbstractVector,
    regridder::AbstractMatrix,
)
    area_out = area(regridder, dims=:out)
    regrid!(grid_out, grid_in, regridder, area_out)
end

"""$(TYPEDSIGNATURES)
regrid a vector `grid_in` using `regridder`. Area vector for the output grid can
be passed on as optional argument to prevent recalculating it from the regridder."""
function regrid(
    grid_in::AbstractVector,
    regridder::AbstractMatrix,
    args...
)
    n_out, n_in = size(regridder)
    grid_out = similar(grid_in, n_out)
    regrid!(grid_out, grid_in, regridder, args...)
end