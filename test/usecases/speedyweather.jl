using ConservativeRegridding
using SpeedyWeather
using GeoInterface
using GeometryOps
using Test

# Copied from SpeedyWeather.jl/ext/SpeedyWeatherGeoMakieExt/get_faces.jl
"""
Transpose (and copy) the 4 vertices of every grid point to obtain the faces of the grid.
Return a 6xN matrix `faces` of Point2{Float64} where the first 4 rows are the vertices (E, S, W, N)
of every grid points ij in 1:N, row 5 is duplicated north vertex to close the grid cell,
row 6 is NaN to separate grid cells when drawing them as a continuous line with `vec(faces)`."""
function get_faces(
    Grid::Type{<:AbstractGridArray},
    nlat_half::Integer;
    add_nan::Bool = false,
)
    npoints = RingGrids.get_npoints2D(Grid, nlat_half)

    # vertex east, south, west, north (i.e. clockwise for every grid point)
    E, S, W, N = RingGrids.get_vertices(Grid, nlat_half)

    @boundscheck size(N) == size(W) == size(S) == size(E) || throw(BoundsError("Vertices must have the same size"))
    @boundscheck size(N) == (2, npoints) || throw(BoundsError("Number of vertices and npoints do not agree"))

    # number of vertices = 4, 5 to close the polygon, 6 to add a nan
    # to prevent grid lines to be drawn between cells
    nvertices = add_nan ? 6 : 5

    # allocate faces as Point2{Float64} so that no data copy has to be made in Makie
    faces = Matrix{NTuple{2, Float64}}(undef, nvertices, npoints)

    @inbounds for ij in 1:npoints
        faces[1, ij] = (E[1, ij], E[2, ij])  # clockwise
        faces[2, ij] = (S[1, ij], S[2, ij])
        faces[3, ij] = (W[1, ij], W[2, ij])
        faces[4, ij] = (N[1, ij], N[2, ij])
        faces[5, ij] = (E[1, ij], E[2, ij])  # back to east to close the polygon
    end

    if add_nan  # add a NaN to separate grid cells
        for ij in 1:npoints
            faces[6, ij] = (NaN, NaN)
        end
    end

    return faces
end

get_faces(grid::AbstractGridArray; kwargs...) = get_faces(typeof(grid), grid.nlat_half; kwargs...)

# Now begin the test for real

src = rand(OctaHEALPixGrid, 5 + 100)
dst = rand(OctaminimalGaussianGrid, 5 + 100)

src_cells = get_faces(src)
dst_cells = get_faces(dst)

regridder = ConservativeRegridding.Regridder(dst_cells, src_cells)

A = regridder.intersections

# Now, let's perform some interpolation!
area1 = vec(sum(A, dims=1))
@test area1 == regridder.src_areas
area2 = vec(sum(A, dims=2))
@test area2 == regridder.dst_areas

values_on_grid1 = A * grid2 ./ area1
@test sum(values_on_grid1 .* area1) == sum(grid2 .* area2)

values_back_on_grid2 = A' * values_on_grid1 ./ area2
@test sum(values_back_on_grid2 .* area2) == sum(grid2 .* area2)
