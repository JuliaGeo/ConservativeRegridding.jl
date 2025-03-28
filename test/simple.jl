using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI


grid1 = begin
    gridpoints = [(i, j) for i in 0:2, j in 0:2]
    [GI.Polygon([GI.LinearRing([gridpoints[i, j], gridpoints[i, j+1], gridpoints[i+1, j+1], gridpoints[i+1, j], gridpoints[i, j]])]) for i in 1:size(gridpoints, 1)-1, j in 1:size(gridpoints, 2)-1] |> vec
end

grid2 = begin
    diamondpoly = GI.Polygon([GI.LinearRing([(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)])])
    trianglepolys = GI.Polygon.([
        [GI.LinearRing([(0, 0), (1, 0), (0, 1), (0, 0)])],
        [GI.LinearRing([(0, 1), (0, 2), (1, 2), (0, 1)])],
        [GI.LinearRing([(1, 2), (2, 1), (2, 2), (1, 2)])],
        [GI.LinearRing([(2, 1), (2, 0), (1, 0), (2, 1)])],
    ])
    [diamondpoly, trianglepolys...]
end

A = @test_nowarn ConservativeRegridding.area_of_intersection_operator(grid1, grid2)

# Now, let's perform some interpolation!
area1 = vec(sum(A, dims=2))
@test area1 == GO.area.(grid1)
area2 = vec(sum(A, dims=1))
@test area2 == GO.area.(grid2)

values_on_grid2 = [0, 0, 5, 0, 0]

values_on_grid1 = A * values_on_grid2 ./ area1
@test sum(values_on_grid1 .* area1) == sum(values_on_grid2 .* area2)

values_back_on_grid2 = A' * values_on_grid1 ./ area2
@test sum(values_back_on_grid2 .* area2) == sum(values_on_grid2 .* area2)
# We can see here that some data has diffused into the central diamond cell of grid2,
# since it was overlapped by the top left cell of grid1.
