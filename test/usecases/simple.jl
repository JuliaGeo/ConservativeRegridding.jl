using ConservativeRegridding
using ConservativeRegridding: Trees
using Test

import GeometryOps as GO, GeoInterface as GI
import SortTileRecursiveTree
using SparseArrays

# Target grid
polys1 = begin
    gridpoints = [(i, j) for i in 0:2, j in 0:2]
    [GI.Polygon([GI.LinearRing([gridpoints[i, j], gridpoints[i, j+1], gridpoints[i+1, j+1], gridpoints[i+1, j], gridpoints[i, j]])]) for i in 1:size(gridpoints, 1)-1, j in 1:size(gridpoints, 2)-1] |> vec
end

# Source grid
polys2 = begin
    diamondpoly = GI.Polygon([GI.LinearRing([(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)])])
    trianglepolys = GI.Polygon.([
        [GI.LinearRing([(0, 0), (1, 0), (0, 1), (0, 0)])],
        [GI.LinearRing([(0, 1), (0, 2), (1, 2), (0, 1)])],
        [GI.LinearRing([(1, 2), (2, 1), (2, 2), (1, 2)])],
        [GI.LinearRing([(2, 1), (2, 0), (1, 0), (2, 1)])],
    ])
    [diamondpoly, trianglepolys...]
end

tree1 = GO.SpatialTreeInterface.FlatNoTree(polys1)
tree2 = GO.SpatialTreeInterface.FlatNoTree(polys2)

@testset "Simple Regridding" begin
    # Construct a regridder from grid2 to grid1
    R = ConservativeRegridding.Regridder(GO.Planar(), tree1, tree2; normalize=false)
    A = R.intersections
    # Now, let's perform some interpolation!
    area1 = vec(sum(A, dims=2))
    @test area1 == GO.area.(polys1)
    area2 = vec(sum(A, dims=1))
    @test area2 == GO.area.(polys2)

    values_on_grid2 = [0, 0, 5, 0, 0]

    # Regrid from the source grid2 to the target grid1
    values_on_grid1 = A * values_on_grid2 ./ area1
    @test sum(values_on_grid1 .* area1) == sum(values_on_grid2 .* area2)

    # Regrid from the target grid1 to the source grid2 using the transpose of A
    values_back_on_grid2 = A' * values_on_grid1 ./ area2
    @test sum(values_back_on_grid2 .* area2) == sum(values_on_grid2 .* area2)
    # We can see here that some data has diffused into the central diamond cell of grid2,
    # since it was overlapped by the top left cell of grid1.\
end

# Test transpose functionality with temporary vectors
@testset "Transposed Regridder" begin
    regridder = ConservativeRegridding.Regridder(GO.Planar(), tree1, tree2; normalize=false)
    regridder_T = transpose(regridder)

    # Verify that transpose swaps the areas
    @test regridder_T.src_areas === regridder.dst_areas
    @test regridder_T.dst_areas === regridder.src_areas

    # Verify that transpose swaps the temporary vectors too
    @test regridder_T.src_temp === regridder.dst_temp
    @test regridder_T.dst_temp === regridder.src_temp

    # Test regridding in reverse direction
    src_on_grid1 = Float64[1, 2, 3, 4]
    dst_on_grid2 = zeros(Float64, length(polys2))
    ConservativeRegridding.regrid!(dst_on_grid2, regridder_T, src_on_grid1)
    @test sum(dst_on_grid2 .* regridder_T.dst_areas) ≈ sum(src_on_grid1 .* regridder_T.src_areas)
end

@testset "Regridding with STRtrees" begin
    # Construct STRtrees from the polygons
    str1 = SortTileRecursiveTree.STRtree(polys1)
    str2 = SortTileRecursiveTree.STRtree(polys2)
    # Wrap the STRtrees in a GeometryMaintainingTreeWrapper, so we can carry the geometries around.
    str_tree1 = Trees.GeometryMaintainingTreeWrapper(polys1, str1)
    str_tree2 = Trees.GeometryMaintainingTreeWrapper(polys2, str2)

    R = ConservativeRegridding.Regridder(GO.Planar(), str_tree1, str_tree2; normalize=false)
    A = R.intersections
    area1 = vec(sum(A, dims=2))
    @test area1 == GO.area.(polys1)
    area2 = vec(sum(A, dims=1))
    @test area2 == GO.area.(polys2)
end