using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
using SparseArrays

# Target grid
grid1 = begin
    gridpoints = [(i, j) for i in 0:2, j in 0:2]
    [GI.Polygon([GI.LinearRing([gridpoints[i, j], gridpoints[i, j+1], gridpoints[i+1, j+1], gridpoints[i+1, j], gridpoints[i, j]])]) for i in 1:size(gridpoints, 1)-1, j in 1:size(gridpoints, 2)-1] |> vec
end

# Source grid
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

# Construct a regridder from grid2 to grid1
A = @test_nowarn ConservativeRegridding.intersection_areas(grid1, grid2)

# Now, let's perform some interpolation!
area1 = vec(sum(A, dims=2))
@test area1 == GO.area.(grid1)
area2 = vec(sum(A, dims=1))
@test area2 == GO.area.(grid2)

values_on_grid2 = [0, 0, 5, 0, 0]

# Regrid from the source grid2 to the target grid1
values_on_grid1 = A * values_on_grid2 ./ area1
@test sum(values_on_grid1 .* area1) == sum(values_on_grid2 .* area2)

# Regrid from the target grid1 to the source grid2 using the transpose of A
values_back_on_grid2 = A' * values_on_grid1 ./ area2
@test sum(values_back_on_grid2 .* area2) == sum(values_on_grid2 .* area2)
# We can see here that some data has diffused into the central diamond cell of grid2,
# since it was overlapped by the top left cell of grid1.

# Test the Regridder struct with temporary vectors
@testset "Regridder with temporary vectors" begin
    # Create vertex arrays (not polygons) for the Regridder constructor
    gridpoints = [(i, j) for i in 0:2, j in 0:2]
    grid1_vertices = [[gridpoints[i, j], gridpoints[i, j+1], gridpoints[i+1, j+1], gridpoints[i+1, j], gridpoints[i, j]] for i in 1:size(gridpoints, 1)-1, j in 1:size(gridpoints, 2)-1] |> vec

    grid2_vertices = [
        [(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)],  # diamond
        [(0, 0), (1, 0), (0, 1), (0, 0)],          # triangle 1
        [(0, 1), (0, 2), (1, 2), (0, 1)],          # triangle 2
        [(1, 2), (2, 1), (2, 2), (1, 2)],          # triangle 3
        [(2, 1), (2, 0), (1, 0), (2, 1)]           # triangle 4
    ]

    regridder = ConservativeRegridding.Regridder(grid1_vertices, grid2_vertices; normalize=false)

    # Test that temporary vectors are properly initialized
    @test length(regridder.dst_temp) == length(grid1_vertices)
    @test length(regridder.src_temp) == length(grid2_vertices)
    @test all(regridder.dst_temp .== 0)
    @test all(regridder.src_temp .== 0)

    # Test regridding with dense vectors (should use optimized path)
    src_dense = Float64[0, 0, 5, 0, 0]
    dst_dense = zeros(Float64, length(grid1_vertices))
    ConservativeRegridding.regrid!(dst_dense, regridder, src_dense)
    @test sum(dst_dense .* regridder.dst_areas) ≈ sum(src_dense .* regridder.src_areas)

    # Test regridding with non-contiguous arrays (should use temporary vectors)
    # Create a view to simulate non-contiguous memory
    src_matrix = reshape(Float64[0, 0, 5, 0, 0, 1, 2, 3, 4, 5], 5, 2)
    src_view = view(src_matrix, :, 1)
    dst_view = view(zeros(Float64, length(grid1_vertices), 2), :, 1)

    ConservativeRegridding.regrid!(dst_view, regridder, src_view)
    @test sum(dst_view .* regridder.dst_areas) ≈ sum(src_view .* regridder.src_areas)

    # Test mixed cases: dense dst, non-contiguous src
    dst_dense2 = zeros(Float64, length(grid1_vertices))
    ConservativeRegridding.regrid!(dst_dense2, regridder, src_view)
    @test dst_dense2 ≈ dst_view

    # Test mixed cases: non-contiguous dst, dense src
    dst_view2 = view(zeros(Float64, length(grid1_vertices), 2), :, 1)
    ConservativeRegridding.regrid!(dst_view2, regridder, src_dense)
    @test sum(dst_view2 .* regridder.dst_areas) ≈ sum(src_dense .* regridder.src_areas)
end

# Test transpose functionality with temporary vectors
@testset "Transposed Regridder" begin
    # Create vertex arrays for the Regridder constructor
    gridpoints = [(i, j) for i in 0:2, j in 0:2]
    grid1_vertices = [[gridpoints[i, j], gridpoints[i, j+1], gridpoints[i+1, j+1], gridpoints[i+1, j], gridpoints[i, j]] for i in 1:size(gridpoints, 1)-1, j in 1:size(gridpoints, 2)-1] |> vec

    grid2_vertices = [
        [(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)],  # diamond
        [(0, 0), (1, 0), (0, 1), (0, 0)],          # triangle 1
        [(0, 1), (0, 2), (1, 2), (0, 1)],          # triangle 2
        [(1, 2), (2, 1), (2, 2), (1, 2)],          # triangle 3
        [(2, 1), (2, 0), (1, 0), (2, 1)]           # triangle 4
    ]

    regridder = ConservativeRegridding.Regridder(grid1_vertices, grid2_vertices; normalize=false)
    regridder_T = transpose(regridder)

    # Verify that transpose swaps the areas
    @test regridder_T.src_areas === regridder.dst_areas
    @test regridder_T.dst_areas === regridder.src_areas

    # Verify that transpose swaps the temporary vectors too
    @test regridder_T.src_temp === regridder.dst_temp
    @test regridder_T.dst_temp === regridder.src_temp

    # Test regridding in reverse direction
    src_on_grid1 = Float64[1, 2, 3, 4]
    dst_on_grid2 = zeros(Float64, length(grid2_vertices))
    ConservativeRegridding.regrid!(dst_on_grid2, regridder_T, src_on_grid1)
    @test sum(dst_on_grid2 .* regridder_T.dst_areas) ≈ sum(src_on_grid1 .* regridder_T.src_areas)
end
