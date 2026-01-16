using ConservativeRegridding
using ConservativeRegridding: compute_adjacency
using Test
import GeometryOps as GO

@testset "Adjacency computation" begin
    @testset "Structured grid (3x3)" begin
        # Create a 3x3 grid of points (2x2 cells)
        points = [(Float64(i), Float64(j)) for i in 0:2, j in 0:2]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # Cell layout (linear indices):
        # 2 4
        # 1 3
        # Cell 1 (0,0): neighbors are 2, 3, 4 (8-connectivity)
        @test sort(adj[1]) == [2, 3, 4]
        # Cell 2 (1,0): neighbors are 1, 3, 4
        @test sort(adj[2]) == [1, 3, 4]
        # Cell 3 (0,1): neighbors are 1, 2, 4
        @test sort(adj[3]) == [1, 2, 4]
        # Cell 4 (1,1): neighbors are 1, 2, 3
        @test sort(adj[4]) == [1, 2, 3]
    end

    @testset "Structured grid (4x4 points = 3x3 cells)" begin
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # Center cell (index 5 in column-major: i=2, j=2) should have 8 neighbors
        # Linear index for (2,2): 2 + (2-1)*3 = 5
        @test length(adj[5]) == 8

        # Corner cell (1,1) -> index 1 should have 3 neighbors
        @test length(adj[1]) == 3

        # Edge cell (2,1) -> index 2 should have 5 neighbors
        @test length(adj[2]) == 5
    end
end
