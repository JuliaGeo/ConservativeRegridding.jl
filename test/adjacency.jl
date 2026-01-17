using ConservativeRegridding
using ConservativeRegridding: compute_adjacency
using Test
import GeometryOps as GO
import GeoInterface as GI

@testset "Adjacency computation" begin
    @testset "Single cell grid (1x1)" begin
        # Create a 2x2 grid of points (1x1 cells) - the minimal grid
        points = [(Float64(i), Float64(j)) for i in 0:1, j in 0:1]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # Single cell has no neighbors
        @test length(adj) == 1
        @test isempty(adj[1])
    end

    @testset "2x2 grid (CellBasedGrid)" begin
        # Create a 3x3 grid of points (2x2 cells)
        points = [(Float64(i), Float64(j)) for i in 0:2, j in 0:2]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        @test length(adj) == 4

        # Cell layout (linear indices, column-major):
        # 2 4
        # 1 3
        # All cells in 2x2 are corners with 3 neighbors each
        @test sort(adj[1]) == [2, 3, 4]
        @test sort(adj[2]) == [1, 3, 4]
        @test sort(adj[3]) == [1, 2, 4]
        @test sort(adj[4]) == [1, 2, 3]

        # All cells have exactly 3 neighbors
        @test all(length(adj[i]) == 3 for i in 1:4)
    end

    @testset "3x3 grid (CellBasedGrid)" begin
        # Create a 4x4 grid of points (3x3 cells)
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        @test length(adj) == 9

        # Cell layout (linear indices, column-major):
        # 3 6 9
        # 2 5 8
        # 1 4 7

        # Corner cells (indices 1, 3, 7, 9) have 3 neighbors
        @test length(adj[1]) == 3  # bottom-left corner
        @test length(adj[3]) == 3  # top-left corner
        @test length(adj[7]) == 3  # bottom-right corner
        @test length(adj[9]) == 3  # top-right corner

        # Edge cells (indices 2, 4, 6, 8) have 5 neighbors
        @test length(adj[2]) == 5  # left edge
        @test length(adj[4]) == 5  # bottom edge
        @test length(adj[6]) == 5  # top edge
        @test length(adj[8]) == 5  # right edge

        # Interior cell (index 5) has 8 neighbors
        @test length(adj[5]) == 8

        # Verify specific neighbors for interior cell
        @test sort(adj[5]) == [1, 2, 3, 4, 6, 7, 8, 9]

        # Verify specific neighbors for corner cell
        @test sort(adj[1]) == [2, 4, 5]  # neighbors are above, right, and diagonal
        @test sort(adj[9]) == [5, 6, 8]  # neighbors are below, left, and diagonal
    end

    @testset "Larger grid (5x5)" begin
        # Create a 6x6 grid of points (5x5 cells)
        points = [(Float64(i), Float64(j)) for i in 0:5, j in 0:5]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        @test length(adj) == 25

        # Count cells by neighbor count
        corner_count = count(i -> length(adj[i]) == 3, 1:25)
        edge_count = count(i -> length(adj[i]) == 5, 1:25)
        interior_count = count(i -> length(adj[i]) == 8, 1:25)

        # 5x5 grid: 4 corners, 12 edge cells, 9 interior cells
        @test corner_count == 4
        @test edge_count == 12
        @test interior_count == 9
    end

    @testset "RegularGrid" begin
        # Test with RegularGrid
        x = 0.0:1.0:4.0
        y = 0.0:1.0:4.0
        grid = ConservativeRegridding.RegularGrid(GO.Planar(), x, y)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # 4x4 cells
        @test length(adj) == 16

        # Count cells by neighbor count
        corner_count = count(i -> length(adj[i]) == 3, 1:16)
        edge_count = count(i -> length(adj[i]) == 5, 1:16)
        interior_count = count(i -> length(adj[i]) == 8, 1:16)

        # 4x4 grid: 4 corners, 8 edge cells, 4 interior cells
        @test corner_count == 4
        @test edge_count == 8
        @test interior_count == 4

        # Interior cell 6 (i=2, j=2) has all 8 neighbors
        @test length(adj[6]) == 8
        @test sort(adj[6]) == [1, 2, 3, 5, 7, 9, 10, 11]
    end

    @testset "ExplicitPolygonGrid" begin
        # Create a 3x3 grid of explicit polygons
        polygons = Matrix{Any}(undef, 3, 3)
        for i in 1:3, j in 1:3
            x0, y0 = Float64(i - 1), Float64(j - 1)
            ring = GI.LinearRing([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1), (x0, y0)])
            polygons[i, j] = GI.Polygon([ring])
        end

        grid = ConservativeRegridding.ExplicitPolygonGrid(GO.Planar(), polygons)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # 3x3 cells
        @test length(adj) == 9

        # Count cells by neighbor count
        corner_count = count(i -> length(adj[i]) == 3, 1:9)
        edge_count = count(i -> length(adj[i]) == 5, 1:9)
        interior_count = count(i -> length(adj[i]) == 8, 1:9)

        @test corner_count == 4
        @test edge_count == 4
        @test interior_count == 1

        # Interior cell 5 has all 8 neighbors
        @test length(adj[5]) == 8
        @test sort(adj[5]) == [1, 2, 3, 4, 6, 7, 8, 9]
    end

    @testset "Symmetry of adjacency" begin
        # Verify that adjacency is symmetric: if A is neighbor of B, then B is neighbor of A
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        for i in 1:length(adj)
            for j in adj[i]
                @test i in adj[j]  # if j is neighbor of i, then i is neighbor of j
            end
        end
    end

    @testset "Non-square grid (3x5)" begin
        # Create a 4x6 grid of points (3x5 cells)
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:5]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        @test length(adj) == 15  # 3 * 5 = 15 cells

        # Count cells by neighbor count
        corner_count = count(i -> length(adj[i]) == 3, 1:15)
        edge_count = count(i -> length(adj[i]) == 5, 1:15)
        interior_count = count(i -> length(adj[i]) == 8, 1:15)

        # 3x5 grid: 4 corners, 2*(3-2) + 2*(5-2) = 2 + 6 = 8 edge cells, (3-2)*(5-2) = 3 interior cells
        @test corner_count == 4
        @test edge_count == 8
        @test interior_count == 3
    end

    @testset "Linear index mapping" begin
        # Verify the column-major linear index mapping
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)
        ni = 3  # number of cells in i direction

        # Cell at (i, j) has linear index: i + (j-1) * ni
        # Verify cell (2, 2) is interior with 8 neighbors
        idx_22 = 2 + (2 - 1) * ni  # = 5
        @test length(adj[idx_22]) == 8

        # Verify cell (1, 1) is corner with 3 neighbors
        idx_11 = 1 + (1 - 1) * ni  # = 1
        @test length(adj[idx_11]) == 3

        # Verify cell (3, 3) is corner with 3 neighbors
        idx_33 = 3 + (3 - 1) * ni  # = 9
        @test length(adj[idx_33]) == 3
    end
end
