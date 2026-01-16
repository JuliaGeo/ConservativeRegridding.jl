using ConservativeRegridding
using ConservativeRegridding: compute_gradient_coefficients, GradientInfo
using Test
import GeometryOps as GO

@testset "Gradient computation" begin
    @testset "Simple 3x3 grid" begin
        # Create a 4x4 grid of points (3x3 cells)
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # Should have gradient info for each cell
        @test length(grad_info) == 9

        # Center cell (index 5) should have valid gradients (8 neighbors surrounding it)
        @test grad_info[5].valid
        @test length(grad_info[5].neighbor_indices) == 8

        # Corner cell (index 1) has 3 neighbors but they form a triangle
        # that doesn't contain the corner centroid, so it's invalid
        # This is expected - boundary cells typically have invalid gradients
        @test length(grad_info[1].neighbor_indices) == 3
        # Corner cells are invalid because centroid is outside neighbor polygon
        @test !grad_info[1].valid
    end

    @testset "2x2 grid - edge case" begin
        # 3x3 points = 2x2 cells
        points = [(Float64(i), Float64(j)) for i in 0:2, j in 0:2]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # All cells have exactly 3 neighbors
        for i in 1:4
            @test length(grad_info[i].neighbor_indices) == 3
        end

        # In a 2x2 grid, all cells are corner cells, so no centroid is inside
        # the triangle formed by its 3 neighbors. All should be invalid.
        for i in 1:4
            @test !grad_info[i].valid
        end
    end

    @testset "Larger grid - interior cells valid" begin
        # Create a 5x5 grid of points (4x4 cells)
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # Should have gradient info for each cell
        @test length(grad_info) == 16

        # Interior cells (those with 8 neighbors) should be valid
        # In a 4x4 grid, the interior 2x2 cells are at linear indices:
        # Cell (2,2) = 2 + (2-1)*4 = 6
        # Cell (3,2) = 3 + (2-1)*4 = 7
        # Cell (2,3) = 2 + (3-1)*4 = 10
        # Cell (3,3) = 3 + (3-1)*4 = 11
        for idx in [6, 7, 10, 11]
            @test grad_info[idx].valid
            @test length(grad_info[idx].neighbor_indices) == 8
        end
    end
end
