using ConservativeRegridding
using ConservativeRegridding: compute_gradient_coefficients, GradientInfo, compute_adjacency
using Test
import GeometryOps as GO
import GeoInterface as GI

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

    @testset "1x1 grid - no valid gradients" begin
        # 2x2 points = 1x1 cells (single cell)
        points = [(Float64(i), Float64(j)) for i in 0:1, j in 0:1]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 1
        # Single cell has no neighbors, so it's invalid (< 3 neighbors)
        @test !grad_info[1].valid
        @test isempty(grad_info[1].neighbor_indices)
    end

    @testset "Conservation property - sum of coefficients is zero" begin
        # For valid gradient computations, the sum of all gradient coefficients
        # (source + all neighbors) should be zero (or very close to it)
        points = [(Float64(i), Float64(j)) for i in 0:5, j in 0:5]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        for idx in 1:length(grad_info)
            gi = grad_info[idx]
            if gi.valid
                # Sum all coefficients (source + neighbors)
                total_x = gi.src_grad[1]
                total_y = gi.src_grad[2]
                for ng in gi.neighbor_grads
                    total_x += ng[1]
                    total_y += ng[2]
                end
                @test abs(total_x) < 1e-10
                @test abs(total_y) < 1e-10
            end
        end
    end

    @testset "Centroid computation" begin
        # Verify centroids are computed correctly for uniform grids
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        ni = 4  # cells in i direction
        for idx in 1:16
            gi = grad_info[idx]
            # Convert linear index to (i, j)
            j_idx = div(idx - 1, ni) + 1
            i_idx = mod(idx - 1, ni) + 1
            # Expected centroid for unit square cells
            expected_x = i_idx - 0.5
            expected_y = j_idx - 0.5
            @test gi.centroid[1] ≈ expected_x atol=1e-10
            @test gi.centroid[2] ≈ expected_y atol=1e-10
        end
    end

    @testset "Invalid cells for insufficient neighbors" begin
        # Cells with < 3 neighbors are automatically invalid
        # 2x2 points = 1x1 cells
        points_1x1 = [(Float64(i), Float64(j)) for i in 0:1, j in 0:1]
        grid_1x1 = ConservativeRegridding.CellBasedGrid(GO.Planar(), points_1x1)
        tree_1x1 = ConservativeRegridding.TopDownQuadtreeCursor(grid_1x1)

        grad_info_1x1 = compute_gradient_coefficients(GO.Planar(), tree_1x1)

        # Single cell has 0 neighbors, should be invalid
        @test !grad_info_1x1[1].valid
        @test length(grad_info_1x1[1].neighbor_indices) == 0

        # Check that invalid cells have zero gradients
        @test grad_info_1x1[1].src_grad == (0.0, 0.0)
    end

    @testset "Invalid cells for centroid outside neighbor polygon" begin
        # Corner cells have 3 neighbors forming a triangle that doesn't contain the centroid
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # All 4 corner cells should be invalid
        @test !grad_info[1].valid   # (1,1)
        @test !grad_info[3].valid   # (3,1)
        @test !grad_info[7].valid   # (1,3)
        @test !grad_info[9].valid   # (3,3)

        # They should still have 3 neighbor indices
        @test length(grad_info[1].neighbor_indices) == 3
        @test length(grad_info[3].neighbor_indices) == 3
        @test length(grad_info[7].neighbor_indices) == 3
        @test length(grad_info[9].neighbor_indices) == 3
    end

    @testset "RegularGrid gradient computation" begin
        x = 0.0:1.0:5.0
        y = 0.0:1.0:5.0
        grid = ConservativeRegridding.RegularGrid(GO.Planar(), x, y)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 25

        # Interior cells should be valid
        # In 5x5, interior is 3x3 = 9 cells
        valid_count = count(gi -> gi.valid, grad_info)
        @test valid_count >= 9  # at least interior cells

        # Check conservation for valid cells
        for gi in grad_info
            if gi.valid
                total_x = gi.src_grad[1]
                total_y = gi.src_grad[2]
                for ng in gi.neighbor_grads
                    total_x += ng[1]
                    total_y += ng[2]
                end
                @test abs(total_x) < 1e-10
                @test abs(total_y) < 1e-10
            end
        end
    end

    @testset "ExplicitPolygonGrid gradient computation" begin
        # Create a 3x3 grid of explicit polygons
        polygons = Matrix{Any}(undef, 3, 3)
        for i in 1:3, j in 1:3
            x0, y0 = Float64(i - 1), Float64(j - 1)
            ring = GI.LinearRing([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1), (x0, y0)])
            polygons[i, j] = GI.Polygon([ring])
        end

        grid = ConservativeRegridding.ExplicitPolygonGrid(GO.Planar(), polygons)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 9

        # Center cell (index 5) should be valid
        @test grad_info[5].valid
        @test length(grad_info[5].neighbor_indices) == 8

        # Check conservation for center cell
        gi = grad_info[5]
        total_x = gi.src_grad[1]
        total_y = gi.src_grad[2]
        for ng in gi.neighbor_grads
            total_x += ng[1]
            total_y += ng[2]
        end
        @test abs(total_x) < 1e-10
        @test abs(total_y) < 1e-10
    end

    @testset "Gradient symmetry for uniform grid" begin
        # For a symmetric uniform grid, the center cell should have symmetric gradients
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # Center cell (index 6 for 4x4 grid, at position i=2, j=2)
        # With centroid at (1.5, 1.5)
        gi = grad_info[6]
        @test gi.valid

        # For symmetric interior cell, src_grad should be (0, 0) or very close
        @test abs(gi.src_grad[1]) < 1e-10
        @test abs(gi.src_grad[2]) < 1e-10

        # The neighbor gradients should exhibit rotational symmetry
        # Specifically, opposite neighbors should have opposite gradient contributions
        # But this depends on sorting order, so we just verify conservation
        total_x = gi.src_grad[1]
        total_y = gi.src_grad[2]
        for ng in gi.neighbor_grads
            total_x += ng[1]
            total_y += ng[2]
        end
        @test abs(total_x) < 1e-10
        @test abs(total_y) < 1e-10
    end

    @testset "Non-uniform grid spacing" begin
        # Test with non-uniform spacing to ensure gradients still work
        points = [(Float64(i)^1.5, Float64(j)^1.5) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 16

        # Check conservation still holds for valid cells
        for gi in grad_info
            if gi.valid
                total_x = gi.src_grad[1]
                total_y = gi.src_grad[2]
                for ng in gi.neighbor_grads
                    total_x += ng[1]
                    total_y += ng[2]
                end
                @test abs(total_x) < 1e-10
                @test abs(total_y) < 1e-10
            end
        end
    end

    @testset "Non-square grid (3x5)" begin
        # Create a 4x6 grid of points (3x5 cells)
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:5]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 15

        # Check that interior cells are valid
        # Interior cells for 3x5 grid are at i=2 for all interior j
        # Linear indices for (2, 2), (2, 3), (2, 4)
        ni = 3
        interior_indices = [2 + (2-1)*ni, 2 + (3-1)*ni, 2 + (4-1)*ni]  # [5, 8, 11]
        for idx in interior_indices
            @test grad_info[idx].valid
            @test length(grad_info[idx].neighbor_indices) == 8
        end

        # Check conservation for interior cells
        for idx in interior_indices
            gi = grad_info[idx]
            total_x = gi.src_grad[1]
            total_y = gi.src_grad[2]
            for ng in gi.neighbor_grads
                total_x += ng[1]
                total_y += ng[2]
            end
            @test abs(total_x) < 1e-10
            @test abs(total_y) < 1e-10
        end
    end

    @testset "GradientInfo constructor for invalid cells" begin
        # Test the invalid GradientInfo constructor
        centroid = (1.5, 1.5)
        neighbor_indices = [1, 2, 3]
        gi = GradientInfo{Float64}(centroid, neighbor_indices)

        @test !gi.valid
        @test gi.centroid == centroid
        @test gi.src_grad == (0.0, 0.0)
        @test gi.neighbor_indices == neighbor_indices
        @test length(gi.neighbor_grads) == 3
        @test all(ng == (0.0, 0.0) for ng in gi.neighbor_grads)
    end

    @testset "Edge cell validity patterns" begin
        # For a 4x4 grid, verify which edge cells are valid
        # Due to geometry, some edge cells may be valid if their 5 neighbors
        # form a pentagon that contains their centroid
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)
        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # Count edge cells (5 neighbors)
        edge_indices = [i for i in 1:16 if length(adj[i]) == 5]
        @test length(edge_indices) == 8  # 4x4 grid has 8 edge cells

        # Verify all edge cells have 5 neighbor indices in gradient info
        for idx in edge_indices
            @test length(grad_info[idx].neighbor_indices) == 5
        end
    end

    @testset "Neighbor sorting is counter-clockwise" begin
        # Verify neighbors are sorted counter-clockwise around the source centroid
        points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # For interior cell 6 at (1.5, 1.5), neighbors should be sorted by angle
        gi = grad_info[6]
        @test gi.valid

        # Compute angles from centroid to each neighbor centroid
        src_x, src_y = gi.centroid
        ni = 4
        angles = Float64[]
        for neighbor_idx in gi.neighbor_indices
            j_idx = div(neighbor_idx - 1, ni) + 1
            i_idx = mod(neighbor_idx - 1, ni) + 1
            neighbor_x = i_idx - 0.5
            neighbor_y = j_idx - 0.5
            angle = atan(neighbor_y - src_y, neighbor_x - src_x)
            push!(angles, angle)
        end

        # Angles should be in increasing order (with possible wrap-around)
        # Check that the angles are monotonically increasing (allowing for wrap)
        for i in 1:length(angles)-1
            diff = angles[i+1] - angles[i]
            # Allow for small numerical errors or the wrap-around case
            @test diff >= -1e-10 || (diff < -π)  # either increasing or wrapping from π to -π
        end
    end

    @testset "Large grid performance sanity check" begin
        # Test that gradient computation works for larger grids
        points = [(Float64(i), Float64(j)) for i in 0:20, j in 0:20]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        @test length(grad_info) == 400  # 20x20 cells

        # Count valid cells
        valid_count = count(gi -> gi.valid, grad_info)

        # Interior cells: (20-2)*(20-2) = 324
        # Some edge cells may also be valid
        @test valid_count >= 324

        # Spot check conservation for a few cells
        for idx in [105, 205, 305]  # some interior cells
            gi = grad_info[idx]
            if gi.valid
                total_x = gi.src_grad[1]
                total_y = gi.src_grad[2]
                for ng in gi.neighbor_grads
                    total_x += ng[1]
                    total_y += ng[2]
                end
                @test abs(total_x) < 1e-10
                @test abs(total_y) < 1e-10
            end
        end
    end
end
