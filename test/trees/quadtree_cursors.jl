using ConservativeRegridding.Trees
using Test
import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI

# Helper to build a matrix of lon/lat points covering the globe
function make_lonlat_point_matrix(nx, ny)
    lons = range(-180, 180, length=nx+1)
    lats = range(-90, 90, length=ny+1)
    return [(lon, lat) for lon in lons, lat in lats]
end

function make_unitspherical_point_matrix(nx, ny)
    lons = range(-180, 180, length=nx+1)
    lats = range(-90, 90, length=ny+1)
    return [GO.UnitSpherical.UnitSphereFromGeographic()((lon, lat)) for lon in lons, lat in lats]
end

# Helper to count all leaf cells reachable from a cursor
function count_leaves(cursor)
    if STI.isleaf(cursor)
        return length(collect(STI.child_indices_extents(cursor)))
    else
        total = 0
        for i in 1:STI.nchild(cursor)
            total += count_leaves(STI.getchild(cursor, i))
        end
        return total
    end
end

# Helper to verify tree can be fully traversed without error
function traverse_tree(cursor)
    if STI.isleaf(cursor)
        return true
    end
    for i in 1:STI.nchild(cursor)
        child = STI.getchild(cursor, i)
        traverse_tree(child)
    end
    return true
end

# Create test grids
function make_cellbased_grid(nx, ny)
    CellBasedGrid(GO.Spherical(), make_unitspherical_point_matrix(nx, ny))
end

function make_regular_grid(nx, ny)
    lons = collect(range(-180.0, 180.0, length=nx+1))
    lats = collect(range(-90.0, 90.0, length=ny+1))
    RegularGrid(GO.Spherical(), lons, lats)
end

@testset "STI dual_depth_first_search - self intersection" begin
    # This test verifies that when querying a grid against itself,
    # each cell is found to intersect itself (diagonal of intersection matrix)
    for (nx, ny) in [(4, 4), (16, 16), (13, 17)]
        @testset "$(nx)×$(ny) grid" begin
            grid = make_regular_grid(nx, ny)
            cursor1 = QuadtreeCursor(grid)
            cursor2 = QuadtreeCursor(grid)

            # Collect all intersecting pairs found by dual tree search
            # Note: Use GO.UnitSpherical._intersects for SphericalCap intersection
            found_pairs = Set{Tuple{Int,Int}}()
            STI.dual_depth_first_search(GO.UnitSpherical._intersects, cursor1, cursor2) do i1, i2
                push!(found_pairs, (i1, i2))
            end

            # Every cell should intersect itself (diagonal entries)
            total_cells = nx * ny
            for i in 1:total_cells
                @test (i, i) in found_pairs
            end

            # Total pairs found should be at least the diagonal
            @test length(found_pairs) >= total_cells
        end
    end
end

@testset "QuadtreeCursor" begin
    @testset "Basic construction" begin
        grid = make_cellbased_grid(16, 16)
        cursor = QuadtreeCursor(grid)

        @test cursor.grid === grid
        @test cursor.idx == CartesianIndex(1, 1)
        @test cursor.level >= 1
    end

    @testset "STI compliance - CellBasedGrid" begin
        for (nx, ny) in [(16, 16), (13, 17), (3, 5)]
            @testset "$(nx)×$(ny) grid" begin
                grid = make_cellbased_grid(nx, ny)
                cursor = QuadtreeCursor(grid)

                # isspatialtree
                @test STI.isspatialtree(typeof(cursor)) == true

                # Root should not be a leaf for grids > 2×2
                if nx > 2 && ny > 2
                    @test STI.isleaf(cursor) == false
                end

                # nchild returns valid count
                if !STI.isleaf(cursor)
                    nc = STI.nchild(cursor)
                    @test nc >= 1
                    @test nc <= 4
                end

                # getchild returns valid cursor
                if !STI.isleaf(cursor)
                    child = STI.getchild(cursor, 1)
                    @test child isa QuadtreeCursor
                    @test child.grid === grid
                    @test child.level == cursor.level - 1
                end

                # getchild throws for invalid index
                if !STI.isleaf(cursor)
                    nc = STI.nchild(cursor)
                    @test_throws ArgumentError STI.getchild(cursor, nc + 1)
                end

                # node_extent returns SphericalCap
                extent = STI.node_extent(cursor)
                @test extent isa GO.UnitSpherical.SphericalCap

                # Full traversal succeeds
                @test traverse_tree(cursor) == true

                # All leaves reachable, count matches grid size
                @test count_leaves(cursor) == nx * ny
            end
        end
    end

    @testset "STI compliance - RegularGrid" begin
        for (nx, ny) in [(16, 16), (13, 17), (3, 5)]
            @testset "$(nx)×$(ny) grid" begin
                grid = make_regular_grid(nx, ny)
                cursor = QuadtreeCursor(grid)

                @test STI.isspatialtree(typeof(cursor)) == true

                if nx > 2 && ny > 2
                    @test STI.isleaf(cursor) == false
                end

                # node_extent returns SphericalCap
                extent = STI.node_extent(cursor)
                @test extent isa GO.UnitSpherical.SphericalCap

                # Full traversal succeeds
                @test traverse_tree(cursor) == true

                # All leaves reachable
                @test count_leaves(cursor) == nx * ny
            end
        end
    end

    @testset "leaf_idxs correctness" begin
        grid = make_cellbased_grid(8, 8)
        cursor = QuadtreeCursor(grid)

        # At root, leaf_idxs should cover entire grid
        irange, jrange = Trees.leaf_idxs(cursor)
        @test first(irange) == 1
        @test last(irange) == 8
        @test first(jrange) == 1
        @test last(jrange) == 8
    end
end

@testset "STI dual_depth_first_search - TopDownQuadtreeCursor self intersection" begin
    # This test verifies that when querying a grid against itself,
    # each cell is found to intersect itself (diagonal of intersection matrix)
    for (nx, ny) in [(4, 4), (16, 16), (13, 17)]
        @testset "$(nx)×$(ny) grid" begin
            grid = make_regular_grid(nx, ny)
            cursor1 = TopDownQuadtreeCursor(grid)
            cursor2 = TopDownQuadtreeCursor(grid)

            # Collect all intersecting pairs found by dual tree search
            # Note: TopDownQuadtreeCursor returns (i, j) tuples, not CartesianIndex
            found_pairs = Set{Tuple{Int,Int}}()
            STI.dual_depth_first_search(GO.UnitSpherical._intersects, cursor1, cursor2) do idx1, idx2
                push!(found_pairs, (idx1, idx2))
            end

            # Every cell should intersect itself (diagonal entries)
            total_cells = nx * ny
            for i in 1:total_cells
                @test (i, i) in found_pairs
            end

            # Total pairs found should be at least the diagonal
            @test length(found_pairs) >= total_cells
        end
    end
end

@testset "TopDownQuadtreeCursor" begin
    @testset "Basic construction" begin
        grid = make_cellbased_grid(16, 16)
        cursor = TopDownQuadtreeCursor(grid)

        @test cursor.grid === grid
        @test cursor.leafranges == (1:16, 1:16)
    end

    @testset "STI compliance - CellBasedGrid" begin
        for (nx, ny) in [(16, 16), (13, 17), (3, 5)]
            @testset "$(nx)×$(ny) grid" begin
                grid = make_cellbased_grid(nx, ny)
                cursor = TopDownQuadtreeCursor(grid)

                # isspatialtree
                @test STI.isspatialtree(typeof(cursor)) == true

                # Root should not be a leaf for grids > 2×2
                if nx > 2 && ny > 2
                    @test STI.isleaf(cursor) == false
                end

                # nchild returns valid count
                if !STI.isleaf(cursor)
                    nc = STI.nchild(cursor)
                    @test nc >= 2
                    @test nc <= 4
                end

                # getchild returns valid cursor
                if !STI.isleaf(cursor)
                    child = STI.getchild(cursor, 1)
                    @test child isa TopDownQuadtreeCursor
                    @test child.grid === grid
                end

                # node_extent returns SphericalCap
                extent = STI.node_extent(cursor)
                @test extent isa GO.UnitSpherical.SphericalCap

                # Full traversal succeeds
                @test traverse_tree(cursor) == true

                # All leaves reachable, count matches grid size
                @test count_leaves(cursor) == nx * ny
            end
        end
    end

    @testset "STI compliance - RegularGrid" begin
        for (nx, ny) in [(16, 16), (13, 17), (3, 5)]
            @testset "$(nx)×$(ny) grid" begin
                grid = make_regular_grid(nx, ny)
                cursor = TopDownQuadtreeCursor(grid)

                @test STI.isspatialtree(typeof(cursor)) == true

                # node_extent returns SphericalCap
                extent = STI.node_extent(cursor)
                @test extent isa GO.UnitSpherical.SphericalCap

                # Full traversal succeeds
                @test traverse_tree(cursor) == true

                # All leaves reachable
                @test count_leaves(cursor) == nx * ny
            end
        end
    end

    @testset "show methods" begin
        grid = make_cellbased_grid(8, 8)
        cursor = TopDownQuadtreeCursor(grid)

        str = sprint(show, cursor)
        @test contains(str, "TopDownQuadtreeCursor")
        @test contains(str, "1:8")
    end
end
