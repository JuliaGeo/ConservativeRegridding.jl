using ConservativeRegridding.Trees
using Test
import ConservativeRegridding
import GeoInterface as GI, GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
import Extents
using SmallCollections: SmallVector, capacity

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

function collect_leaf_indices!(out, cursor)
    if STI.isleaf(cursor)
        for (index, _) in STI.child_indices_extents(cursor)
            push!(out, index)
        end
    else
        for i in 1:STI.nchild(cursor)
            collect_leaf_indices!(out, STI.getchild(cursor, i))
        end
    end
    return out
end

collect_leaf_indices(cursor) = collect_leaf_indices!(Int[], cursor)

function collect_leaf_shapes!(out, cursor)
    if STI.isleaf(cursor)
        push!(out, length.(cursor.leafranges))
    else
        for i in 1:STI.nchild(cursor)
            collect_leaf_shapes!(out, STI.getchild(cursor, i))
        end
    end
    return out
end

collect_leaf_shapes(cursor) = collect_leaf_shapes!(Tuple{Int, Int}[], cursor)

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
            # Spherical caps participate in the public extent protocol.
            found_pairs = Set{Tuple{Int,Int}}()
            STI.dual_depth_first_search(Extents.intersects, cursor1, cursor2) do i1, i2
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

    @testset "leaf entries are eager and fixed-capacity" begin
        grid = RegularGrid(GO.Planar(), 0.0:1.0:2.0, 0.0:1.0:2.0)
        leaf = QuadtreeCursor(grid)
        entries = @inferred STI.child_indices_extents(leaf)

        @test entries isa SmallVector{4}
        @test capacity(entries) == 4
        @test length(entries) == 4
        @test first.(entries) == collect(1:4)
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
            STI.dual_depth_first_search(Extents.intersects, cursor1, cursor2) do idx1, idx2
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
        @test cursor.leafsize == (2, 2)
    end

    @testset "Configurable leaf size" begin
        grid = make_cellbased_grid(17, 11)
        cursor = TopDownQuadtreeCursor(grid; leafsize = (4, 4))

        @test all(shape -> shape[1] <= 4 && shape[2] <= 4,
                  collect_leaf_shapes(cursor))
        @test sort(collect_leaf_indices(cursor)) == collect(1:17 * 11)

        # Once one dimension fits in the configured leaf, descent splits only
        # the other dimension instead of creating unnecessary siblings.
        strip = TopDownQuadtreeCursor(grid, (3:5, 1:11); leafsize = (4, 4))
        @test STI.nchild(strip) == 2
        @test all(child -> child.leafranges[1] == 3:5, STI.getchild(strip))

        @test_throws ArgumentError TopDownQuadtreeCursor(grid; leafsize = (0, 4))
    end

    @testset "leaf entries are eager and fixed-capacity" begin
        grid = RegularGrid(GO.Planar(), 0.0:1.0:4.0, 0.0:1.0:3.0)
        leaf = TopDownQuadtreeCursor(grid; leafsize = (4, 4))
        entries = @inferred STI.child_indices_extents(leaf)

        @test entries isa SmallVector{16}
        @test capacity(entries) == 16
        @test length(entries) == 12
        @test collect(entries) == collect(entries)

        # The leaf-size value is part of the cursor type and remains so throughout
        # descent, giving every leaf a statically known SmallVector capacity.
        root = TopDownQuadtreeCursor(
            RegularGrid(GO.Planar(), 0.0:1.0:16.0, 0.0:1.0:12.0);
            leafsize = (4, 4))
        @test typeof(STI.getchild(root, 1)) === typeof(root)
    end

    @testset "specialized cursor leaves are eager and inferred" begin
        grid = RegularGrid(GO.Planar(), 0.0:1.0:2.0, 0.0:1.0:2.0)

        offset = Trees.IndexOffsetQuadtreeCursor(grid, 10)
        offset_entries = @inferred STI.child_indices_extents(offset)
        @test offset_entries isa SmallVector{4}
        @test capacity(offset_entries) == 4
        @test first.(offset_entries) == collect(11:14)

        cart2lin = reshape(collect(21:24), 2, 2)
        lin2cart = vec(collect(CartesianIndices((2, 2))))
        reordered = Trees.ReorderedTopDownQuadtreeCursor(
            grid, Trees.Reorderer2D(cart2lin, lin2cart))
        reordered_entries = @inferred STI.child_indices_extents(reordered)
        @test isconcretetype(typeof(reordered.ordering))
        @test reordered_entries isa SmallVector{4}
        @test capacity(reordered_entries) == 4
        @test first.(reordered_entries) == collect(21:24)
    end

    @testset "Restricted cursors keep global indices" begin
        grid = make_cellbased_grid(17, 11)
        cursor = TopDownQuadtreeCursor(grid, (3:10, 4:9); leafsize = (4, 4))
        expected = sort([i + (j - 1) * 17 for i in 3:10 for j in 4:9])

        @test sort(collect_leaf_indices(cursor)) == expected
        @test ncells(cursor) == (8, 6)
        @test split_weight(cursor) == 48
        @test cell_index_count(cursor) == 17 * 11

        dst_grid = make_cellbased_grid(13, 7)
        dst = TopDownQuadtreeCursor(dst_grid, (2:6, 3:7))
        @test ConservativeRegridding.output_matrix_size(nothing, cursor, dst) ==
              (13 * 7, 17 * 11)

        intersections = ConservativeRegridding.intersection_areas(
            GO.Spherical(), GOCore.False(), cursor, cursor)
        @test size(intersections) == (17 * 11, 17 * 11)
        @test all(!iszero(intersections[i, i]) for i in expected)

        wrapped = Trees.KnownFullSphereExtentWrapper(cursor)
        @test cell_index_count(wrapped) == cell_index_count(cursor)
        geometry_wrapper = Trees.GeometryMaintainingTreeWrapper(nothing, cursor)
        @test cell_index_count(geometry_wrapper) == cell_index_count(cursor)

        offset = Trees.IndexOffsetQuadtreeCursor(grid, 200)
        @test cell_index_count(offset) == 200 + 17 * 11

        cart2lin = reshape(collect(201:(200 + 17 * 11)), 17, 11)
        lin2cart = vec(collect(CartesianIndices((17, 11))))
        reordered = Trees.ReorderedTopDownQuadtreeCursor(
            grid, Trees.Reorderer2D(cart2lin, lin2cart))
        @test cell_index_count(reordered) == 200 + 17 * 11
        @test cell_index_count(STI.getchild(reordered, 1)) == 200 + 17 * 11

        localized = Trees.IndexLocalizerRewrapperTree(reordered, 200)
        @test cell_index_count(localized) == 200 + 17 * 11
    end

    @testset "Direct child construction does not allocate sibling tuples" begin
        grid = make_cellbased_grid(64, 48)
        cursor = TopDownQuadtreeCursor(grid; leafsize = (4, 4))

        # Warm compilation before measuring the accessor itself.
        STI.getchild(cursor, 1)
        allocated = @allocated(STI.getchild(cursor, 1))
        @test allocated == 0 skip = VERSION < v"1.12"

        @test STI.getchild(cursor, 1).leafranges == (1:32, 1:24)
        @test STI.getchild(cursor, 2).leafranges == (1:32, 25:48)
        @test STI.getchild(cursor, 3).leafranges == (33:64, 1:24)
        @test STI.getchild(cursor, 4).leafranges == (33:64, 25:48)
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
