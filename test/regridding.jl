using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
import GeometryOpsCore
import Extents
using SparseArrays

@testset "Custom intersection_operator" begin
    make_square() = GI.Polygon([GI.LinearRing([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])])

    dst_polys = [make_square() for _ in 1:2]
    src_polys = [make_square() for _ in 1:3]

    dst_tree = GO.SpatialTreeInterface.FlatNoTree(dst_polys)
    src_tree = GO.SpatialTreeInterface.FlatNoTree(src_polys)

    @testset "operator is called + writes positive areas" begin
        calls = Ref(0)
        op = (p1, p2) -> (calls[] += 1; 2.5)

        R = ConservativeRegridding.Regridder(GO.Planar(), dst_tree, src_tree; intersection_operator = op, normalize = false, threaded = false)
        Aop = R.intersections
        @test calls[] == length(dst_polys) * length(src_polys)
        @test size(Aop) == (length(dst_polys), length(src_polys))
        @test nnz(Aop) == length(dst_polys) * length(src_polys)
        @test all(nonzeros(Aop) .== 2.5)
    end

    @testset "non-positive areas are ignored" begin
        calls = Ref(0)
        op = (p1, p2) -> (calls[] += 1; -1.0)

        R = ConservativeRegridding.Regridder(GO.Planar(), dst_tree, src_tree; intersection_operator = op, normalize = false, threaded = false)

        Aop = R.intersections
        @test calls[] == length(dst_polys) * length(src_polys)
        @test nnz(Aop) == 0
        @test Aop == spzeros(eltype(Aop), size(Aop)...)
    end
end

@testset "regrid! with n-dimensional arrays" begin
    function make_grid(nx, ny)
        polys = Matrix{GI.Polygon}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            x0, x1 = (i-1)/nx, i/nx
            y0, y1 = (j-1)/ny, j/ny
            ring = GI.LinearRing([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
            polys[i,j] = GI.Polygon([ring])
        end
        polys
    end

    src = make_grid(2, 2)
    dst = make_grid(3, 3)
    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst, src; threaded=false)

    @testset "Vector (existing behavior, no regression)" begin
        src_vec = ones(4)
        dst_vec = zeros(9)
        ConservativeRegridding.regrid!(dst_vec, r, src_vec)
        @test all(dst_vec .≈ 1.0)
    end

    @testset "Matrix" begin
        src_mat = ones(4, 3)
        dst_mat = zeros(9, 3)
        ConservativeRegridding.regrid!(dst_mat, r, src_mat)
        @test all(dst_mat .≈ 1.0)
    end

    @testset "3D array" begin
        src_3d = ones(4, 3, 2)
        dst_3d = zeros(9, 3, 2)
        ConservativeRegridding.regrid!(dst_3d, r, src_3d)
        @test all(dst_3d .≈ 1.0)
    end

    @testset "dims keyword" begin
        @testset "dims=1 (default)" begin
            src_mat = ones(4, 3)
            dst_mat = zeros(9, 3)
            ConservativeRegridding.regrid!(dst_mat, r, src_mat; dims=1)
            @test all(dst_mat .≈ 1.0)
        end

        @testset "dims=2 (spatial dimension last)" begin
            src_mat = ones(3, 4)
            dst_mat = zeros(3, 9)
            ConservativeRegridding.regrid!(dst_mat, r, src_mat; dims=2)
            @test all(dst_mat .≈ 1.0)
        end

        @testset "dims=2 on 3D array (spatial in the middle)" begin
            src_3d = ones(2, 4, 3)
            dst_3d = zeros(2, 9, 3)
            ConservativeRegridding.regrid!(dst_3d, r, src_3d; dims=2)
            @test all(dst_3d .≈ 1.0)
        end

        @testset "dims=3 on 3D array (spatial dimension last)" begin
            src_3d = ones(3, 2, 4)
            dst_3d = zeros(3, 2, 9)
            ConservativeRegridding.regrid!(dst_3d, r, src_3d; dims=3)
            @test all(dst_3d .≈ 1.0)
        end
    end
end

# Regression test for GitHub issue #66 + verifies the `should_parallelize` dispatch hook:
# Planar grids ship no default policy, so threaded regridding requires the tree author
# to define `should_parallelize` for their tree type. Here we override on the package's
# own TopDownQuadtreeCursor (acting as the "tree author") and verify that:
#   1. the override is actually invoked during the dual DFS, and
#   2. threaded planar regridding produces correct results.
@testset "Planar grid threaded regridding (#66)" begin
    function make_grid(nx, ny)
        polys = Matrix{GI.Polygon}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            x0, x1 = (i-1)/nx, i/nx
            y0, y1 = (j-1)/ny, j/ny
            ring = GI.LinearRing([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
            polys[i,j] = GI.Polygon([ring])
        end
        polys
    end

    # The planar default errors when called on a node type that has no
    # specific method. Use a synthetic node type so this assertion is
    # independent of any method added later in this session.
    struct _UnsupportedPlanarNode end
    @test_throws ErrorException ConservativeRegridding.Trees.should_parallelize(
        _UnsupportedPlanarNode(), Extents.Extent(X=(0.0, 1.0), Y=(0.0, 1.0)),
    )

    # Grids must be large enough that neither tree's top-level cursor is already a
    # leaf — otherwise the dual DFS short-circuits before consulting `should_parallelize`.
    src = make_grid(8, 8)
    dst = make_grid(16, 16)

    # Inject a tree-aware policy via the `WithParallelizePolicy` wrapper and
    # confirm it's called during construction. The wrapper is detected at the
    # dual-DFS call site (intersection_areas.jl) and threaded through a local
    # closure, so the policy callback gets the root tree as its first arg
    # without that being a Julia dispatch axis.
    call_count = Ref(0)
    src_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), src)
    dst_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), dst)
    src_w = ConservativeRegridding.Trees.WithParallelizePolicy(
        src_tree, (tree, node, extent) -> (call_count[] += 1; true))
    dst_w = ConservativeRegridding.Trees.WithParallelizePolicy(
        dst_tree, (tree, node, extent) -> (call_count[] += 1; true))

    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_w, src_w; threaded=true)
    @test call_count[] > 0
    @test r isa ConservativeRegridding.Regridder
    @test size(r) == (16*16, 8*8)
    A = r.intersections
    @test sum(A) > 0
end

# Verifies the instance-level WithParallelizePolicy wrapper: dispatching
# on the wrapper short-circuits the type-level default and calls the
# user-supplied closure instead.
@testset "WithParallelizePolicy wrapper" begin
    function make_grid(nx, ny)
        polys = Matrix{GI.Polygon}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            x0, x1 = (i-1)/nx, i/nx
            y0, y1 = (j-1)/ny, j/ny
            ring = GI.LinearRing([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
            polys[i,j] = GI.Polygon([ring])
        end
        polys
    end

    src_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), make_grid(8, 8))
    dst_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), make_grid(16, 16))

    policy_calls = Ref(0)
    src_wrapped = ConservativeRegridding.Trees.WithParallelizePolicy(
        src_tree, (tree, node, extent) -> (policy_calls[] += 1; true),
    )
    dst_wrapped = ConservativeRegridding.Trees.WithParallelizePolicy(
        dst_tree, (tree, node, extent) -> (policy_calls[] += 1; true),
    )

    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_wrapped, src_wrapped; threaded=true)
    @test policy_calls[] > 0
    @test r isa ConservativeRegridding.Regridder
    @test size(r) == (16*16, 8*8)
    @test sum(r.intersections) > 0
end
