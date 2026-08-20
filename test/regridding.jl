using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
import GeometryOpsCore
import Extents
using SparseArrays

@testset "DefaultIntersectionOperator task-local cache" begin
    op = ConservativeRegridding.DefaultIntersectionOperator(GO.Spherical())
    task_op = ConservativeRegridding.task_local_operator(op)

    @test op.cache === nothing
    @test task_op.cache isa GO.SutherlandHodgmanCache
    @test task_op.cache !== ConservativeRegridding.task_local_operator(op).cache

    planar_op = ConservativeRegridding.DefaultIntersectionOperator(GO.Planar())
    @test ConservativeRegridding.task_local_operator(planar_op) === planar_op
end

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

@testset "IntersectionGridOperator" begin
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

    @testset "Planar" begin
        src, dst = make_grid(2, 2), make_grid(3, 3)
        op = ConservativeRegridding.IntersectionGridOperator(GO.Planar())
        r = ConservativeRegridding.Regridder(GO.Planar(), dst, src; intersection_operator = op, normalize = false, threaded = false)
        A = r.intersections

        @test eltype(A) <: GI.Polygon
        @test isconcretetype(eltype(A))
        @test size(A) == (9, 4)
        # Both grids tile the unit square, so the intersection polygons cover it exactly.
        @test sum(p -> GO.area(GO.Planar(), p), nonzeros(A)) ≈ 1.0

        # Areas of the stored polygons reproduce the default (area-weight) regridder.
        r_default = ConservativeRegridding.Regridder(GO.Planar(), dst, src; normalize = false, threaded = false)
        rows, cols, polys = findnz(A)
        @test all(
            GO.area(GO.Planar(), poly) ≈ r_default.intersections[i, j]
            for (i, j, poly) in zip(rows, cols, polys)
        )
    end

    @testset "Spherical" begin
        tf = GO.UnitSpherical.UnitSphereFromGeographic()
        corners(lons, lats) = [tf((lon, lat)) for lon in lons, lat in lats]
        src_pts = corners(range(-180, 180; length = 9), range(-90, 90; length = 5))
        dst_pts = corners(range(-180, 180; length = 7), range(-90, 90; length = 4))

        op = ConservativeRegridding.IntersectionGridOperator(GO.Spherical())
        r = ConservativeRegridding.Regridder(GO.Spherical(), dst_pts, src_pts; intersection_operator = op, normalize = false, threaded = false)
        A = r.intersections

        @test all(p -> GI.trait(p) isa Union{GI.MultiPolygonTrait, GI.PolygonTrait}, SparseArrays.nonzeros(A))
        @test isconcretetype(eltype(A))
        @test size(A) == (18, 32)
        # Both grids cover the full sphere, so the intersection polygons do too.
        total_area = sum(p -> GO.area(GO.Spherical(), p), nonzeros(A))
        @test total_area ≈ 4π * GO.Spherical().radius^2

        r_default = ConservativeRegridding.Regridder(GO.Spherical(), dst_pts, src_pts; normalize = false, threaded = false)
        rows, cols, polys = findnz(A)
        @test all(
            isapprox(GO.area(GO.Spherical(), poly), r_default.intersections[i, j]; rtol = 1e-10)
            for (i, j, poly) in zip(rows, cols, polys)
        )
    end
end

import GeometryOpsCore

@testset "regrid! dense vs strided dispatch" begin
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

    src_grid = make_grid(4, 4)
    dst_grid = make_grid(8, 8)
    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_grid, src_grid; threaded=false)

    src = collect(1.0:16.0)

    # Reference result from the all-dense path.
    reference = zeros(64)
    ConservativeRegridding.regrid!(reference, r, src)

    # `mul!` produces non-NaN output, so a NaN-fill before each call lets us detect
    # which temp buffers were written into.

    # This should never use the regridder's temp buffers, instead performing a direct `mul!`.
    @testset "Dense -> Dense" begin
        fill!(r.src_temp, NaN)
        fill!(r.dst_temp, NaN)
        dst = zeros(64)
        ConservativeRegridding.regrid!(dst, r, src)
        @test dst == reference
        @test all(isnan, r.src_temp)
        @test all(isnan, r.dst_temp)
    end

    # This should use the regridder's destination buffer, but not its source buffer.
    @testset "Dense -> Strided" begin
        fill!(r.src_temp, NaN)
        fill!(r.dst_temp, NaN)
        big_dst = zeros(128)
        dst_view = @view big_dst[1:2:end]
        @test !(dst_view isa DenseVector)
        ConservativeRegridding.regrid!(dst_view, r, src)
        @test dst_view == reference
        @test all(isnan, r.src_temp)
        @test !any(isnan, r.dst_temp)
    end

    # This should use the regridder's source buffer, but not its destination buffer.
    @testset "Strided -> Dense" begin
        fill!(r.src_temp, NaN)
        fill!(r.dst_temp, NaN)
        big_src = zeros(32)
        big_src[1:2:end] .= src
        src_view = @view big_src[1:2:end]
        @test !(src_view isa DenseVector)
        dst = zeros(64)
        ConservativeRegridding.regrid!(dst, r, src_view)
        @test dst == reference
        @test !any(isnan, r.src_temp)
        @test all(isnan, r.dst_temp)
    end

    # This should use the regridder's source and destination buffers.
    @testset "Strided -> Strided" begin
        fill!(r.src_temp, NaN)
        fill!(r.dst_temp, NaN)
        big_src = zeros(32)
        big_src[1:2:end] .= src
        src_view = @view big_src[1:2:end]
        big_dst = zeros(128)
        dst_view = @view big_dst[1:2:end]
        @test !(src_view isa DenseVector)
        @test !(dst_view isa DenseVector)
        ConservativeRegridding.regrid!(dst_view, r, src_view)
        @test dst_view == reference
        @test !any(isnan, r.src_temp)
        @test !any(isnan, r.dst_temp)
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

    @testset "dimension validation" begin
        @test_throws ArgumentError ConservativeRegridding.regrid!(zeros(9, 3), r, ones(4, 3); dims=0)
        @test_throws ArgumentError ConservativeRegridding.regrid!(zeros(9, 3), r, ones(4, 3); dims=3)

        # Non-spatial axes must match for the built-in NDSliceLoop.
        @test_throws DimensionMismatch ConservativeRegridding.regrid!(zeros(9, 3, 1), r, ones(4, 3))
        @test_throws DimensionMismatch ConservativeRegridding.regrid!(zeros(9, 4), r, ones(4, 3))
        @test_throws DimensionMismatch ConservativeRegridding.regrid!(zeros(2, 9, 3), r, ones(2, 4, 4); dims=2)
    end
end

@testset "Custom AbstractDimensionalSlicer subtype" begin
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

    src_grid = make_grid(2, 2)
    dst_grid = make_grid(3, 3)
    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_grid, src_grid; threaded=false)

    # A field type whose data lives in a Matrix but is conceptually 1-D (single slice = vec).
    struct FlatMatrixField{T} <: AbstractArray{T,2}
        data::Matrix{T}
    end
    Base.size(f::FlatMatrixField) = size(f.data)
    Base.getindex(f::FlatMatrixField, I...) = getindex(f.data, I...)
    Base.setindex!(f::FlatMatrixField, v, I...) = setindex!(f.data, v, I...)

    # The custom slicer yields a single 1-D view: vec(matrix).
    struct FlatMatrixSlicer{T} <: ConservativeRegridding.AbstractDimensionalSlicer
        array::Matrix{T}
    end
    Base.parent(s::FlatMatrixSlicer) = s.array
    ConservativeRegridding.slice_views(s::FlatMatrixSlicer) = (vec(parent(s)),)

    # Wire the field into the pipeline.
    ConservativeRegridding.extract_source_arraylike(src::FlatMatrixField, r; kwargs...) =
        FlatMatrixSlicer(src.data)
    ConservativeRegridding.extract_dest_arraylike(dst::FlatMatrixField, r; kwargs...) =
        FlatMatrixSlicer(dst.data)

    src_field = FlatMatrixField(ones(2, 2))   # 4 cells
    dst_field = FlatMatrixField(zeros(3, 3))  # 9 cells
    ConservativeRegridding.regrid!(dst_field, r, src_field)
    @test all(dst_field.data .≈ 1.0)
end

@testset "Non-strided AbstractArray does not hit NDSliceLoop dispatch" begin
    # Simulates an Oceananigans.Field / ClimaCore.Fields.Field style wrapper:
    # subtypes AbstractArray but is NOT StridedArray.
    struct NotStridedField{T,N} <: AbstractArray{T,N}
        data::Array{T,N}
    end
    Base.size(f::NotStridedField) = size(f.data)
    Base.getindex(f::NotStridedField, I...) = getindex(f.data, I...)

    f = NotStridedField(ones(3, 4))
    @test !(f isa StridedArray)
    @test !(typeof(f) <: StridedArray)

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
    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), make_grid(3, 4), make_grid(2, 2); threaded=false)

    # 1. Today's behavior: no method matches a non-strided AbstractArray. Locks in the status quo;
    #    `nothing` regridder distinguishes "no such method exists at all" from "errored downstream".
    @test_throws MethodError ConservativeRegridding.extract_source_arraylike(f, nothing)
    @test_throws MethodError ConservativeRegridding.extract_source_arraylike(f, r)

    # 2. Forward-looking guard: if a future maintainer adds a fallback
    #    `extract_source_arraylike(::AbstractArray, ::Any)` that erroneously routes through the
    #    StridedArray N-D path (i.e. returns an `AbstractDimensionalSlicer`), this test must fail.
    #    Avoids `hasmethod`/`which`, which can surprise around `where` clauses and kwargs — we
    #    just assert the observable return value isn't a slicer. `try`/`catch` swallows the
    #    MethodError today; the `!isa(::AbstractDimensionalSlicer)` check fires the moment a
    #    misguided fallback gets added and starts returning successfully.
    result = try
        ConservativeRegridding.extract_source_arraylike(f, r)
    catch
        nothing
    end
    @test !(result isa ConservativeRegridding.AbstractDimensionalSlicer)
end

# Regression test for GitHub issue #66: planar grids used to ship no default
# parallelize policy, so threaded planar regridding errored. The budget-frontier
# traversal has no policy hook at all, so planar trees now thread out of the box.
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

    src = make_grid(8, 8)
    dst = make_grid(16, 16)
    src_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), src)
    dst_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), dst)

    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_tree, src_tree; threaded=true)
    @test r isa ConservativeRegridding.Regridder
    @test size(r) == (16*16, 8*8)
    @test sum(r.intersections) > 0

    serial = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_tree, src_tree; threaded=false)
    @test r.intersections == serial.intersections
end

# `WithParallelizePolicy` is deprecated - its policy is no longer consulted - but a
# wrapped tree must still traverse (through the `AbstractTreeWrapper` forwarding) and
# regrid to exactly what the unwrapped tree gives.
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

    src_wrapped = ConservativeRegridding.Trees.WithParallelizePolicy(
        src_tree, (tree, node, extent) -> true,
    )
    dst_wrapped = ConservativeRegridding.Trees.WithParallelizePolicy(
        dst_tree, (tree, node, extent) -> true,
    )

    # The wrapper forwards `split_weight` to the tree it wraps, so the frontier sizes
    # tasks identically either way.
    @test ConservativeRegridding.Trees.split_weight(src_wrapped) ==
        ConservativeRegridding.Trees.split_weight(src_tree)

    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_wrapped, src_wrapped; threaded=true)
    @test r isa ConservativeRegridding.Regridder
    @test size(r) == (16*16, 8*8)
    @test sum(r.intersections) > 0

    plain = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst_tree, src_tree; threaded=true)
    @test r.intersections == plain.intersections
end

# Forwards a real tree but reports one fixed `split_weight` everywhere, the way a tree
# whose `Trees.ncells` answers for its whole grid does at every node.
struct LyingWeightTree{T}
    tree::T
    total::Int
end
GO.SpatialTreeInterface.isspatialtree(::Type{<:LyingWeightTree}) = true
GO.SpatialTreeInterface.isleaf(w::LyingWeightTree) = GO.SpatialTreeInterface.isleaf(w.tree)
GO.SpatialTreeInterface.node_extent(w::LyingWeightTree) = GO.SpatialTreeInterface.node_extent(w.tree)
GO.SpatialTreeInterface.nchild(w::LyingWeightTree) = GO.SpatialTreeInterface.nchild(w.tree)
GO.SpatialTreeInterface.getchild(w::LyingWeightTree) =
    (LyingWeightTree(c, w.total) for c in GO.SpatialTreeInterface.getchild(w.tree))
GO.SpatialTreeInterface.getchild(w::LyingWeightTree, i::Int) =
    LyingWeightTree(GO.SpatialTreeInterface.getchild(w.tree, i), w.total)
GO.SpatialTreeInterface.child_indices_extents(w::LyingWeightTree) =
    GO.SpatialTreeInterface.child_indices_extents(w.tree)
ConservativeRegridding.Trees.split_weight(w::LyingWeightTree) = w.total

# Two trees that share no intersecting cell pair must yield an all-zero matrix, not an
# error. The threaded path used to hit `reduce` over an empty collection at two places:
# the merge of the per-task results (MultithreadedDualDepthFirstSearch), and the COO
# assembly handed zero partitions (assemble_sparse_matrix_coo). Both must agree with the
# sequential path, which has always handled this correctly. The frontier reaches the
# first at any task budget, so it is swept here.
@testset "Threaded intersection with no intersecting pairs" begin
    function make_grid(nx, ny; x0 = 0.0, y0 = 0.0)
        polys = Matrix{GI.Polygon}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            a, b = x0 + (i-1)/nx, x0 + i/nx
            c, d = y0 + (j-1)/ny, y0 + j/ny
            polys[i,j] = GI.Polygon([GI.LinearRing([(a,c),(b,c),(b,d),(a,d),(a,c)])])
        end
        polys
    end

    # Grids must be big enough that neither root cursor is a leaf, or the dual DFS
    # short-circuits into a task before it can come up empty.
    src_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), make_grid(8, 8))
    dst_tree = ConservativeRegridding.Trees.treeify(GeometryOpsCore.Planar(), make_grid(16, 16; x0 = 10.0, y0 = 10.0))

    expected = ConservativeRegridding.intersection_areas(
        GeometryOpsCore.Planar(), GeometryOpsCore.False(), dst_tree, src_tree)
    @test nnz(expected) == 0

    for tree_pair in ((src_tree, dst_tree),
                      (ConservativeRegridding.Trees.WithParallelizePolicy(src_tree, (t, n, e) -> true),
                       ConservativeRegridding.Trees.WithParallelizePolicy(dst_tree, (t, n, e) -> true)))
        A = ConservativeRegridding.intersection_areas(
            GeometryOpsCore.Planar(), GeometryOpsCore.True(), tree_pair[2], tree_pair[1])
        @test A == expected
        @test size(A) == size(expected)
        @test eltype(A) == eltype(expected)
    end
end

# The threaded candidate search must be an exact stand-in for the serial one: same
# pairs, same order, at any task budget. Two asymmetric grid pairs (a spherical one,
# which weighs pairs by cap overlap, and a planar one, which falls back to the product
# of subtree sizes) are checked against `dual_depth_first_search`, and the assembled
# weight matrices are checked threaded against unthreaded.
@testset "Threaded dual query matches the serial traversal" begin
    MDDFS = ConservativeRegridding.MultithreadedDualDepthFirstSearch

    function serial_pairs(predicate, t1, t2)
        pairs = Tuple{Int, Int}[]
        GO.SpatialTreeInterface.dual_depth_first_search(predicate, t1, t2) do i1, i2
            push!(pairs, (i1, i2))
        end
        return pairs
    end

    sphere_points(nx, ny) =
        [GO.UnitSpherical.UnitSphereFromGeographic()((lon, lat))
         for lon in range(-180, 180, length = nx + 1), lat in range(-80, 80, length = ny + 1)]

    cases = (
        (name = "spherical",
         manifold = GO.Spherical(),
         predicate = GO.UnitSpherical._intersects,
         src = ConservativeRegridding.Trees.treeify(GO.Spherical(), sphere_points(60, 40)),
         dst = ConservativeRegridding.Trees.treeify(GO.Spherical(), sphere_points(48, 32))),
        (name = "planar",
         manifold = GO.Planar(),
         predicate = Extents.intersects,
         src = ConservativeRegridding.Trees.treeify(
             GO.Planar(), (collect(range(0.0, 1.0, length = 61)), collect(range(0.0, 1.0, length = 41)))),
         dst = ConservativeRegridding.Trees.treeify(
             GO.Planar(), (collect(range(0.1, 1.1, length = 49)), collect(range(0.0, 1.0, length = 33))))),
    )

    for case in cases
        @testset "$(case.name)" begin
            expected = serial_pairs(case.predicate, case.src, case.dst)
            @test length(expected) > 1000

            @test MDDFS.multithreaded_dual_query(case.predicate, case.src, case.dst) == expected
            for chunks_per_thread in (1, 8, 64)
                @test MDDFS.multithreaded_dual_query(
                    case.predicate, case.src, case.dst; chunks_per_thread) == expected
            end
            # The old five-argument shape still works; its closures are ignored.
            @test MDDFS.multithreaded_dual_query(
                case.predicate, (n, e) -> false, (n, e) -> false, case.src, case.dst) == expected

            threaded = ConservativeRegridding.intersection_areas(
                case.manifold, GeometryOpsCore.True(), case.dst, case.src)
            serial = ConservativeRegridding.intersection_areas(
                case.manifold, GeometryOpsCore.False(), case.dst, case.src)
            @test threaded == serial
            @test nnz(threaded) > 1000
        end
    end
end

# A `split_weight` that answers the whole grid at every node makes `pair_weight`
# cancel to a constant: splits then inflated the frontier's estimated total until the
# share test stalled, leaving single pairs holding up to half the traversal (measured
# 50% on a raster-to-DGG regrid). The frontier now conserves its estimate across
# splits, so even a saturated weight yields a balanced frontier. Counts are exact:
# the frontier and the per-pair searches are serial and deterministic.
@testset "Frontier stays balanced when split_weight saturates" begin
    MDDFS = ConservativeRegridding.MultithreadedDualDepthFirstSearch

    sphere_points(nx, ny) =
        [GO.UnitSpherical.UnitSphereFromGeographic()((lon, lat))
         for lon in range(-180, 180, length = nx + 1), lat in range(-80, 80, length = ny + 1)]

    for sizes in (((120, 80), (12, 8)), ((120, 80), (120, 80)))
        src = ConservativeRegridding.Trees.treeify(GO.Spherical(), sphere_points(sizes[1]...))
        dst = ConservativeRegridding.Trees.treeify(GO.Spherical(), sphere_points(sizes[2]...))
        w1 = LyingWeightTree(src, prod(sizes[1]))
        w2 = LyingWeightTree(dst, prod(sizes[2]))
        pred = GO.UnitSpherical._intersects

        pairs = MDDFS.frontier(pred, w1, w2; nchunks = 64)
        chunks = [MDDFS._inner_dfs_f(pred, p[1], p[3]) for p in pairs]

        # Weights only order the splits: concatenation still matches the serial search.
        @test reduce(vcat, chunks) == MDDFS._inner_dfs_f(pred, src, dst)

        # No frontier pair keeps more than a tenth of the candidate pairs (the
        # broken estimator left 23-25% on both grid pairs; the fix leaves 5-6%).
        counts = map(length, chunks)
        @test maximum(counts) <= sum(counts) ÷ 10
    end
end

# The threaded assembly builds the CSC out of column windows, straight off the per-task
# COO triplets, instead of concatenating them and calling `sparse` once. The two must
# agree bit for bit, duplicates included: `sparse` folds repeated coordinates in input
# order, so a window that keeps a column's entries in that order rounds identically.
using Random

@testset "Windowed COO assembly is bit-identical to `sparse`" begin
    CR = ConservativeRegridding

    function fixture(rng, nrows, ncols, nchunks, nper, ndup)
        map(1:nchunks) do _
            rows = rand(rng, 1:nrows, nper)
            cols = rand(rng, 1:ncols, nper)
            vals = randn(rng, nper)
            # Repeat the head of the chunk so whole columns carry duplicate coordinates,
            # both within a chunk and across chunks.
            (vcat(rows, rows[1:ndup]), vcat(cols, cols[1:ndup]), vcat(vals, randn(rng, ndup)))
        end
    end

    rng = MersenneTwister(20260820)
    shapes = ((200, 5000, 32, 4000), (50, 97, 16, 8000), (1, 3, 8, 30000),
              (5000, 200, 32, 6000), (300, 4097, 7, 20000), (300, 100_000, 32, 5000))
    for (nrows, ncols, nchunks, nper) in shapes
        chunks = fixture(rng, nrows, ncols, nchunks, nper, nper ÷ 3)
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, nrows, ncols)
        @test nnz(expected) < length(rows)   # the fixture really does carry duplicates
        # Sweep the window count explicitly: the default depends on `Threads.nthreads()`.
        for nwindows in (2, 3, 8)
            got = CR._sparse_from_chunks(chunks, nrows, ncols, nwindows)
            @test got.colptr == expected.colptr
            @test got.rowval == expected.rowval
            @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
        end
    end

    # A skewed column distribution puts the window cuts inside a single hot block.
    let nrows = 200, ncols = 20_000, nchunks = 32, nper = 5000
        chunks = map(1:nchunks) do _
            cols = [rand(rng) < 0.9 ? rand(rng, 1:200) : rand(rng, 1:ncols) for _ in 1:nper]
            (rand(rng, 1:nrows, nper), cols, randn(rng, nper))
        end
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, nrows, ncols)
        for nwindows in (2, 8)
            got = CR._sparse_from_chunks(chunks, nrows, ncols, nwindows)
            @test got.colptr == expected.colptr
            @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
        end
    end

    # Chunks that produced nothing are a normal outcome of the frontier's split.
    let nrows = 100, ncols = 3000, nper = 20_000
        chunks = map(1:16) do c
            n = iseven(c) ? 0 : nper
            (rand(rng, 1:nrows, n), rand(rng, 1:ncols, n), randn(rng, n))
        end
        rows, cols, vals = CR._concat_chunks(chunks)
        @test CR._sparse_from_chunks(chunks, nrows, ncols, 4) == sparse(rows, cols, vals, nrows, ncols)
    end
end

# The window bounds are a deterministic cut of a coarse column histogram: strictly
# increasing (no empty window), block-aligned, and balanced on entry count.
@testset "Column windows are balanced and non-empty" begin
    CR = ConservativeRegridding

    for blockcount in (fill(100, 64), [1000; fill(1, 63)], [fill(0, 60); fill(500, 4)],
                       collect(1:64), reverse(collect(1:64)))
        for nwindows in (2, 3, 8, 64)
            bnd = CR._window_bounds(blockcount, nwindows)
            @test length(bnd) == nwindows + 1
            @test bnd[1] == 1
            @test bnd[end] == length(blockcount) + 1
            @test all(diff(bnd) .>= 1)
        end
    end

    # An even histogram must split evenly, exactly.
    @test CR._window_bounds(fill(100, 64), 4) == [1, 17, 33, 49, 65]

    # One heavy block cannot be split, so it takes a window of its own and the rest
    # spread over what is left.
    @test CR._window_bounds([1000; fill(1, 63)], 4) == [1, 2, 3, 4, 65]
end

# Cases the random fixtures above do not reach: a fold whose order decides the result,
# signed zeros and subnormals, `Bool`'s `|`, ragged chunks, and the default window count.
@testset "Windowed COO assembly: directed cases" begin
    CR = ConservativeRegridding

    # `1e16 + 1.0 - 1e16` is `0.0` in input order and `1.0` in either other order, and here
    # the three duplicates of one coordinate sit in three different chunks.
    cancel = [([1], [7], [1e16]), ([1], [7], [1.0]), ([1], [7], [-1e16])]
    for nwindows in (2, 3, 8)
        @test CR._sparse_from_chunks(cancel, 4, 20_000, nwindows).nzval == [0.0]
    end

    # `-0.0 + 0.0` is `+0.0`, and two subnormals add without rounding.
    signs = [([1, 2], [3, 5], [-0.0, 5.0e-324]), ([1, 2], [3, 5], [0.0, 5.0e-324])]
    bools = [([1, 2], [3, 5], [true, false]), ([1, 2], [3, 5], [false, false])]
    for chunks in (cancel, signs, bools)
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, 4, 20_000)
        for nwindows in (2, 3, 8)
            got = CR._sparse_from_chunks(chunks, 4, 20_000, nwindows)
            @test got.colptr == expected.colptr
            @test got.rowval == expected.rowval
            @test reinterpret(UInt8, got.nzval) == reinterpret(UInt8, expected.nzval)
        end
    end

    # A chunk whose three vectors disagree in length is an error, as it is for `sparse`,
    # not a silent truncation or an unchecked read.
    @test_throws ArgumentError CR._sparse_from_chunks([([1, 1], [1], [2.0, 3.0])], 1, 20_000, 2)

    # The default window count depends on `Threads.nthreads()`; the result must not.
    let nrows = 64, ncols = 30_000, rng = MersenneTwister(20260821)
        chunks = [(rand(rng, 1:nrows, 45_000), rand(rng, 1:ncols, 45_000), randn(rng, 45_000)) for _ in 1:3]
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, nrows, ncols)
        got = CR._sparse_from_chunks(chunks, nrows, ncols)
        @test got.colptr == expected.colptr
        @test got.rowval == expected.rowval
        @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
    end
end
