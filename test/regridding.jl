using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
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

    # If a Regridder is around, extract_source_arraylike should NOT return an
    # AbstractDimensionalSlicer for this type — it should fall through to the
    # universal pipeline driver and error (no extract method) until an extension
    # defines one.
    src_grid = ones(2, 2)  # placeholder, won't actually be used
    # We don't actually run regrid! — just verify the dispatch doesn't pick the slicer path.
    @test_throws MethodError ConservativeRegridding.extract_source_arraylike(f, nothing)
end
