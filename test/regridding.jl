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
