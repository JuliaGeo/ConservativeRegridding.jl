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

        R = ConservativeRegridding.Regridder(GO.Planar(), dst_tree, src_tree; intersection_operator = op, normalize = false)
        Aop = R.intersections
        @test calls[] == length(dst_polys) * length(src_polys)
        @test size(Aop) == (length(dst_polys), length(src_polys))
        @test nnz(Aop) == length(dst_polys) * length(src_polys)
        @test all(nonzeros(Aop) .== 2.5)
    end

    @testset "non-positive areas are ignored" begin
        calls = Ref(0)
        op = (p1, p2) -> (calls[] += 1; -1.0)

        R = ConservativeRegridding.Regridder(GO.Planar(), dst_tree, src_tree; intersection_operator = op, normalize = false)

        Aop = R.intersections
        @test calls[] == length(dst_polys) * length(src_polys)
        @test nnz(Aop) == 0
        @test Aop == spzeros(eltype(Aop), size(Aop)...)
    end
end
