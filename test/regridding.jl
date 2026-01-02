using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
using SparseArrays

@testset "Custom intersection_operator" begin
    make_square() = GI.Polygon([GI.LinearRing([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])])

    dst = [make_square() for _ in 1:2]
    src = [make_square() for _ in 1:3]

    @testset "operator is called + writes positive areas" begin
        calls = Ref(0)
        op = (p1, p2) -> (calls[] += 1; 2.5)

        Aop = ConservativeRegridding.intersection_areas(dst, src; intersection_operator = op)

        @test calls[] == length(dst) * length(src)
        @test size(Aop) == (length(dst), length(src))
        @test nnz(Aop) == length(dst) * length(src)
        @test all(nonzeros(Aop) .== 2.5)
    end

    @testset "non-positive areas are ignored" begin
        calls = Ref(0)
        op = (p1, p2) -> (calls[] += 1; -1.0)

        Aop = ConservativeRegridding.intersection_areas(dst, src; intersection_operator = op)

        @test calls[] == length(dst) * length(src)
        @test nnz(Aop) == 0
        @test Aop == spzeros(eltype(Aop), size(Aop)...)
    end
end
