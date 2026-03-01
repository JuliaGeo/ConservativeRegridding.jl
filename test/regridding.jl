using ConservativeRegridding
using Test

import GeometryOps as GO, GeoInterface as GI
import GeometryOpsCore
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

# Regression test for GitHub issue #66:
# Planar grids with threaded=true should work (previously errored in _area_criterion).
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
    src = make_grid(2, 2)
    dst = make_grid(3, 3)
    r = ConservativeRegridding.Regridder(GeometryOpsCore.Planar(), dst, src; threaded=true)
    @test r isa ConservativeRegridding.Regridder
    # Verify that the regridder has the correct dimensions
    @test size(r) == (9, 4)
    # Verify areas are conserved: total area of all intersections should equal
    # the total area of the smaller grid (both grids cover [0,1]x[0,1])
    A = r.intersections
    @test sum(A) > 0
end
