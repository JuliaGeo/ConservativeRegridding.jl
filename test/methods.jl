using ConservativeRegridding
using Test

@testset "Method types" begin
    @test Conservative1stOrder() isa ConservativeRegridding.AbstractRegridMethod
    @test Conservative2ndOrder() isa ConservativeRegridding.AbstractRegridMethod
end

@testset "Regridder stores method" begin
    # Create simple test grids
    import GeometryOps as GO, GeoInterface as GI
    make_square(x, y) = GI.Polygon([GI.LinearRing([
        (x, y), (x+1.0, y), (x+1.0, y+1.0), (x, y+1.0), (x, y)
    ])])

    dst_polys = [make_square(0.0, 0.0), make_square(1.0, 0.0)]
    src_polys = [make_square(0.0, 0.0), make_square(0.5, 0.0), make_square(1.0, 0.0)]

    # Wrap in FlatNoTree for compatibility with Regridder
    dst = GO.SpatialTreeInterface.FlatNoTree(dst_polys)
    src = GO.SpatialTreeInterface.FlatNoTree(src_polys)

    R = ConservativeRegridding.Regridder(GO.Planar(), dst, src)
    @test R.method isa Conservative1stOrder
end

@testset "Method keyword argument" begin
    import GeometryOps as GO, GeoInterface as GI
    make_square(x, y) = GI.Polygon([GI.LinearRing([
        (x, y), (x+1.0, y), (x+1.0, y+1.0), (x, y+1.0), (x, y)
    ])])

    dst_polys = [make_square(0.0, 0.0), make_square(1.0, 0.0)]
    src_polys = [make_square(0.0, 0.0), make_square(0.5, 0.0), make_square(1.0, 0.0)]

    # Wrap in FlatNoTree for compatibility with Regridder
    dst = GO.SpatialTreeInterface.FlatNoTree(dst_polys)
    src = GO.SpatialTreeInterface.FlatNoTree(src_polys)

    # Default should be Conservative1stOrder
    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst, src)
    @test R1.method isa Conservative1stOrder

    # Explicit Conservative1stOrder
    R2 = ConservativeRegridding.Regridder(GO.Planar(), dst, src; method=Conservative1stOrder())
    @test R2.method isa Conservative1stOrder

    # Results should be identical
    @test R1.intersections == R2.intersections
end
