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

@testset "Conservative2ndOrder construction" begin
    import GeometryOps as GO
    using SparseArrays: nnz

    # Create a 5x5 grid of points (4x4 cells) - need enough cells for gradients
    src_points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
    dst_points = [(Float64(i)*2, Float64(j)*2) for i in 0:2, j in 0:2]  # 2x2 cells, coarser

    R = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    @test R.method isa Conservative2ndOrder
    @test size(R.intersections, 1) == 4  # 2x2 dst cells
    @test size(R.intersections, 2) == 16 # 4x4 src cells

    # 2nd order matrix should be denser than 1st order (has neighbor contributions)
    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative1stOrder())
    @test nnz(R.intersections) >= nnz(R1.intersections)
end

@testset "Conservative2ndOrder transpose error" begin
    import GeometryOps as GO

    src_points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
    dst_points = [(Float64(i)*2, Float64(j)*2) for i in 0:2, j in 0:2]

    R = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    @test_throws ErrorException transpose(R)
end
