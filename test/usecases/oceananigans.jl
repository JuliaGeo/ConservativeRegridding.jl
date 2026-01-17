using ConservativeRegridding
using ConservativeRegridding.Trees

import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI

using Statistics
using Test

using Oceananigans


@testset "Lat-long upscaling" begin
    coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

    dst = CenterField(coarse_grid)
    src = CenterField(fine_grid)

    set!(src, (x, y, z) -> x)

    regridder = ConservativeRegridding.Regridder(dst, src)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    @test mean(interior(dst)) == mean(interior(src))

    set!(dst, (x, y, z) -> rand())

    ConservativeRegridding.regrid!(vec(interior(src)), transpose(regridder), vec(interior(dst)))

    @test mean(dst) ≈ mean(src) rtol=1e-5
end

@testset "Tripolar to lat-long" begin
    dst_grid = LatitudeLongitudeGrid(size=(360, 180, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    src_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)

    dst = CenterField(dst_grid)
    src = CenterField(src_grid)

    regridder = ConservativeRegridding.Regridder(dst, src)

    set!(src, (x, y, z) -> abs(x))
    set!(dst, (x, y, z) -> 0)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-10

    set!(src, (x, y, z) -> rand())
    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))
    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-5
end

@testset "Rectilinear (planar) upscaling" begin
    large_domain_grid = RectilinearGrid(size=(100, 100), x=(0, 2), y=(0, 2), topology=(Periodic, Periodic, Flat))
    small_domain_grid = RectilinearGrid(size=(200, 200), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))

    src = CenterField(small_domain_grid)
    dst = CenterField(large_domain_grid)

    set!(src, 1)

    regridder = ConservativeRegridding.Regridder(dst, src; threaded = false)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    # Compute the integral and make sure it is the same as the original field
    dst_int = Field(Integral(dst))
    src_int = Field(Integral(src))

    compute!(dst_int)
    compute!(src_int)

    @test only(dst_int) ≈ only(src_int)
end