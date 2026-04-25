using ConservativeRegridding
using Test

using RingGrids
using SpeedyWeather  # loads the dual-dependency SpeedyWeather extension too
import GeometryOps as GO

# Smoke test: treeify succeeds, Regridder builds, dimensions match.
@testset "FullClenshawGrid: Regridder construction" begin
    src = rand(FullClenshawGrid, 12)
    dst = rand(FullClenshawGrid, 16)
    R = ConservativeRegridding.Regridder(dst, src)
    @test R isa ConservativeRegridding.Regridder
    n_dst, n_src = size(R.intersections)
    @test n_dst == length(dst)
    @test n_src == length(src)
end

# Conservation: a constant field must stay constant everywhere after regrid.
# This is the canary for partition errors (gaps / overlaps, especially at poles).
@testset "FullClenshawGrid: constant-field preservation" begin
    src = rand(FullClenshawGrid, 12)
    dst = rand(FullClenshawGrid, 18)
    R = ConservativeRegridding.Regridder(dst, src)

    src_ones = ones(Float64, length(src))
    dst_out  = zeros(Float64, length(dst))
    ConservativeRegridding.regrid!(dst_out, R, src_ones)

    @test all(isapprox.(dst_out, 1.0; atol = 1e-10))
end

# Conservation: area-weighted mean of a non-constant analytic field.
@testset "FullClenshawGrid: area-weighted mean conservation" begin
    src_grid = FullClenshawGrid(16)
    dst_grid = FullClenshawGrid(24)
    R = ConservativeRegridding.Regridder(dst_grid, src_grid)

    # Analytic field with a non-vanishing spherical mean, so that `rtol`
    # checks are well-posed (a field like sin(2λ)·cos(3φ) integrates to
    # exactly zero by orthogonality and would make the relative check degenerate).
    src_lond = RingGrids.get_lond(src_grid)
    src_latd = RingGrids.get_latd(src_grid)
    src_vals = Vector{Float64}(undef, RingGrids.get_npoints(src_grid))
    nlon_src = length(src_lond)
    for (ring, φ) in enumerate(src_latd), (i, λ) in enumerate(src_lond)
        src_vals[(ring - 1) * nlon_src + i] =
            2.0 + 0.5 * sin(2 * deg2rad(λ)) * cos(3 * deg2rad(φ))
    end

    dst_vals = zeros(Float64, RingGrids.get_npoints(dst_grid))
    ConservativeRegridding.regrid!(dst_vals, R, src_vals)

    src_total = sum(src_vals .* R.src_areas)
    dst_total = sum(dst_vals .* R.dst_areas)
    @test isapprox(src_total, dst_total; rtol = 1e-10)
end

# Abstract-dispatch smoke test: FullGaussianGrid goes through the same path.
@testset "AbstractFullGrid dispatch: FullGaussianGrid ↔ FullClenshawGrid" begin
    src = rand(FullGaussianGrid, 12)
    dst = rand(FullClenshawGrid, 16)
    R = ConservativeRegridding.Regridder(dst, src)
    @test R isa ConservativeRegridding.Regridder

    # Constant-field check on the cross-type pair too.
    src_ones = ones(Float64, length(src))
    dst_out  = zeros(Float64, length(dst))
    ConservativeRegridding.regrid!(dst_out, R, src_ones)
    @test all(isapprox.(dst_out, 1.0; atol = 1e-10))
end
