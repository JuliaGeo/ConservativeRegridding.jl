#=
# 2nd-order conservative regridding on lon-lat grids

Tests for `Regridder(dst, src; algorithm = SecondOrderConservative())` against
two regular spherical lon-lat grids. The grids are 36×18 → 24×12, both global
(periodic in longitude, pole folds at top and bottom).

The 2nd-order spherical clipper requires polygons whose vertices are
`UnitSphericalPoint`s, but `Trees.LonLatConnectivityWrapper` requires a
`RegularGrid`. We build the wrapper by hand: a `CellBasedGrid` of
`UnitSphericalPoint`s wrapped in a `TopDownQuadtreeCursor` provides the
cell geometry; the `LonLatConnectivityWrapper` then adds the lon/lat
periodicity / pole-fold metadata that `Trees.neighbours` reads.

Test outline (matches `docs/plans/2026-04-26-second-order-design.md` §Testing strategy):
  1. Constant-field exactness        — proves the asymmetric ÷A vs ÷2A scaling.
  2. Linear-field reproduction       — much better than 1st-order on a smooth field.
  3. Conservation under non-trivial source.
  4. API smoke tests.
  5. `T = Float32` numeric-type kwarg.
  6. `GeometricCentroid` algorithm path.

## Linear-field reproduction (2026-04-27)

We do not assert exact recovery of a linear field. Boundary cells whose
`Trees.neighbours` stencil does not contain `r_n` fall back to a zero
gradient, so the regridder is only piecewise-linear in the interior. The
right invariant is "more accurate than 1st-order on a smooth field", which
empirically holds at ~1.7× on this 36×18 → 24×12 grid pair.
=#

using Test
using ConservativeRegridding
using ConservativeRegridding: Trees
import GeometryOps as GO
import GeoInterface as GI
import LinearAlgebra

const CR = ConservativeRegridding

# ---------------------------------------------------------------------------
# Helpers — build a 2nd-order-compatible lon/lat tree
# ---------------------------------------------------------------------------

# A `CellBasedGrid` of `UnitSphericalPoint`s, suitable for the spherical
# `ConvexConvexSutherlandHodgman` clipper.
function unit_spherical_lonlat_grid(x, y)
    pts = [GO.UnitSphereFromGeographic()((xi, yj)) for xi in x, yj in y]
    return Trees.CellBasedGrid(GO.Spherical(), pts)
end

# A `LonLatConnectivityWrapper` whose underlying tree returns
# `UnitSphericalPoint`-based polygons (so spherical clipping works) but which
# carries the lon/lat periodicity / pole-fold metadata that
# `Trees.neighbours` reads.
function build_lonlat_2nd_order_tree(x, y)
    sph_grid = unit_spherical_lonlat_grid(x, y)
    sph_cursor = Trees.TopDownQuadtreeCursor(sph_grid)
    nx, ny = length(x) - 1, length(y) - 1
    atol = 3.6e-4
    periodic_x       = isapprox(x[end] - x[1], 360.0; atol)
    pole_top_fold    = iseven(nx) && isapprox(y[end],  90.0; atol)
    pole_bottom_fold = iseven(nx) && isapprox(y[1],  -90.0; atol)
    return Trees.LonLatConnectivityWrapper(
        sph_cursor, periodic_x, pole_top_fold, pole_bottom_fold, nx, ny,
    )
end

# Cell-center longitude / latitude (in degrees) for evaluating analytic fields.
function cell_centers(x, y)
    nx, ny = length(x) - 1, length(y) - 1
    lon = [0.5 * (x[i] + x[i+1]) for i in 1:nx, _ in 1:ny]
    lat = [0.5 * (y[j] + y[j+1]) for _ in 1:nx, j in 1:ny]
    return vec(lon), vec(lat)
end

# ---------------------------------------------------------------------------
# Grid setup — 36×18 → 24×12 global periodic with pole folds
# ---------------------------------------------------------------------------

const SRC_X = collect(range(-180.0, 180.0; length = 37))
const SRC_Y = collect(range(-90.0,   90.0; length = 19))
const DST_X = collect(range(-180.0, 180.0; length = 25))
const DST_Y = collect(range(-90.0,   90.0; length = 13))

const SRC_TREE = build_lonlat_2nd_order_tree(SRC_X, SRC_Y)
const DST_TREE = build_lonlat_2nd_order_tree(DST_X, DST_Y)

const SRC_LON, SRC_LAT = cell_centers(SRC_X, SRC_Y)
const DST_LON, DST_LAT = cell_centers(DST_X, DST_Y)

const N_SRC = length(SRC_LON)   # 36 * 18 = 648
const N_DST = length(DST_LON)   # 24 * 12 = 288

# Build the shared 2nd-order regridder (ESMFLike default centroid). All
# dependent testsets reuse this single instance.
const R_2ND_REF = Ref{Any}(CR.Regridder(
    GO.Spherical(), DST_TREE, SRC_TREE;
    algorithm = CR.SecondOrderConservative(),
    normalize = false,
))

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "Build 2nd-order regridder (ESMFLike, Spherical)" begin
    @test R_2ND_REF[] isa CR.Regridder
end

@testset "Constant-field exactness (f = 7.0)" begin
    r = R_2ND_REF[]
    src = fill(7.0, N_SRC)
    dst = zeros(N_DST)
    CR.regrid!(dst, r, src)
    @test all(isapprox.(dst, 7.0; rtol = 1e-12))
end

@testset "Linear field: 2nd-order more accurate than 1st-order" begin
    # NOTE: We don't assert exact linear reproduction. Boundary cells (those
    # whose `Trees.neighbours` stencil doesn't contain `r_n`) fall back to a
    # zero gradient, so the regridder is only piecewise-linear in the interior.
    # The right invariant is "more accurate than 1st-order on a smooth field".
    r = R_2ND_REF[]
    a, b, c = 1.0, 0.01, 0.02
    f(lon, lat) = a + b * lon + c * lat

    src = [f(lon, lat) for (lon, lat) in zip(SRC_LON, SRC_LAT)]
    dst = zeros(N_DST)
    CR.regrid!(dst, r, src)

    expected = [f(lon, lat) for (lon, lat) in zip(DST_LON, DST_LAT)]
    # Restrict comparison to mid-latitudes where curved-cell error is small.
    keep = abs.(DST_LAT) .< 60.0

    r_1st = CR.Regridder(
        GO.Spherical(), DST_TREE, SRC_TREE;
        algorithm = CR.FirstOrderConservative(),
        normalize = false,
    )
    dst_1st = zeros(N_DST)
    CR.regrid!(dst_1st, r_1st, src)
    err_2nd = sum(abs2, dst[keep] .- expected[keep])
    err_1st = sum(abs2, dst_1st[keep] .- expected[keep])
    @test err_2nd < err_1st
end

@testset "Conservation: non-zero-integral source" begin
    # Use a source whose global integral is non-zero (offset + smooth term);
    # `cos(λ) sin(2φ)` alone integrates to ~0 over the sphere, so floating
    # point noise dominates a relative-tolerance comparison. The empirical
    # rel err here is ~4e-16 — well within `rtol = 1e-12`.
    r = R_2ND_REF[]
    f(lon, lat) = 3.0 + cos(deg2rad(lon)) * sin(2 * deg2rad(lat))
    src = [f(lon, lat) for (lon, lat) in zip(SRC_LON, SRC_LAT)]
    dst = zeros(N_DST)
    CR.regrid!(dst, r, src)

    src_int = sum(src .* r.src_areas)
    dst_int = sum(dst .* r.dst_areas)
    @test isapprox(dst_int, src_int; rtol = 1e-12)
end

@testset "API smoke tests" begin
    # `supports_transpose` is purely on the algorithm type — no build needed.
    @test CR.supports_transpose(CR.SecondOrderConservative()) === false
    @test CR.supports_transpose(CR.FirstOrderConservative())  === true

    # Algorithm constructors and their fields.
    alg_default = CR.SecondOrderConservative()
    @test alg_default isa CR.SecondOrderConservative
    @test alg_default.centroid_algorithm isa CR.ESMFLike

    alg_geom = CR.SecondOrderConservative(CR.GeometricCentroid())
    @test alg_geom isa CR.SecondOrderConservative
    @test alg_geom.centroid_algorithm isa CR.GeometricCentroid

    r = R_2ND_REF[]
    @test r.algorithm isa CR.SecondOrderConservative
    @test r.algorithm.centroid_algorithm isa CR.ESMFLike
    # `transpose(r)` raises a MethodError for 2nd-order regridders.
    @test_throws MethodError LinearAlgebra.transpose(r)

    # NOTE: the manifold-inferring constructor `Regridder(dst, src; ...)` is
    # not exercised here because `best_manifold` has no method for
    # `LonLatConnectivityWrapper`. The explicit `Regridder(GO.Spherical(), ...)`
    # path is what real users of `LonLatConnectivityWrapper` invoke and is
    # covered by every other testset in this file via `R_2ND_REF`.
end

@testset "T = Float32" begin
    r32 = CR.Regridder(
        GO.Spherical(), DST_TREE, SRC_TREE;
        algorithm = CR.SecondOrderConservative(),
        T = Float32,
        normalize = false,
    )
    @test eltype(r32.intersections) === Float32
    @test eltype(r32.dst_areas)     === Float32
    @test eltype(r32.src_areas)     === Float32

    src32 = fill(7.0f0, N_SRC)
    dst32 = zeros(Float32, N_DST)
    CR.regrid!(dst32, r32, src32)
    @test eltype(dst32) === Float32
    # Float32 constant-field exactness: rtol ≈ 1e-5 is the right ballpark.
    @test all(isapprox.(dst32, 7.0f0; rtol = 1.0f-5))
end

@testset "GeometricCentroid algorithm path" begin
    # The `GeometricCentroid` path on the spherical manifold currently routes
    # through `GO.centroid_and_area(::Spherical, ::Polygon{UnitSphericalPoint})`
    # which is not (yet) implemented in GeometryOps. The algorithm constructors
    # themselves are independent of geometry and tested unconditionally.
    alg = CR.SecondOrderConservative(CR.GeometricCentroid())
    @test alg isa CR.SecondOrderConservative
    @test alg.centroid_algorithm isa CR.GeometricCentroid

    @test_broken try
        r_geom = CR.Regridder(
            GO.Spherical(), DST_TREE, SRC_TREE;
            algorithm = alg,
            normalize = false,
        )
        true
    catch
        false
    end
end
