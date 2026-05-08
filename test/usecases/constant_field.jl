using ConservativeRegridding
using ConservativeRegridding.Trees

import GeometryOps as GO
import GeoInterface as GI

using Test
using Oceananigans
using Oceananigans.Grids: RightFaceFolded, RightCenterFolded
import Healpix

using ClimaCore:
    CommonSpaces, Fields, Spaces, Meshes, Topologies, Domains, ClimaComms

# ---------------------------------------------------------------------------
# Helper: regrid a field of ones and check the result.
#
# The check depends on whether the *source* grid covers the whole earth:
#   - global source:     every covered dst cell should be ≈ 1.0
#   - non-global source: undershoot is expected (uncovered regions),
#                        but overshoot (value > 1 + atol) indicates a bug
# ---------------------------------------------------------------------------
function test_constant_regrid(R::ConservativeRegridding.Regridder, src_global, dst_global; atol=1e-4)
    n_dst, n_src = size(R)
    A = R.intersections

    # Forward: src → dst
    @testset let direction = :forward, src_covers_globe = src_global
        src_vals = ones(n_src)
        dst_vals = zeros(n_dst)
        ConservativeRegridding.regrid!(dst_vals, R, src_vals)
        covered = vec(sum(A, dims=2)) .> 0
        if src_global
            max_dev = maximum(abs.(dst_vals[covered] .- 1.0); init=0.0)
            @test max_dev < atol
        else
            max_val = maximum(dst_vals[covered]; init=0.0)
            @test max_val < 1.0 + atol
        end
    end

    # Backward: dst → src via transpose
    # In the backward direction the "source" of the regrid is the dst grid of R,
    # so dst_global determines the check.
    @testset let direction = :backward, src_covers_globe = dst_global
        dst_vals = ones(n_dst)
        src_vals = zeros(n_src)
        ConservativeRegridding.regrid!(src_vals, transpose(R), dst_vals)
        covered = vec(sum(A, dims=1)) .> 0
        if dst_global
            max_dev = maximum(abs.(src_vals[covered] .- 1.0); init=0.0)
            @test max_dev < atol
        else
            max_val = maximum(src_vals[covered]; init=0.0)
            @test max_val < 1.0 + atol
        end
    end
end

# ---------------------------------------------------------------------------
# SE regridder constant-field tests (forward direction only; no transpose).
#
# src_global / dst_global are accepted for API compatibility but ignored:
# ClimaCore cubed-sphere spaces always cover the full sphere.
# ---------------------------------------------------------------------------

# SE → FV: global conservation — area-weighted mean of destination ≈ 1.0.
function test_constant_regrid(R::ConservativeRegridding.SEtoFVRegridder, src_global, dst_global; rtol=2e-2)
    N_fv, N_se_nodes = size(R)
    @testset let direction = :forward
        src_vals = ones(N_se_nodes)
        dst_vals = zeros(N_fv)
        ConservativeRegridding.regrid!(dst_vals, R, src_vals)
        @test isapprox(sum(dst_vals .* R.dst_areas) / sum(R.dst_areas), 1.0; rtol)
    end
end

# FV → SE: every covered SE node should be exactly 1.0.
function test_constant_regrid(R::ConservativeRegridding.FVtoSERegridder, src_global, dst_global; atol=1e-4)
    N_se_nodes, N_fv = size(R)
    @testset let direction = :forward
        src_vals = ones(N_fv)
        dst_vals = zeros(N_se_nodes)
        ConservativeRegridding.regrid!(dst_vals, R, src_vals)
        covered = vec(sum(R.weight_matrix, dims=2)) .> 0
        @test maximum(abs.(dst_vals[covered] .- 1.0); init=0.0) < atol
    end
end

# SE → SE: global conservation — area-weighted mean of destination elements ≈ 1.0.
function test_constant_regrid(R::ConservativeRegridding.SEtoSERegridder, src_global, dst_global; rtol=2e-2)
    N_dst_nodes, N_src_nodes = size(R)
    Nq2 = R.Nq_dst^2
    @testset let direction = :forward
        src_vals = ones(N_src_nodes)
        dst_vals = zeros(N_dst_nodes)
        ConservativeRegridding.regrid!(dst_vals, R, src_vals)
        elem_vals = [dst_vals[(e-1)*Nq2 + 1] for e in 1:length(R.dst_element_areas)]
        @test isapprox(sum(elem_vals .* R.dst_element_areas) / sum(R.dst_element_areas), 1.0; rtol)
    end
end

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

# Oceananigans spherical
lonlat      = LatitudeLongitudeGrid(size=(90, 45, 1),  longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
lonlat_fine = LatitudeLongitudeGrid(size=(180, 90, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

# Tripolar grids: Nx must be divisible by 4 so that Nhalf is even, avoiding
# a self-dual fold cell in RightCenterFolded.
tripolar_f = TripolarGrid(size=(120, 60, 1), fold_topology=RightFaceFolded)
tripolar_c = TripolarGrid(size=(120, 60, 1), fold_topology=RightCenterFolded)

# Oceananigans planar
rect_a = RectilinearGrid(size=(50, 50), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))
rect_b = RectilinearGrid(size=(30, 30), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))

# Healpix
healpix_n = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(16)
healpix_r = Healpix.HealpixMap{Float64, Healpix.RingOrder}(16)

# ClimaCore – regular ordering
climacore_r = CommonSpaces.CubedSphereSpace(;
    radius=GO.Spherical().radius, n_quad_points=4, h_elem=16,
)

# ClimaCore – Gilbert (space-filling curve) ordering
_device   = ClimaComms.device()
_context  = ClimaComms.context(_device)
_h_mesh   = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(GO.Spherical().radius), 16)
_h_topo   = Topologies.Topology2D(_context, _h_mesh, Topologies.spacefillingcurve(_h_mesh))
climacore_g = CommonSpaces.CubedSphereSpace(;
    radius=_h_mesh.domain.radius, n_quad_points=4, h_elem=16,
    h_mesh=_h_mesh, h_topology=_h_topo,
)

# ---------------------------------------------------------------------------
# Grid registry and test pairs
# ---------------------------------------------------------------------------

grids = [
    (name = "LonLat(90×45)",       grid = lonlat,       global_coverage = true),
    (name = "LonLat(180×90)",      grid = lonlat_fine,  global_coverage = true),
    (name = "TripolarF(120×60)",   grid = tripolar_f,   global_coverage = false),
    (name = "TripolarC(120×60)",   grid = tripolar_c,   global_coverage = false),
    (name = "Healpix(Nested,16)",  grid = healpix_n,    global_coverage = true),
    (name = "Healpix(Ring,16)",    grid = healpix_r,    global_coverage = true),
    (name = "ClimaCore(regular)",  grid = climacore_r,  global_coverage = true),
    (name = "ClimaCore(Gilbert)",  grid = climacore_g,  global_coverage = true),
]

# ---------------------------------------------------------------------------
# Tests: all spherical grid pairs
# ---------------------------------------------------------------------------

@testset "Constant-field regridding" begin
    for i in 1:length(grids), j in (i+1):length(grids)
        d = grids[i]
        s = grids[j]
        @testset "$(d.name) ↔ $(s.name)" begin
            R = ConservativeRegridding.Regridder(GO.Spherical(), d.grid, s.grid)
            test_constant_regrid(R, s.global_coverage, d.global_coverage)
        end
    end
end

# ---------------------------------------------------------------------------
# Planar tests (separate section)
# ---------------------------------------------------------------------------

@testset "Constant-field regridding (planar)" begin
    R = ConservativeRegridding.Regridder(rect_a, rect_b; threaded=false)
    test_constant_regrid(R, true, true)
end
