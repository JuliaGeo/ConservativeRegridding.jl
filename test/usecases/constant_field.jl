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

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

is_se_grid(g) = g isa Spaces.AbstractSpectralElementSpace

# `ones`/`zeros` factories that match the grid: ClimaCore Fields for SE spaces
# (so the Field-dispatching pipeline kicks in), flat vectors for everything else.
ones_for(grid, n)  = is_se_grid(grid) ? Fields.ones(grid)  : ones(n)
zeros_for(grid, n) = is_se_grid(grid) ? Fields.zeros(grid) : zeros(n)

# Pull a flat numeric vector out of either a Field or a plain array.
flat(x::Fields.Field) = ClimaCoreExt.se_field_to_vec(x)
flat(x::AbstractVector) = x

# ---------------------------------------------------------------------------
# Helper: regrid a field of ones and check the result.
#
# The check depends on whether the *source* grid covers the whole earth:
#   - global source:     every covered dst cell should be ≈ 1.0
#   - non-global source: undershoot is expected (uncovered regions),
#                        but overshoot (value > 1 + atol) indicates a bug
#
# For SE-flavored sides we pass a ClimaCore Field through `regrid!`; the
# Field-dispatched extract/initialize/finalize methods handle the SE conversion.
# Transpose (backward) is only exercised for FV↔FV pairs — the SE-flavored
# regridders are direction-specific by construction.
# ---------------------------------------------------------------------------
function test_constant_regrid(R, src_grid, dst_grid, src_global, dst_global; atol=1e-4)
    n_dst, n_src = size(R)
    A = R.intersections

    # Forward: src → dst
    @testset let direction = :forward, src_covers_globe = src_global
        src = ones_for(src_grid, n_src)
        dst = zeros_for(dst_grid, n_dst)
        ConservativeRegridding.regrid!(dst, R, src)
        dst_vals = flat(dst)
        covered = vec(sum(A, dims=2)) .> 0
        if src_global
            max_dev = maximum(abs.(dst_vals[covered] .- 1.0); init=0.0)
            @test max_dev < atol
        else
            max_val = maximum(dst_vals[covered]; init=0.0)
            @test max_val < 1.0 + atol
        end
    end

    # Backward via transpose — only valid for FV↔FV pairs.
    if !is_se_grid(src_grid) && !is_se_grid(dst_grid)
        @testset let direction = :backward, src_covers_globe = dst_global
            dst_vals_in = ones(n_dst)
            src_vals_out = zeros(n_src)
            ConservativeRegridding.regrid!(src_vals_out, transpose(R), dst_vals_in)
            covered = vec(sum(A, dims=1)) .> 0
            if dst_global
                max_dev = maximum(abs.(src_vals_out[covered] .- 1.0); init=0.0)
                @test max_dev < atol
            else
                max_val = maximum(src_vals_out[covered]; init=0.0)
                @test max_val < 1.0 + atol
            end
        end
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
        # Skip SE ↔ SE pairs — this PR drops SE → SE regridding.
        is_se_grid(d.grid) && is_se_grid(s.grid) && continue
        @testset "$(d.name) ↔ $(s.name)" begin
            R = ConservativeRegridding.Regridder(GO.Spherical(), d.grid, s.grid)
            test_constant_regrid(R, s.grid, d.grid, s.global_coverage, d.global_coverage)
        end
    end
end

# ---------------------------------------------------------------------------
# Planar tests (separate section)
# ---------------------------------------------------------------------------

@testset "Constant-field regridding (planar)" begin
    R = ConservativeRegridding.Regridder(rect_a, rect_b; threaded=false)
    test_constant_regrid(R, rect_b, rect_a, true, true)
end
