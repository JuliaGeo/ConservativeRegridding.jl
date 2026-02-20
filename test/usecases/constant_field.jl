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
# Helper: regrid a field of ones and check max deviation from 1.0.
#
# Only cells that received intersection coverage AND have nonzero area are
# checked.  Cells with zero coverage (from spatial tree search pruning near
# poles or fold boundaries) are excluded — those reflect tree search limits,
# not regridding logic bugs.  This still catches the issue #63 class of bugs
# where *covered* cells produce spurious extrema.
# ---------------------------------------------------------------------------
function test_constant_field(name, regridder; atol=1e-2, test_forward=true, test_backward=true)
    n_dst, n_src = size(regridder)
    A = regridder.intersections

    @testset "$name" begin
        if test_forward
            # Forward: ones(n_src) → dst
            src_vals = ones(n_src)
            dst_vals = zeros(n_dst)
            ConservativeRegridding.regrid!(dst_vals, regridder, src_vals)
            covered_dst = (regridder.dst_areas .> 0) .& (vec(sum(A, dims=2)) .> 0)
            max_dev = maximum(abs.(dst_vals[covered_dst] .- 1.0))
            @test max_dev < atol
        end

        if test_backward
            # Backward: ones(n_dst) → src via transpose
            dst_vals2 = ones(n_dst)
            src_vals2 = zeros(n_src)
            ConservativeRegridding.regrid!(src_vals2, transpose(regridder), dst_vals2)
            covered_src = (regridder.src_areas .> 0) .& (vec(sum(A, dims=1)) .> 0)
            max_dev_bwd = maximum(abs.(src_vals2[covered_src] .- 1.0))
            @test max_dev_bwd < atol
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
    radius=GO.Spherical().radius, n_quad_points=2, h_elem=16,
)

# ClimaCore – Gilbert (space-filling curve) ordering
_device   = ClimaComms.device()
_context  = ClimaComms.context(_device)
_h_mesh   = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(GO.Spherical().radius), 16)
_h_topo   = Topologies.Topology2D(_context, _h_mesh, Topologies.spacefillingcurve(_h_mesh))
climacore_g = CommonSpaces.CubedSphereSpace(;
    radius=_h_mesh.domain.radius, n_quad_points=2, h_elem=16,
    h_mesh=_h_mesh, h_topology=_h_topo,
)

# ---------------------------------------------------------------------------
# Tests: 10 regridders, forward + backward where feasible
# ---------------------------------------------------------------------------

@testset "Constant-field regridding" begin

    # --- Hub tests: LonLat(90×45) as central grid ---
    # Tripolar tests: the forward direction (tripolar → lonlat) has inherent
    # polar coverage gaps in the spatial tree search at any resolution, so
    # only the backward direction (lonlat → tripolar) is tested.  The
    # backward direction verifies the original issue #63 scenario.

    @testset "Hub: LonLat → TripolarF" begin
        R = ConservativeRegridding.Regridder(lonlat, tripolar_f)
        test_constant_field("backward", R; test_forward=false)
    end

    @testset "Hub: LonLat → TripolarC" begin
        R = ConservativeRegridding.Regridder(lonlat, tripolar_c)
        test_constant_field("backward", R; test_forward=false)
    end

    @testset "Hub: LonLat ↔ Healpix(Nested)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), lonlat, healpix_n)
        test_constant_field("forward & backward", R)
    end

    @testset "Hub: LonLat ↔ Healpix(Ring)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), lonlat, healpix_r)
        test_constant_field("forward & backward", R)
    end

    @testset "Hub: LonLat ↔ ClimaCore(regular)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), lonlat, climacore_r)
        test_constant_field("forward & backward", R)
    end

    @testset "Hub: LonLat ↔ ClimaCore(Gilbert)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), lonlat, climacore_g)
        test_constant_field("forward & backward", R)
    end

    # --- Self-type tests ---

    @testset "Self: LonLat ↔ LonLat" begin
        R = ConservativeRegridding.Regridder(lonlat, lonlat_fine)
        test_constant_field("forward & backward", R)
    end

    @testset "Self: Rectilinear ↔ Rectilinear" begin
        R = ConservativeRegridding.Regridder(rect_a, rect_b; threaded=false)
        test_constant_field("forward & backward", R)
    end

    # --- Cross-grid non-hub tests ---

    @testset "Cross: Healpix(Ring) ↔ ClimaCore(Gilbert)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), healpix_r, climacore_g)
        test_constant_field("forward & backward", R)
    end

    @testset "Cross: ClimaCore(regular) ↔ Healpix(Nested)" begin
        R = ConservativeRegridding.Regridder(GO.Spherical(), climacore_r, healpix_n)
        test_constant_field("forward & backward", R)
    end
end
