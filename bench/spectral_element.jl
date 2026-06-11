# Construction-cost benchmark for spectral-element regridders vs the plain
# finite-volume baseline.
#
#     julia --project=docs bench/spectral_element.jl
#
# (the `docs` environment carries ClimaCore + Oceananigans + Chairmarks.)
#
# For each resolution tier it times `Regridder(dst, src)` — the dual-tree
# candidate search plus sparse intersection-matrix assembly — for three cases:
#
#     FV → FV   LatitudeLongitudeGrid          → LatitudeLongitudeGrid
#     SE → FV   cubed-sphere spectral element  → LatitudeLongitudeGrid
#     FV → SE   LatitudeLongitudeGrid          → cubed-sphere spectral element
#
# SE↔FV assembly does strictly more work than FV→FV (per intersection polygon it
# fan-triangulates and runs a Gauss rule against the SE Lagrange basis), so it is
# expected to be slower and allocate more; this script quantifies the gap.

using ConservativeRegridding
using Chairmarks
using Printf
import GeometryOps as GO
import SparseArrays
using Oceananigans: LatitudeLongitudeGrid
using ClimaCore: CommonSpaces, Meshes, Domains, Topologies, ClimaComms

const RADIUS   = GO.Spherical().radius
const N_QUAD   = 4       # GLL points per direction (Nq)
const THREADED = true    # flip to false to measure serial assembly

# cubed-sphere `h_elem` paired with a target lon×lat grid
const TIERS = [
    (; h_elem = 8,  nlon = 72,  nlat = 36),
    (; h_elem = 16, nlon = 144, nlat = 72),
    (; h_elem = 32, nlon = 288, nlat = 144),
]

latlon(nlon, nlat) = LatitudeLongitudeGrid(
    size = (nlon, nlat, 1), longitude = (0, 360), latitude = (-90, 90),
    z = (0, 1), radius = RADIUS,
)

function cubedsphere_space(h_elem)
    context    = ClimaComms.context()
    h_mesh     = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(RADIUS), h_elem)
    h_topology = Topologies.Topology2D(context, h_mesh)
    return CommonSpaces.CubedSphereSpace(;
        radius = RADIUS, n_quad_points = N_QUAD, h_elem, h_mesh, h_topology,
    )
end

# Build once (compile + correctness + nnz), then time the construction.
function time_construction(name, dst, src)
    R = ConservativeRegridding.Regridder(dst, src; threaded = THREADED)
    s = @b ConservativeRegridding.Regridder($dst, $src; threaded = $THREADED)
    return (; name, time = s.time, allocs = s.allocs, bytes = s.bytes,
            nnz = SparseArrays.nnz(R.intersections))
end

function run()
    @printf("%-5s  %-8s  %10s  %12s  %9s  %10s\n",
            "tier", "case", "time", "allocs", "memory", "nnz")
    println("-"^64)
    for t in TIERS
        fv     = latlon(t.nlon, t.nlat)                  # common FV destination
        fv_src = latlon(cld(t.nlon, 2), cld(t.nlat, 2))  # coarser FV source
        se     = cubedsphere_space(t.h_elem)
        tier   = "h$(t.h_elem)"

        for r in (time_construction("FV → FV", fv, fv_src),
                  time_construction("SE → FV", fv, se),
                  time_construction("FV → SE", se, fv))
            @printf("%-5s  %-8s  %8.1f ms  %12d  %6.1f MB  %10d\n",
                    tier, r.name, r.time * 1e3, round(Int, r.allocs),
                    r.bytes / 2^20, r.nnz)
        end
        @printf("       cubed sphere: %d elems / %d SE nodes;  FV dst: %d cells\n\n",
                6 * t.h_elem^2, 6 * t.h_elem^2 * N_QUAD^2, t.nlon * t.nlat)
    end
end

run()
