# Grid-pair factories for the benchmark suite, parametrized by a resolution tier.
#
# Single source of truth shared by scaling.jl, xesmf.jl, run.jl and compare.jl. Each factory
# returns a NamedTuple describing one (dst, src) regridding workload within ONE grid family:
# a coarse source regridded onto a finer destination (src at half the destination resolution).
# Sweeping the tier isolates SIZE / scaling, which is the focus here — unlike the sweat tests,
# which instead enumerate grid *natures* (tripolar, rotated, gilbert, …).
#
# Idempotent: safe to `include` more than once (run.jl pulls in both scaling.jl and xesmf.jl,
# each of which includes this file).

if !@isdefined(_BENCH_GRIDS_LOADED)
const _BENCH_GRIDS_LOADED = true

import GeometryOps as GO
using Oceananigans: LatitudeLongitudeGrid
import Healpix

const SPHERE_RADIUS = GO.Spherical().radius

# Resolution tiers. Oceananigans is indexed by Ny (Nx = 2Ny); Healpix by nside (a power of 2).
# ncells_dst = 2Ny²  ∈ {512, 2k, 8k, 32k, 131k};  12 nside² ∈ {768, 3k, 12k, 49k, 196k}.
const OCEANANIGANS_NY = [16, 32, 64, 128, 256]
const HEALPIX_NSIDE   = [8, 16, 32, 64, 128]

latlongrid(Nx, Ny) = LatitudeLongitudeGrid(; size = (Nx, Ny, 1),
    longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = SPHERE_RADIUS)

healpixmap(nside) = Healpix.HealpixMap{Float64, Healpix.RingOrder}(nside)

# Oceananigans LatLon workload at tier Ny: dst = (2Ny, Ny), src = (Ny, Ny÷2).
function oceananigans_pair(Ny::Integer)
    dst = latlongrid(2Ny, Ny)
    src = latlongrid(Ny, Ny ÷ 2)
    return (; family = "Oceananigans", tier = Int(Ny),
        ncells_dst = 2Ny^2, ncells_src = Ny * (Ny ÷ 2), dst, src)
end

# Healpix workload at tier nside: dst = nside, src = nside÷2 (still a power of two).
function healpix_pair(nside::Integer)
    dst = healpixmap(nside)
    src = healpixmap(nside ÷ 2)
    return (; family = "Healpix", tier = Int(nside),
        ncells_dst = 12 * nside^2, ncells_src = 12 * (nside ÷ 2)^2, dst, src)
end

# Tier lists honoring a max-destination-cells cap (used to bound CI runtime).
oceananigans_tiers(; maxcells = typemax(Int)) =
    [oceananigans_pair(Ny) for Ny in OCEANANIGANS_NY if 2Ny^2 <= maxcells]
healpix_tiers(; maxcells = typemax(Int)) =
    [healpix_pair(nside) for nside in HEALPIX_NSIDE if 12 * nside^2 <= maxcells]

end # _BENCH_GRIDS_LOADED
