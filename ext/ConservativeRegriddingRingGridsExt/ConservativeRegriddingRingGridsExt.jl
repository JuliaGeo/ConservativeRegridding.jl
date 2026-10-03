module ConservativeRegriddingRingGridsExt

import ConservativeRegridding
using ConservativeRegridding: Trees

using RingGrids

import ConservativeRegridding.Trees: treeify
import GeometryOpsCore: best_manifold, manifold, Manifold, Spherical
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
import GeoInterface as GI
import StaticArrays: SA


best_manifold(grid::RingGrids.AbstractGrid) = Spherical()
best_manifold(field::RingGrids.AbstractField) = best_manifold(field.grid)

treeify(manifold::Spherical, field::RingGrids.AbstractField) = treeify(manifold, field.grid)

# Fallback for every grid without a native tree below (e.g. the reduced octahedral
# Gaussian/Clenshaw/minimal grids): eagerly build each cell from RingGrids' own
# E/S/W/N vertices, in ring (data) order, and let `treeify` R-tree the vector.
function treeify(manifold::Spherical, grid::RingGrids.AbstractGrid; kwargs...)
    E, S, W, N = RingGrids.get_vertices(typeof(grid), grid.nlat_half)
    f = GO.UnitSphereFromGeographic()
    polygons = map(axes(E, 2)) do ij
        e, n, w, s = f((E[1, ij], E[2, ij])), f((N[1, ij], N[2, ij])), f((W[1, ij], W[2, ij])), f((S[1, ij], S[2, ij]))
        # RingGrids' vertices are clockwise; the convex-clip kernel needs CCW (E, N, W, S).
        GI.Polygon([GI.LinearRing([e, n, w, s, e])])
    end
    return treeify(manifold, polygons; kwargs...)
end

function treeify(manifold::Spherical, grid::RingGrids.AbstractFullGrid)
    latd = RingGrids.get_latd(grid)
    lond = RingGrids.get_lond(grid)
    nlat = length(latd)
    nlon = length(lond)

    # Pole-pinned latitude edges (north → south, length nlat + 1).
    lat_edges = Vector{Float64}(undef, nlat + 1)
    lat_edges[1]   =  90.0
    lat_edges[end] = -90.0
    @inbounds for j in 1:nlat - 1
        lat_edges[j + 1] = 0.5 * (latd[j] + latd[j + 1])
    end

    # Cell centers coincide with `lond`, so edges are shifted by half a cell.
    Δlon = 360 / nlon
    lon_edges = [lond[1] - Δlon / 2 + (i - 1) * Δlon for i in 1:nlon + 1]

    points = GO.UnitSphereFromGeographic().(
        [(lon_edges[i], lat_edges[nlat + 2 - j]) for i in 1:nlon + 1, j in 1:nlat + 1]
    )

    lin2cart = [CartesianIndex(i, nlat + 1 - ring) for ring in 1:nlat for i in 1:nlon]
    ordering = Trees.Reorderer2D(lin2cart, nlon, nlat)

    cell_grid = Trees.CellBasedGrid(manifold, points)
    tree      = Trees.ReorderedTopDownQuadtreeCursor(cell_grid, ordering)
    return Trees.KnownFullSphereExtentWrapper(tree)
end

# Reduced OctaHEALPix grids use a per-face range-subdivision quadtree (see octahealpix.jl).
include("octahealpix.jl")

# Standard 12-face HEALPix grids use a per-face range-subdivision quadtree (see healpix.jl).
include("healpix.jl")

end
