module ConservativeRegriddingRingGridsExt

import ConservativeRegridding
using ConservativeRegridding: Trees

using RingGrids

import ConservativeRegridding.Trees: treeify
import GeometryOpsCore: best_manifold, Manifold, Spherical
import GeometryOps as GO


best_manifold(grid::RingGrids.AbstractGrid) = Spherical()
best_manifold(field::RingGrids.AbstractField) = best_manifold(field.grid)

treeify(manifold::Spherical, field::RingGrids.AbstractField) = treeify(manifold, field.grid)

function treeify(manifold::Spherical, grid::RingGrids.AbstractGrid)
    error("Not implemented for $(typeof(grid))")
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

end
