module ConservativeRegriddingRingGridsExt

import ConservativeRegridding
using ConservativeRegridding: Trees

using RingGrids

import ConservativeRegridding.Trees: treeify
import GeometryOpsCore: best_manifold, Manifold, Spherical


best_manifold(grid::RingGrids.AbstractGrid) = Spherical()
best_manifold(field::RingGrids.AbstractField) = best_manifold(field.grid)

treeify(manifold::Spherical, field::RingGrids.AbstractField) = treeify(manifold, field.grid)

function treeify(manifold::Spherical, grid::RingGrids.AbstractGrid)
    error("Not implemented for $(typeof(grid))")
end

end