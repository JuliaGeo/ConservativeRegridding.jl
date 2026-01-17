#=
# Adjacency Computation

Compute cell adjacency for gradient computation in 2nd order regridding.
Neighbors are cells sharing edges or vertices (8-connectivity for quad grids).

Multiple dispatch provides fast paths for structured grids.
=#

import GeometryOpsCore as GOCore

"""
    compute_adjacency(manifold, tree) -> Vector{Vector{Int}}

Compute adjacency list for all cells in the tree.
Returns a vector where `adjacency[i]` contains the linear indices of all neighbors of cell `i`.

For structured grids (CellBasedGrid, RegularGrid, ExplicitPolygonGrid), uses fast index arithmetic.
For unstructured grids, falls back to spatial queries.
"""
function compute_adjacency end

# Fast path for structured grids (CellBasedGrid, RegularGrid, ExplicitPolygonGrid)
function compute_adjacency(manifold::GOCore.Manifold, tree::Trees.TopDownQuadtreeCursor{<:Trees.AbstractCurvilinearGrid})
    grid = Trees.getgrid(tree)
    return _compute_structured_adjacency(grid)
end

"""
    _compute_structured_adjacency(grid) -> Vector{Vector{Int}}

Compute 8-connectivity adjacency for a structured grid using index arithmetic.
"""
function _compute_structured_adjacency(grid::Trees.AbstractCurvilinearGrid)
    ni = Trees.ncells(grid, 1)
    nj = Trees.ncells(grid, 2)
    n_total = ni * nj

    adjacency = Vector{Vector{Int}}(undef, n_total)

    for j in 1:nj, i in 1:ni
        idx = i + (j - 1) * ni  # column-major linear index
        neighbors = Int[]

        # 8-connectivity: all adjacent cells including diagonals
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ni_new, nj_new = i + di, j + dj
            if 1 <= ni_new <= ni && 1 <= nj_new <= nj
                neighbor_idx = ni_new + (nj_new - 1) * ni
                push!(neighbors, neighbor_idx)
            end
        end

        adjacency[idx] = neighbors
    end

    return adjacency
end

# Generic fallback for unstructured grids (spatial queries)
# TODO: Implement using dual DFS to find touching polygons
function compute_adjacency(manifold::GOCore.Manifold, tree)
    error("compute_adjacency not yet implemented for unstructured grids ($(typeof(tree))). " *
          "Use a structured grid (CellBasedGrid, RegularGrid) for 2nd order regridding.")
end
