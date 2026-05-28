function has_optimized_idx_search end # (grid)::Bool
function findidx end # (grid, pos)::idx
function has_optimized_neighbour_search end # (grid)
function neighbours end # (grid, idx)::idxs
function dual_neighbours end # (grid, pos, neighbours)::idxs

# Question here: should a cache also exhibit the tree interface?
# Or be completely separate?
abstract type AbstractNeighbourCache end
function neighbours(cache::AbstractNeighbourCache, position) end
function dual_neighbours(cache::AbstractNeighbourCache, position) end
