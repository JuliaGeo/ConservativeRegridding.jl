
abstract type AbstractQuadtreeCursor end

"""
    QuadtreeCursor(grid::AbstractCurvilinearGrid)

A cursor for traversing the quadtree.  This is used to traverse the quadtree in a depth-first manner, and to get the child nodes of the current node.

## Fields

$(DocStringExtensions.FIELDS)
"""
struct QuadtreeCursor{GridType <: AbstractCurvilinearGrid} <: AbstractQuadtreeCursor
    "The underlying grid."
    grid::GridType
    "The index of the current cell at the current level in the quadtree."
    idx::CartesianIndex{2}
    "The level of the cursor - 1 is the base i.e. smallest polygon level, as you increase the number you increase the size of the thing."
    level::Int
end


function QuadtreeCursor(grid::AbstractCurvilinearGrid)
    # Need 2^(level-1) >= max_size, so level >= log2(max_size) + 1
    max_level = ceil(Int, log2(max(ncells(grid, 1), ncells(grid, 2)))) + 1
    return QuadtreeCursor(grid, CartesianIndex(1, 1), max_level)
end

# ### General methods for any quadtree type

function leaf_idxs(q::QuadtreeCursor)
     # Calculate the range of leaf indices covered by this node
     scale = 2^(q.level - 1)
     psize = (ncells(q.grid, 1), ncells(q.grid, 2))
     
     # Compute and clamp all indices to polygon matrix bounds
     imin = min((q.idx[1] - 1) * scale + 1, psize[1])
     imax = min(q.idx[1] * scale, psize[1])
     jmin = min((q.idx[2] - 1) * scale + 1, psize[2])
     jmax = min(q.idx[2] * scale, psize[2])

     return imin:imax, jmin:jmax
end

function getcell(q::QuadtreeCursor)
    is, js = leaf_idxs(q)
    return (getcell(q.grid, i, j) for i in is, j in js)
end

function getcell(q::QuadtreeCursor, i, j)
    return getcell(q.grid, i, j)
end

STI.isspatialtree(::Type{<: QuadtreeCursor}) = true
function STI.nchild(q::QuadtreeCursor)
    # Children are at level q.level - 1, each covering child_scale cells
    child_scale = 2^(q.level - 2)

    # Parent's start position in each dimension
    parent_start_i = (q.idx[1] - 1) * 2^(q.level - 1) + 1
    parent_start_j = (q.idx[2] - 1) * 2^(q.level - 1) + 1

    # Second-half children start at parent_start + child_scale
    second_half_start_i = parent_start_i + child_scale
    second_half_start_j = parent_start_j + child_scale

    ncells_i = ncells(q.grid, 1)
    ncells_j = ncells(q.grid, 2)

    # Count children in each dimension:
    # - 2 if second-half children's start is within bounds
    # - 1 if only first-half children exist
    # - 0 if parent itself is out of bounds
    ni = if second_half_start_i <= ncells_i
        2
    elseif parent_start_i <= ncells_i
        1
    else
        0
    end

    nj = if second_half_start_j <= ncells_j
        2
    elseif parent_start_j <= ncells_j
        1
    else
        0
    end

    return ni * nj
end

function STI.isleaf(q::QuadtreeCursor)
    # A node is a leaf if it's at level 2 (natural leaf level) OR if it has no children
    # (edge case for odd-sized grids where boundary nodes can't split further)
    return q.level <= 2 || STI.nchild(q) == 0
end

function STI.child_indices_extents(q::QuadtreeCursor)
    @assert STI.isleaf(q) "Child indices and extents are only valid for leaf nodes."
    irange, jrange = leaf_idxs(q)
    idxs = CartesianIndices((irange, jrange))
    # Create an extent for each cell in the leaf range
    extents = [cell_range_extent(q.grid, i:i, j:j) for i in irange, j in jrange]
    return zip((cartesian_to_linear_idx(q.grid, idx) for idx in idxs), extents)
end

function STI.getchild(q::QuadtreeCursor, child_idx::Int)
    child_idx > STI.nchild(q) && throw(ArgumentError("Invalid child index; got $child_idx, but there are only $(STI.nchild(q)) children in the node."))

    # Determine which children exist based on bounds
    child_scale = 2^(q.level - 2)
    parent_start_i = (q.idx[1] - 1) * 2^(q.level - 1) + 1
    parent_start_j = (q.idx[2] - 1) * 2^(q.level - 1) + 1
    second_half_start_i = parent_start_i + child_scale
    second_half_start_j = parent_start_j + child_scale
    ncells_i = ncells(q.grid, 1)
    ncells_j = ncells(q.grid, 2)

    has_second_i = second_half_start_i <= ncells_i
    has_second_j = second_half_start_j <= ncells_j

    # Map child_idx to (di, dj) offset based on which children exist
    # Children are indexed column-major: (1,1), (2,1), (1,2), (2,2) for full 2×2
    if has_second_i && has_second_j
        # 2×2 layout
        offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    elseif has_second_i
        # 2×1 layout (only first column of j)
        offsets = ((0, 0), (1, 0))
    elseif has_second_j
        # 1×2 layout (only first row of i)
        offsets = ((0, 0), (0, 1))
    else
        # 1×1 layout
        offsets = ((0, 0),)
    end

    di, dj = offsets[child_idx]
    new_i = 2 * (q.idx[1] - 1) + 1 + di
    new_j = 2 * (q.idx[2] - 1) + 1 + dj

    return QuadtreeCursor(q.grid, CartesianIndex(new_i, new_j), q.level - 1)
end

function _corner_and_other_points_for_circle_from_corners(q::QuadtreeCursor{<: CellBasedGrid})
    if STI.isleaf(q)
        idx = q.idx.I
        pmat = q.grid.points
        pts = (pmat[idx[1], idx[2]], pmat[idx[1], idx[2]+1], pmat[idx[1]+1, idx[2]+1], pmat[idx[1]+1, idx[2]])
        return (pts, pts)
    else
        # Compute and clamp all indices to polygon matrix bounds
        irange, jrange = leaf_idxs(q)
        imin, imax = extrema(irange)
        jmin, jmax = extrema(jrange)

        grid_points = q.grid.points
        corner_points = (grid_points[imin, jmin], grid_points[imin, jmax], grid_points[imax, jmin], grid_points[imax, jmax])

        # Collect points from all border polygons
        other_points = typeof(GI.getpoint(GI.getexterior(getcell(q.grid, imin, jmin)), 1))[]
        sizehint!(other_points, (imax - imin + 1) * (jmax - jmin + 1))
        # Top and bottom rows (all columns)
        append!(other_points, view(q.grid.points, imin, jmin:jmax))
        if imax != imin
            append!(other_points, view(q.grid.points, imax, jmin:jmax))
        end
        if jmax != jmin
            append!(other_points, view(q.grid.points, imin:imax, jmax))
        end

        return (corner_points, other_points)
    end
end

function STI.node_extent(q::QuadtreeCursor)
    return cell_range_extent(q.grid, leaf_idxs(q)...)
end

function istoplevel(q::QuadtreeCursor)
    max_level = ceil(Int, log2(max(ncells(q.grid, 1), ncells(q.grid, 2)))) + 1
    return q.level == max_level && q.idx == CartesianIndex(1, 1)
end

#=
## TopDownQuadtreeCursor

The idea here is to divide the grid into four quadrants instead of assembling it from 2x2 squares.
=#

struct TopDownQuadtreeCursor{GridType <: AbstractCurvilinearGrid} <: AbstractQuadtreeCursor
    grid::GridType
    leafranges::NTuple{2, UnitRange{Int}}
end
function TopDownQuadtreeCursor(grid::AbstractCurvilinearGrid)
    return TopDownQuadtreeCursor(grid, (1:ncells(grid, 1), 1:ncells(grid, 2)))
end

getgrid(q::TopDownQuadtreeCursor) = q.grid

function Base.show(io::IO, q::TopDownQuadtreeCursor)
    print(io, "TopDownQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end
function Base.show(io::IO, ::MIME"text/plain", q::TopDownQuadtreeCursor)
    print(io, "TopDownQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end

STI.isspatialtree(::Type{<: TopDownQuadtreeCursor}) = true

function STI.isleaf(q::TopDownQuadtreeCursor)
    return all(length.(q.leafranges) .<= 2)
end

function STI.child_indices_extents(q::TopDownQuadtreeCursor)
    idxs = (cartesian_to_linear_idx(q.grid, CartesianIndex((i, j))) for i in q.leafranges[1], j in q.leafranges[2])
    extents = (cell_range_extent(q.grid, i:i, j:j) for i in q.leafranges[1], j in q.leafranges[2])
    return zip(idxs, extents)
end

function STI.nchild(q::TopDownQuadtreeCursor)
    i_is_one = length(q.leafranges[1]) == 1 # length-1 in i
    j_is_one = length(q.leafranges[2]) == 1 # length-1 in j

    if i_is_one && j_is_one
        error("This should be unreachable - `irange` is length 1 and so is `jrange`")
    elseif i_is_one
        return 2
    elseif j_is_one
        return 2
    else
        return 4
    end
end

function STI.getchild(q::TopDownQuadtreeCursor, i::Int)
    i_is_one = length(q.leafranges[1]) == 1 # length-1 in i
    j_is_one = length(q.leafranges[2]) == 1 # length-1 in j

    vals = if i_is_one && j_is_one
        error("This should be unreachable - `irange` is length 1 and so is `jrange`")
    elseif i_is_one
        j_split_point = length(q.leafranges[2]) ÷ 2
        (
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1], q.leafranges[2][1:j_split_point])),
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1], q.leafranges[2][j_split_point+1:end]))
        )
    elseif j_is_one
        i_split_point = length(q.leafranges[1]) ÷ 2
        (
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2])),
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2]))
        )
    else
        i_split_point = length(q.leafranges[1]) ÷ 2
        j_split_point = length(q.leafranges[2]) ÷ 2
        (
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2][1:j_split_point])),
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2][j_split_point+1:end])),
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2][1:j_split_point])),
            TopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2][j_split_point+1:end]))
        )
    end
    return vals[i]
end

function STI.getchild(q::TopDownQuadtreeCursor)
    return (STI.getchild(q, i) for i in 1:STI.nchild(q))
end

function STI.node_extent(q::TopDownQuadtreeCursor)
    return cell_range_extent(q.grid, q.leafranges[1], q.leafranges[2])
end

function getcell(q::TopDownQuadtreeCursor)
    return (getcell(q.grid, i, j) for i in q.leafranges[1], j in q.leafranges[2])
end

function getcell(q::TopDownQuadtreeCursor, i::Int)
    leaf_ij = linear_to_cartesian_idx(q.grid, i)
    return getcell(q.grid, leaf_ij)
end

function ncells(q::TopDownQuadtreeCursor, dim::Int)
    return length(q.leafranges[dim])
end

function ncells(q::TopDownQuadtreeCursor)
    return length.(q.leafranges)
end

function istoplevel(q::TopDownQuadtreeCursor)
    return length(q.leafranges[1]) == ncells(q.grid, 1) && length(q.leafranges[2]) == ncells(q.grid, 2)
end