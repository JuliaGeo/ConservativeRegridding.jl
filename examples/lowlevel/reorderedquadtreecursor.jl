import GeometryOps as GO


#=
## ReorderedTopDownQuadtreeCursor

The idea here is to divide the grid into four quadrants instead of assembling it from 2x2 squares.
=#

struct Reorderer2D
    cart2lin::AbstractMatrix{Int}
    lin2cart::AbstractVector{CartesianIndex{2}}
end

function Reorderer2D(lin2cart::AbstractVector{CartesianIndex{2}}, n, m)
    cart2lin = zeros(Int, n, m)
    for (i, idx) in enumerate(lin2cart)
        cart2lin[idx] = i
    end
    return Reorderer2D(cart2lin, lin2cart)
end

struct ReorderedTopDownQuadtreeCursor{GridType <: Trees.AbstractCurvilinearGrid} <: Trees.AbstractQuadtreeCursor
    grid::GridType
    leafranges::NTuple{2, UnitRange{Int}}
    ordering::Reorderer2D
end
function ReorderedTopDownQuadtreeCursor(grid::Trees.AbstractCurvilinearGrid, ordering::Reorderer2D)
    @assert size(ordering.cart2lin) == (Trees.ncells(grid, 1), Trees.ncells(grid, 2))
    return ReorderedTopDownQuadtreeCursor(grid, (1:Trees.ncells(grid, 1), 1:Trees.ncells(grid, 2)), ordering)
end

Trees.getgrid(q::ReorderedTopDownQuadtreeCursor) = q.grid

function Base.show(io::IO, q::ReorderedTopDownQuadtreeCursor)
    print(io, "ReorderedTopDownQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end
function Base.show(io::IO, ::MIME"text/plain", q::ReorderedTopDownQuadtreeCursor)
    print(io, "ReorderedTopDownQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end

STI.isspatialtree(::Type{<: ReorderedTopDownQuadtreeCursor}) = true

function STI.isleaf(q::ReorderedTopDownQuadtreeCursor)
    return all(length.(q.leafranges) .<= 2)
end

function STI.child_indices_extents(q::ReorderedTopDownQuadtreeCursor)
    idxs = (q.ordering.cart2lin[CartesianIndex(i, j)] for i in q.leafranges[1], j in q.leafranges[2])
    extents = (Trees.cell_range_extent(q.grid, i:i, j:j) for i in q.leafranges[1], j in q.leafranges[2])
    return zip(idxs, extents)
end

function STI.nchild(q::ReorderedTopDownQuadtreeCursor)
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

function STI.getchild(q::ReorderedTopDownQuadtreeCursor, i::Int)
    i_is_one = length(q.leafranges[1]) == 1 # length-1 in i
    j_is_one = length(q.leafranges[2]) == 1 # length-1 in j

    vals = if i_is_one && j_is_one
        error("This should be unreachable - `irange` is length 1 and so is `jrange`")
    elseif i_is_one
        j_split_point = length(q.leafranges[2]) ÷ 2
        (
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1], q.leafranges[2][1:j_split_point]), q.ordering),
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1], q.leafranges[2][j_split_point+1:end]), q.ordering)
        )
    elseif j_is_one
        i_split_point = length(q.leafranges[1]) ÷ 2
        (
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2]), q.ordering),
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2]), q.ordering)
        )
    else
        i_split_point = length(q.leafranges[1]) ÷ 2
        j_split_point = length(q.leafranges[2]) ÷ 2
        (
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2][1:j_split_point]), q.ordering),
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][1:i_split_point], q.leafranges[2][j_split_point+1:end]), q.ordering),
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2][1:j_split_point]), q.ordering),
            ReorderedTopDownQuadtreeCursor(q.grid, (q.leafranges[1][i_split_point+1:end], q.leafranges[2][j_split_point+1:end]), q.ordering)
        )
    end
    return vals[i]
end

function STI.getchild(q::ReorderedTopDownQuadtreeCursor)
    return (STI.getchild(q, i) for i in 1:STI.nchild(q))
end

function STI.node_extent(q::ReorderedTopDownQuadtreeCursor)
    return Trees.cell_range_extent(q.grid, q.leafranges[1], q.leafranges[2])
end

function Trees.getcell(q::ReorderedTopDownQuadtreeCursor)
    order = sortperm(vec(q.ordering.cart2lin[q.leafranges[1], q.leafranges[2]]))
    return (Trees.getcell(q.grid, CartesianIndices(q.leafranges)[i]) for i in order)
end

function Trees.getcell(q::ReorderedTopDownQuadtreeCursor, i::Int)
    leaf_ij = q.ordering.lin2cart[i]
    return Trees.getcell(q.grid, leaf_ij)
end

function Trees.ncells(q::ReorderedTopDownQuadtreeCursor, dim::Int)
    return length(q.leafranges[dim])
end

function Trees.ncells(q::ReorderedTopDownQuadtreeCursor)
    return length.(q.leafranges)
end

function istoplevel(q::ReorderedTopDownQuadtreeCursor)
    return length(q.leafranges[1]) == Trees.ncells(q.grid, 1) && length(q.leafranges[2]) == Trees.ncells(q.grid, 2)
end

