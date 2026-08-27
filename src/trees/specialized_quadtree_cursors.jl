import GeometryOps as GO


#=
## IndexOffsetQuadtreeCursor

Provides and accepts global indices for polygons by applying
an `index_offset` to the grid-local linear indices. This is
useful for multi-grid scenarios (e.g., cubed sphere faces)
where each sub-grid's indices must be offset into a global
index space.
=#

struct IndexOffsetQuadtreeCursor{GridType <: Trees.AbstractCurvilinearGrid} <: Trees.AbstractQuadtreeCursor
    grid::GridType
    leafranges::NTuple{2, UnitRange{Int}}
    index_offset::Int
end
function IndexOffsetQuadtreeCursor(grid::Trees.AbstractCurvilinearGrid, index_offset)
    return IndexOffsetQuadtreeCursor(grid, (1:Trees.ncells(grid, 1), 1:Trees.ncells(grid, 2)), index_offset)
end

Trees.getgrid(q::IndexOffsetQuadtreeCursor) = q.grid

function Base.show(io::IO, q::IndexOffsetQuadtreeCursor)
    print(io, "IndexOffsetQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end
function Base.show(io::IO, ::MIME"text/plain", q::IndexOffsetQuadtreeCursor)
    print(io, "IndexOffsetQuadtreeCursor(($(q.leafranges[1])), ($(q.leafranges[2])))")
end

STI.isspatialtree(::Type{<: IndexOffsetQuadtreeCursor}) = true

function STI.isleaf(q::IndexOffsetQuadtreeCursor)
    return all(length.(q.leafranges) .<= 2)
end

@inline function _leaf_entry(q::IndexOffsetQuadtreeCursor, ij::CartesianIndex{2})
    i, j = Tuple(ij)
    index = Trees.cartesian_to_linear_idx(q.grid, ij) + q.index_offset
    return (index, Trees.cell_range_extent(q.grid, i:i, j:j))
end

function STI.child_indices_extents(q::IndexOffsetQuadtreeCursor)
    @assert STI.isleaf(q) "Child indices and extents are only valid for leaf nodes."
    return _materialize_leaf_entries(q, CartesianIndices(q.leafranges), Val(4))
end

function STI.nchild(q::IndexOffsetQuadtreeCursor)
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

function STI.getchild(q::IndexOffsetQuadtreeCursor, i::Int)
    n = STI.nchild(q)
    1 <= i <= n || throw(BoundsError(q, i))

    i_is_one = length(q.leafranges[1]) == 1 # length-1 in i
    j_is_one = length(q.leafranges[2]) == 1 # length-1 in j

    if i_is_one
        j_split_point = length(q.leafranges[2]) ÷ 2
        jrange = i == 1 ? q.leafranges[2][1:j_split_point] :
                          q.leafranges[2][j_split_point+1:end]
        return IndexOffsetQuadtreeCursor(
            q.grid, (q.leafranges[1], jrange), q.index_offset)
    elseif j_is_one
        i_split_point = length(q.leafranges[1]) ÷ 2
        irange = i == 1 ? q.leafranges[1][1:i_split_point] :
                          q.leafranges[1][i_split_point+1:end]
        return IndexOffsetQuadtreeCursor(
            q.grid, (irange, q.leafranges[2]), q.index_offset)
    else
        i_split_point = length(q.leafranges[1]) ÷ 2
        j_split_point = length(q.leafranges[2]) ÷ 2
        irange = i <= 2 ? q.leafranges[1][1:i_split_point] :
                          q.leafranges[1][i_split_point+1:end]
        jrange = isodd(i) ? q.leafranges[2][1:j_split_point] :
                            q.leafranges[2][j_split_point+1:end]
        return IndexOffsetQuadtreeCursor(
            q.grid, (irange, jrange), q.index_offset)
    end
end

function STI.getchild(q::IndexOffsetQuadtreeCursor)
    return (STI.getchild(q, i) for i in 1:STI.nchild(q))
end

function STI.node_extent(q::IndexOffsetQuadtreeCursor)
    return Trees.cell_range_extent(q.grid, q.leafranges[1], q.leafranges[2])
end

STI.node_extent_is_expensive(::Type{<: IndexOffsetQuadtreeCursor{G}}) where {G} =
    Trees.extent_is_expensive(G)

function Trees.getcell(q::IndexOffsetQuadtreeCursor)
    return (Trees.getcell(q.grid, ij) for ij in CartesianIndices(q.leafranges))
end

function Trees.getcell(q::IndexOffsetQuadtreeCursor, i::Int)
    leaf_ij = Trees.linear_to_cartesian_idx(q.grid, i - q.index_offset)

    return try
        Trees.getcell(q.grid, leaf_ij)
    catch e
        @show i q.index_offset
        rethrow(e)
    end
end

function Trees.ncells(q::IndexOffsetQuadtreeCursor, dim::Int)
    return length(q.leafranges[dim])
end

function Trees.ncells(q::IndexOffsetQuadtreeCursor)
    return length.(q.leafranges)
end

Trees.cell_index_count(q::IndexOffsetQuadtreeCursor) =
    q.index_offset + Trees.cell_index_count(q.grid)

function istoplevel(q::IndexOffsetQuadtreeCursor)
    return length(q.leafranges[1]) == Trees.ncells(q.grid, 1) && length(q.leafranges[2]) == Trees.ncells(q.grid, 2)
end



















import GeometryOps as GO


#=
## ReorderedTopDownQuadtreeCursor

The idea here is to divide the grid into four quadrants instead of assembling it from 2x2 squares.
=#

struct Reorderer2D{
        CartToLinear <: AbstractMatrix{Int},
        LinearToCart <: AbstractVector{CartesianIndex{2}},
    }
    cart2lin::CartToLinear
    lin2cart::LinearToCart
end

function Reorderer2D(lin2cart::AbstractVector{CartesianIndex{2}}, n, m)
    cart2lin = zeros(Int, n, m)
    for (i, idx) in enumerate(lin2cart)
        cart2lin[idx] = i
    end
    return Reorderer2D(cart2lin, lin2cart)
end

struct ReorderedTopDownQuadtreeCursor{
        GridType <: Trees.AbstractCurvilinearGrid,
        Ordering <: Reorderer2D,
    } <: Trees.AbstractQuadtreeCursor
    grid::GridType
    leafranges::NTuple{2, UnitRange{Int}}
    ordering::Ordering
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

@inline function _leaf_entry(q::ReorderedTopDownQuadtreeCursor, ij::CartesianIndex{2})
    i, j = Tuple(ij)
    index = q.ordering.cart2lin[ij]
    return (index, Trees.cell_range_extent(q.grid, i:i, j:j))
end

function STI.child_indices_extents(q::ReorderedTopDownQuadtreeCursor)
    @assert STI.isleaf(q) "Child indices and extents are only valid for leaf nodes."
    return _materialize_leaf_entries(q, CartesianIndices(q.leafranges), Val(4))
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
    n = STI.nchild(q)
    1 <= i <= n || throw(BoundsError(q, i))

    i_is_one = length(q.leafranges[1]) == 1 # length-1 in i
    j_is_one = length(q.leafranges[2]) == 1 # length-1 in j

    if i_is_one
        j_split_point = length(q.leafranges[2]) ÷ 2
        jrange = i == 1 ? q.leafranges[2][1:j_split_point] :
                          q.leafranges[2][j_split_point+1:end]
        return ReorderedTopDownQuadtreeCursor(
            q.grid, (q.leafranges[1], jrange), q.ordering)
    elseif j_is_one
        i_split_point = length(q.leafranges[1]) ÷ 2
        irange = i == 1 ? q.leafranges[1][1:i_split_point] :
                          q.leafranges[1][i_split_point+1:end]
        return ReorderedTopDownQuadtreeCursor(
            q.grid, (irange, q.leafranges[2]), q.ordering)
    else
        i_split_point = length(q.leafranges[1]) ÷ 2
        j_split_point = length(q.leafranges[2]) ÷ 2
        irange = i <= 2 ? q.leafranges[1][1:i_split_point] :
                          q.leafranges[1][i_split_point+1:end]
        jrange = isodd(i) ? q.leafranges[2][1:j_split_point] :
                            q.leafranges[2][j_split_point+1:end]
        return ReorderedTopDownQuadtreeCursor(
            q.grid, (irange, jrange), q.ordering)
    end
end

function STI.getchild(q::ReorderedTopDownQuadtreeCursor)
    return (STI.getchild(q, i) for i in 1:STI.nchild(q))
end

function STI.node_extent(q::ReorderedTopDownQuadtreeCursor)
    return Trees.cell_range_extent(q.grid, q.leafranges[1], q.leafranges[2])
end

STI.node_extent_is_expensive(::Type{<: ReorderedTopDownQuadtreeCursor{G}}) where {G} =
    Trees.extent_is_expensive(G)

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

# `cart2lin` may map a local grid (for example one cubed-sphere face) into a
# larger global data layout.  Its maximum is the end of the dense one-based
# domain needed to store every emitted leaf index.
Trees.cell_index_count(q::ReorderedTopDownQuadtreeCursor) =
    maximum(q.ordering.cart2lin)

function istoplevel(q::ReorderedTopDownQuadtreeCursor)
    return length(q.leafranges[1]) == Trees.ncells(q.grid, 1) && length(q.leafranges[2]) == Trees.ncells(q.grid, 2)
end
