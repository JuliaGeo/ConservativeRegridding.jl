#=
# Wrappers

For trees, wrappers are used to selectively override certain SpatialTreeInterface methods.

This is useful if e.g. you know that a tree covers the entire sphere, 
and you want to avoid extent calculations there by substituting a full sphere extent.

Wrappers are also generally good for other things -- like caching, or adding additional metadata.
=#

import GeometryOps: SpatialTreeInterface as STI

abstract type AbstractTreeWrapper end
Base.parent(wrapper::AbstractTreeWrapper) = error("Base.parent not implemented for $(typeof(wrapper))\nThis must be implemented as it is a core part of the STI interface.")
# In general, everything will forward back to the parent tree - except what you override.
STI.isspatialtree(::Type{<: AbstractTreeWrapper}) = true
STI.nchild(wrapper::AbstractTreeWrapper) = STI.nchild(parent(wrapper))
STI.getchild(wrapper::AbstractTreeWrapper, i::Int) = STI.getchild(parent(wrapper), i)
STI.isleaf(wrapper::AbstractTreeWrapper) = STI.isleaf(parent(wrapper))
STI.child_indices_extents(wrapper::AbstractTreeWrapper) = STI.child_indices_extents(parent(wrapper))
STI.node_extent(wrapper::AbstractTreeWrapper) = STI.node_extent(parent(wrapper))

getcell(wrapper::AbstractTreeWrapper, args...) = getcell(parent(wrapper), args...)
ncells(wrapper::AbstractTreeWrapper, args...) = ncells(parent(wrapper), args...)
cell_range_extent(wrapper::AbstractTreeWrapper, args...) = cell_range_extent(parent(wrapper), args...)

#=
## KnownFullSphereExtentWrapper

The idea here is that when you try to determine the spherical cap for a grid,
it's very hard to determine that the grid covers the entire sphere - there's 
usually a grid-cell-sized hole from most algorithms.  And they take a while.

So instead of doing that we can just intelligently wrap in this.
=#

"""
    KnownFullSphereExtentWrapper(tree)

A wrapper around a SpatialTreeInterface-compliant spatial tree that
will always return a `node_extent` that is a spherical cap covering the entire sphere.

All other STI methods are forwarded to the wrapped tree.

This is useful for trees that are known to cover the entire sphere, since the user
does not have to compute large extents.
"""
struct KnownFullSphereExtentWrapper{T} <: AbstractTreeWrapper
    tree::T
    function KnownFullSphereExtentWrapper(tree::T) where T
        if !STI.isspatialtree(tree)
            throw(ArgumentError("Tree is not a spatial tree"))
        end
        return new{T}(tree)
    end
end
Base.parent(w::KnownFullSphereExtentWrapper) = w.tree

# This is the only specialization - you want a full sphere extent
# This is actually larger than a full sphere but it's fine for "extent check" style things.  
# 
# If anyone tries to make the cap a polygon then we'll have a problem, but we would have had one anyway,
# since we don't have a great way to represent a `POLYGON FULL` (in WKT parlance).
STI.node_extent(w::KnownFullSphereExtentWrapper) = GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint((0.,0.,1.)), Float64(pi) |> nextfloat)

struct GeometryMaintainingTreeWrapper{Geoms, Tree}
    geoms::Geoms
    tree::Tree
end

Base.parent(wrapper::GeometryMaintainingTreeWrapper) = wrapper.tree
# In general, everything will forward back to the parent tree - except what you override.
STI.isspatialtree(::Type{<: GeometryMaintainingTreeWrapper}) = true
STI.nchild(wrapper::GeometryMaintainingTreeWrapper) = STI.nchild(parent(wrapper))
STI.getchild(wrapper::GeometryMaintainingTreeWrapper, i::Int) = GeometryMaintainingTreeWrapper(wrapper.geoms, STI.getchild(parent(wrapper), i))
STI.isleaf(wrapper::GeometryMaintainingTreeWrapper) = STI.isleaf(parent(wrapper))
STI.child_indices_extents(wrapper::GeometryMaintainingTreeWrapper) = STI.child_indices_extents(parent(wrapper))
STI.node_extent(wrapper::GeometryMaintainingTreeWrapper) = STI.node_extent(parent(wrapper))

getcell(wrapper::GeometryMaintainingTreeWrapper, args...) = getcell(wrapper.geoms, args...)
ncells(wrapper::GeometryMaintainingTreeWrapper, args...) = ncells(wrapper.tree, args...)
cell_range_extent(wrapper::GeometryMaintainingTreeWrapper, args...) = cell_range_extent(wrapper.geoms, args...)

getcell(wrapper::GeometryMaintainingTreeWrapper{G, T}) where {G <: AbstractVector, T} = wrapper.geoms
getcell(wrapper::GeometryMaintainingTreeWrapper{G, T}, i::Integer) where {G <: AbstractVector, T} = wrapper.geoms[i]


#=
## CubedSphereToplevelTree

A wrapper around a vector of quadtree cursors that represents a cubed sphere.

This is effectively the prototype of a tree node that contains a vector of potentially many different kinds of trees directly.
It might be replaced by something more general later.
=#
struct CubedSphereToplevelTree{QuadtreeCursorType <: Trees.AbstractQuadtreeCursor}
    quadtrees::Vector{QuadtreeCursorType}
end

STI.isspatialtree(::Type{<: CubedSphereToplevelTree}) = true
STI.isleaf(::CubedSphereToplevelTree) = false
STI.nchild(cs::CubedSphereToplevelTree) = length(cs.quadtrees)
STI.getchild(cs::CubedSphereToplevelTree, i::Int) = cs.quadtrees[i]
STI.node_extent(cs::CubedSphereToplevelTree) = GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint((0.,0.,1.)), Float64(pi) |> nextfloat)
GOCore.best_manifold(c::CubedSphereToplevelTree) = GOCore.best_manifold(first(c.quadtrees))

Trees.getcell(c::CubedSphereToplevelTree) = Iterators.flatten(Trees.getcell(qt) for qt in c.quadtrees)
function Trees.getcell(c::CubedSphereToplevelTree, i::Int)
    n_cells_in_face = prod(Trees.ncells(first(c.quadtrees)))
    face_idx = fld(i - 1, n_cells_in_face) + 1
    # face_local_idx = i - (face_idx - 1) * n_cells_in_face
    return Trees.getcell(c.quadtrees[face_idx], i)
end

Trees.ncells(c::CubedSphereToplevelTree) = prod(Trees.ncells(first(c.quadtrees))) * 6