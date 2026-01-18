module ConservativeRegriddingHealpixExt


using ConservativeRegridding
using ConservativeRegridding.Trees

import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI, UnitSpherical
using StaticArrays: SA
using LinearAlgebra: normalize

# Import Healpix as a regular Julia package
import Healpix

GOCore.best_manifold(::Healpix.HealpixMap{T, O}) where {T, O <: Healpix.Order} = GO.Spherical()

Trees.treeify(::GO.Spherical, map::Healpix.HealpixMap{T, O}) where {T, O <: Healpix.Order} = HealpixRootNode{O}(map.resolution.nside)

#=
## Core Types
=#

"""
    HealpixRootNode{O <: Healpix.Order}

The entry point for a HEALPix spatial tree, representing the full sphere with 12 base face children.

- `O` parameter selects index ordering for `getcell`: `Healpix.NestedOrder` or `Healpix.RingOrder`
- `nside_max` is the leaf resolution (must be power of 2)
"""
struct HealpixRootNode{O <: Healpix.Order}
    nside_max::Int
end

"""
    HealpixTreeNode{O <: Healpix.Order}

Represents a pixel at a specific level in the HEALPix hierarchy.

- Level 0 contains the 12 base faces (pixels 0-11)
- Each non-leaf node has exactly 4 children
- Leaf level is `log2(nside_max)`
- Pixel index always stored in nested ordering (hierarchy only exists there)
"""
struct HealpixTreeNode{O <: Healpix.Order}
    nside_max::Int  # Leaf resolution
    level::Int      # Current level (0 = base faces, log2(nside_max) = leaves)
    pixel::Int      # Pixel index in nested ordering (0-based)
end

#=
## Convenience Constructors
=#

"""
    HealpixTree(nside::Int)
    HealpixTree(O, nside::Int)

Create a HEALPix spatial tree with the given resolution (nside).
Default ordering is NestedOrder. Pass ordering type as first argument for other orderings.
"""
HealpixTree(nside::Int) = HealpixRootNode{Healpix.NestedOrder}(nside)
HealpixTree(::Type{O}, nside::Int) where O <: Healpix.Order = HealpixRootNode{O}(nside)

#=
## Helper Functions
=#

"""
    get_pixel_corners_nested(res::Healpix.Resolution, nested_pix::Int)

Get the 4 corner points of a HEALPix pixel as UnitSphericalPoints.
The pixel index should be 0-based and in nested ordering.
"""
function get_pixel_corners_nested(res::Healpix.Resolution, nested_pix::Int)
    # boundariesRing needs ring-order pixel (1-based)
    ring_pix = Healpix.nest2ring(res, nested_pix + 1)

    # Get 4 corner points as 3D Cartesian vectors (step=1 gives 4 points)
    cartesian = Healpix.boundariesRing(res, ring_pix, 1, Float64)

    # Convert to UnitSphericalPoints (already on unit sphere)
    return (
        GO.UnitSphericalPoint(cartesian[1, 1], cartesian[1, 2], cartesian[1, 3]),
        GO.UnitSphericalPoint(cartesian[2, 1], cartesian[2, 2], cartesian[2, 3]),
        GO.UnitSphericalPoint(cartesian[3, 1], cartesian[3, 2], cartesian[3, 3]),
        GO.UnitSphericalPoint(cartesian[4, 1], cartesian[4, 2], cartesian[4, 3]),
    )
end

#=
## SpatialTreeInterface Implementation - HealpixRootNode
=#

STI.isspatialtree(::Type{<:HealpixRootNode}) = true
STI.isleaf(::HealpixRootNode) = false
STI.nchild(::HealpixRootNode) = 12

function STI.getchild(node::HealpixRootNode{O}, i::Int) where O
    # Children are the 12 base faces at level 0
    return HealpixTreeNode{O}(node.nside_max, 0, i - 1)  # 0-based pixel
end

function STI.node_extent(node::HealpixRootNode)
    # Full sphere cap
    return GO.UnitSpherical.SphericalCap(
        GO.UnitSphericalPoint(0.0, 0.0, 1.0),
        Float64(π) |> nextfloat
    )
end

#=
## SpatialTreeInterface Implementation - HealpixTreeNode
=#

STI.isspatialtree(::Type{<:HealpixTreeNode}) = true
STI.isleaf(node::HealpixTreeNode) = node.level == Int(log2(node.nside_max))
STI.nchild(node::HealpixTreeNode) = STI.isleaf(node) ? 0 : 4

function STI.getchild(node::HealpixTreeNode{O}, i::Int) where O
    # Nested ordering child relationship: child = 4*parent + offset
    child_pixel = 4 * node.pixel + (i - 1)  # Convert 1-based to 0-based offset
    return HealpixTreeNode{O}(node.nside_max, node.level + 1, child_pixel)
end

function STI.node_extent(node::HealpixTreeNode)
    # Compute spherical cap from the pixel corners at this level
    res_at_level = Healpix.Resolution(2^node.level)
    corners = get_pixel_corners_nested(res_at_level, node.pixel)
    return Trees.circle_from_four_corners(corners, ())
end

#=
## child_indices_extents Implementation
=#

function STI.child_indices_extents(node::HealpixTreeNode{O}) where O
    if !STI.isleaf(node)
        error("child_indices_extents only valid for leaf nodes")
    end
    res = Healpix.Resolution(node.nside_max)
    nested_pix = node.pixel
    # Convert to output ordering
    idx = O == Healpix.NestedOrder ? nested_pix + 1 : Healpix.nest2ring(res, nested_pix + 1)
    extent = STI.node_extent(node)
    return ((idx, extent),)  # Single-element tuple
end

#=
## getcell Implementation
=#

function Trees.getcell(node::HealpixRootNode{O}) where O
    npix = 12 * node.nside_max^2
    return (Trees.getcell(node, i) for i in 1:npix)
end

function Trees.getcell(node::HealpixRootNode{O}, i::Int) where O
    res = Healpix.Resolution(node.nside_max)
    # Convert index based on ordering type
    pix = O == Healpix.NestedOrder ? i - 1 : Healpix.ring2nest(res, i) - 1
    corners = get_pixel_corners_nested(res, pix)
    return GI.Polygon(SA[GI.LinearRing(SA[
        corners[1], corners[2], corners[3], corners[4], corners[1]
    ])])
end

Trees.ncells(node::HealpixRootNode) = 12 * node.nside_max^2

GOCore.best_manifold(::HealpixRootNode) = GO.Spherical()
GOCore.best_manifold(::HealpixTreeNode) = GO.Spherical()

Trees.treeify(::GO.Spherical, node::HealpixRootNode) = node
Trees.treeify(::GO.Spherical, node::HealpixTreeNode) = node

end