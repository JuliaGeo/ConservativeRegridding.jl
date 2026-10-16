module ConservativeRegriddingHealpixExt


using ConservativeRegridding
using ConservativeRegridding.Trees

import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI, UnitSpherical
using StaticArrays: SA
using SmallCollections: FixedVector
using LinearAlgebra: normalize

# Import Healpix as a regular Julia package
import Healpix

import GridInterface as GrI
using GridInterface: UnitSphericalPoint
using GridInterface.Structured
import GridInterface.Structured: nfaces, facesize, facevertex, faceposition, facecell, facecross, facecolumnmajor,
    FaceSeams, seamcross

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
    return FixedVector(((idx, extent),))
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
Trees.cell_index_count(node::HealpixTreeNode) = 12 * node.nside_max^2

# Frontier task sizing: exact O(1) cell counts.
Trees.split_weight(node::HealpixRootNode) = 12 * node.nside_max^2
Trees.split_weight(node::HealpixTreeNode) = (node.nside_max >> node.level)^2

GOCore.best_manifold(::HealpixRootNode) = GO.Spherical()
GOCore.best_manifold(::HealpixTreeNode) = GO.Spherical()

Trees.treeify(::GO.Spherical, node::HealpixRootNode) = node
Trees.treeify(::GO.Spherical, node::HealpixTreeNode) = node

#=
## `regrid!` pipeline

A `HealpixMap` is a thin wrapper around an AbstractVector `m.pixels`.
Forwarding the call to that backing array, accessible through `parent(map)`,
gets us automatic denseness detection, and whatever dispatches are defined for
exotic arrays (GPU, distributed, sparse, etc) will also work if those back a Healpix.jl map.
=#

function ConservativeRegridding.extract_source_arraylike(src::Healpix.HealpixMap, regridder; kwargs...)
    return ConservativeRegridding.extract_source_arraylike(parent(src), regridder; kwargs...)
end

function ConservativeRegridding.extract_dest_arraylike(dst::Healpix.HealpixMap, regridder; kwargs...)
    return ConservativeRegridding.extract_dest_arraylike(parent(dst), regridder; kwargs...)
end

function ConservativeRegridding.initialize_regridding!(regridder, src::Healpix.HealpixMap, src_arraylike::AbstractVector; kwargs...)
    return ConservativeRegridding.initialize_regridding!(regridder, parent(src), src_arraylike; kwargs...)
end

Base.@constprop :aggressive function ConservativeRegridding.finalize_regridding!(
    dst::Healpix.HealpixMap, regridder, dst_arraylike::AbstractVector; kwargs...
)
    return ConservativeRegridding.finalize_regridding!(parent(dst), regridder, dst_arraylike; kwargs...)
end


#####
##### GridInterface provider: a `Healpix.HealpixMap` as the 12 base faces of
##### nside × nside pixels. Lattice vertex (a, b) of face f is Healpix.jl's
##### `xyf2loc`/`loc2vec` at the face coordinates ((n+1-a)/n, (n+1-b)/n): the
##### face lattice is turned by 180° so that the lattice ring of a cell is
##### exactly the corner order `boundariesRing` (hence `get_pixel_corners_nested`)
##### emits, (x+1, y+1), (x, y+1), (x, y), (x+1, y); for a power-of-two nside the
##### values are bit-identical to `boundariesRing`'s `(xc ± dc, yc ± dc)`.
##### Positions are the map's own pixel numbers (nested or ring, 1-based); the
##### tree is the derived face quadtree (`HealpixRootNode` stays the native tree).
#####

"""
    HealpixMapGrid(map::Healpix.HealpixMap)
    HealpixMapGrid{O}(nside)

The pixels of a HEALPix map of order `O` (`Healpix.NestedOrder` or
`Healpix.RingOrder`) as a 12-face structured grid on `Spherical()`; cell `k`
is pixel `k` of the map's order. Pixel corners are `boundariesRing`'s
(curved edges, `CurvedEdges`), positions through `xyf2pixNest`/`xyf2pixRing`.
"""
struct HealpixMapGrid{O<:Healpix.Order} <: AbstractStructuredGrid
    res::Healpix.Resolution
    seams::FaceSeams
end
function HealpixMapGrid{O}(res::Healpix.Resolution) where {O<:Healpix.Order}
    g = HealpixMapGrid{O}(res, FaceSeams(Matrix{Tuple{Int,Int}}(undef, 0, 0)))
    return HealpixMapGrid{O}(res, FaceSeams(g))
end
HealpixMapGrid{O}(nside::Integer) where {O<:Healpix.Order} = HealpixMapGrid{O}(Healpix.Resolution(nside))
HealpixMapGrid(map::Healpix.HealpixMap{T,O}) where {T,O} = HealpixMapGrid{O}(map.resolution)
GOCore.best_manifold(::HealpixMapGrid) = GO.Spherical()
@inline function _hploc(x, y, face)
    z, phi, st, has = Healpix.xyf2loc(x, y, face)
    v = Healpix.loc2vec(z, phi, st, has, Float64)
    return UnitSphericalPoint(v[1], v[2], v[3])
end
nfaces(::HealpixMapGrid) = 12
facesize(g::HealpixMapGrid, f) = (g.res.nside, g.res.nside)
@inline facevertex(g::HealpixMapGrid, f, a, b) = (n = g.res.nside; _hploc((n + 1 - a) / n, (n + 1 - b) / n, f - 1))
@inline faceposition(g::HealpixMapGrid{Healpix.NestedOrder}, f, i, j) = (n = g.res.nside; Healpix.xyf2pixNest(g.res, n - i, n - j, f - 1))
@inline faceposition(g::HealpixMapGrid{Healpix.RingOrder}, f, i, j) = (n = g.res.nside; Healpix.xyf2pixRing(g.res, n - i, n - j, f - 1))
@inline facecell(g::HealpixMapGrid{Healpix.NestedOrder}, k) = ((x, y, f) = Healpix.pix2xyfNest(g.res, k); n = g.res.nside; (f + 1, n - x, n - y))
@inline facecell(g::HealpixMapGrid{Healpix.RingOrder}, k) = ((x, y, f) = Healpix.pix2xyfRing(g.res, k); n = g.res.nside; (f + 1, n - x, n - y))
@inline facecross(g::HealpixMapGrid, f, i, j, s) = seamcross(g.seams, g, f, i, j, s)
facecolumnmajor(::HealpixMapGrid, f) = false
GrI.isglobal(::HealpixMapGrid) = true
GrI.edgekind(::HealpixMapGrid) = GrI.CurvedEdges()
GrI.cellshape(::HealpixMapGrid) = GrI.SimpleCells()
GrI.location_is_analytic(::HealpixMapGrid) = true
# Healpix.jl's own point location, checked against the geodesic ring (the
# pixel's curved edges and the ring differ near the edges), else the tree.
function GrI.cellat(g::HealpixMapGrid{O}, p) where {O}
    p = GrI.check_point(GO.Spherical(), p)
    θ = acos(clamp(p[3], -1.0, 1.0)); ϕ = mod(atan(p[2], p[1]), 2π)
    k = O === Healpix.NestedOrder ? Healpix.ang2pixNest(g.res, θ, ϕ) : Healpix.ang2pixRing(g.res, θ, ϕ)
    GrI.cell_contains(g, k, p) && return k
    return GrI.fallback_cellat(GO.Spherical(), g, p)
end

end
