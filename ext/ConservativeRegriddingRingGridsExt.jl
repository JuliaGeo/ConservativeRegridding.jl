module ConservativeRegriddingRingGridsExt

import ConservativeRegridding
using ConservativeRegridding: Trees

using RingGrids

import ConservativeRegridding.Trees: treeify
import GeometryOpsCore: best_manifold, Manifold, Spherical
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
import GeoInterface as GI
import StaticArrays: SA


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

#=
## OctaHEALPix native quadtree

Reduced HEALPix grids are not curvilinear (points-per-ring varies), so the
`CellBasedGrid` path above can't represent them. Instead we exploit OctaHEALPix's
native nested hierarchy: 4 base faces, each pixel subdividing into 4 children
(`child = 4·parent + offset`), leaves at `log2(nside)`. This mirrors the
`HealpixExt` quadtree, but computes pixel corners from RingGrids' own closed-form
OctaHEALPix geometry (per-pixel, allocation-free — no grid-sized vertex matrix).
=#

"""
    OctaHEALPixRootNode(nside)

Entry point for an OctaHEALPix spatial tree: the full sphere with 4 octahedral
base-face children. `nside == nlat_half` and must be a power of two (the nested
hierarchy only exists then). `leaf_level` caches `log2(nside)` so the dual-DFS
hot path never recomputes it.
"""
struct OctaHEALPixRootNode
    nside::Int
    leaf_level::Int
end
function OctaHEALPixRootNode(nside::Integer)
    ispow2(nside) || throw(ArgumentError(
        """
        OctaHEALPix conservative regridding requires `nlat_half` to be a power of two;
        got $nside.
        """))
    return OctaHEALPixRootNode(Int(nside), trailing_zeros(Int(nside)))
end

treeify(::Spherical, grid::RingGrids.OctaHEALPixGrid) = OctaHEALPixRootNode(grid.nlat_half)

best_manifold(::OctaHEALPixRootNode) = Spherical()

Trees.ncells(node::OctaHEALPixRootNode) = 4 * node.nside^2

STI.isspatialtree(::Type{<:OctaHEALPixRootNode}) = true
STI.isleaf(::OctaHEALPixRootNode) = false
STI.nchild(::OctaHEALPixRootNode) = 4
STI.node_extent(::OctaHEALPixRootNode) =
    GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint(0.0, 0.0, 1.0), Float64(π) |> nextfloat)

"""
    OctaHEALPixTreeNode(nside, leaf_level, level, pixel)

A pixel at `level` in the OctaHEALPix hierarchy. `pixel` is the 0-based nested
index at this level; its 4 children are `4·pixel + (0:3)`. A node is a leaf once
`level == leaf_level` (`== log2(nside)`).
"""
struct OctaHEALPixTreeNode
    nside::Int
    leaf_level::Int
    level::Int
    pixel::Int
end

best_manifold(::OctaHEALPixTreeNode) = Spherical()

STI.isspatialtree(::Type{<:OctaHEALPixTreeNode}) = true
STI.isleaf(node::OctaHEALPixTreeNode) = node.level == node.leaf_level
STI.nchild(node::OctaHEALPixTreeNode) = STI.isleaf(node) ? 0 : 4

# Base faces live at level 0, nested pixels 0..3.
STI.getchild(root::OctaHEALPixRootNode, i::Int) =
    OctaHEALPixTreeNode(root.nside, root.leaf_level, 0, i - 1)
# Nested subdivision: child pixel = 4·parent + offset.
STI.getchild(node::OctaHEALPixTreeNode, i::Int) =
    OctaHEALPixTreeNode(node.nside, node.leaf_level, node.level + 1, 4 * node.pixel + (i - 1))

#=
### Closed-form cell geometry

A pixel's four corners follow from OctaHEALPix's analytic ring latitudes
(Górski et al. 2005 eq. 4) and equidistant per-ring longitudes — no
`AnvilInterpolator` and no grid-sized vertex matrix. These reproduce
`RingGrids.get_vertices` exactly, per-pixel and allocation-free.
=#

# cells per quadrant on ring v (v = 1..2nside-1)
_octa_m(v, nside) = min(v, 2nside - v)

# latitude (degrees) at vertical corner level v = 0..2nside (0 = N pole, 2nside = S pole)
function _octa_lat(v, nside)
    v == 0      && return 90.0
    v == 2nside && return -90.0
    return v <= nside ? 90 - acosd(1 - (v / nside)^2) :
                      -(90 - acosd(1 - ((2nside - v) / nside)^2))
end

# (E, S, W, N) corners as (lon, lat) for a north/equator cell (ring j ≤ nside)
function _octa_north_corners(nside, j, q, iq)
    mj   = _octa_m(j, nside)
    clon = 90 * (q - 1) + (2iq - 1) * 45 / mj
    latj = _octa_lat(j, nside)
    E = (mod(clon + 45 / mj, 360.0), latj)
    W = (mod(clon - 45 / mj, 360.0), latj)
    N = j == 1 ? (clon, 90.0) :
        (mod(90 * (q - 1) + (iq - 1) * 90 / _octa_m(j - 1, nside), 360.0), _octa_lat(j - 1, nside))
    # S neighbour is equator-ward (more cells, index iq) except on the equator,
    # where ring j+1 is pole-ward (fewer cells, index iq-1). The last ring's S
    # corner is the south pole (only reached for nside == 1, i.e. a base face).
    if (j + 1) == 2nside
        S = (clon, -90.0)
    else
        kS = (j + 1) <= nside ? iq : iq - 1
        S = (mod(90 * (q - 1) + kS * 90 / _octa_m(j + 1, nside), 360.0), _octa_lat(j + 1, nside))
    end
    return E, S, W, N
end

# (E, S, W, N) corners for any ring j; the south hemisphere mirrors the north
# across the equator (longitude kept, latitude negated, N/S swapped).
function _octa_corners_lonlat(nside, j, q, iq)
    j <= nside && return _octa_north_corners(nside, j, q, iq)
    E, S, W, N = _octa_north_corners(nside, 2nside - j, q, iq)
    return (E[1], -E[2]), (N[1], -N[2]), (W[1], -W[2]), (S[1], -S[2])
end

# ring index of a ring-order point (analytic inverse of the cumulative ring counts)
function _octa_whichring(ij, nside)
    if ij <= 2nside * (nside + 1)             # northern hemisphere incl. equator
        return ceil(Int, (-1 + sqrt(1 + 2ij)) / 2)
    else                                       # south: mirror the count from the south pole
        s = ceil(Int, (-1 + sqrt(1 + 2 * (4nside^2 + 1 - ij))) / 2)
        return 2nside - s
    end
end

# first ring-order index of ring j
_octa_ring_start(j, nside) =
    j <= nside ? 2j * (j - 1) + 1 : 4nside^2 - 2(2nside - j) * (2nside - j + 1) + 1

# ring-order index → (ring j, quadrant q, in-quadrant index iq), all 1-based
function _octa_ring_jqi(ij, nside)
    j    = _octa_whichring(ij, nside)
    i0   = ij - _octa_ring_start(j, nside)    # 0-based within the ring
    nlon = min(4j, 8nside - 4j)
    q    = mod((4i0) ÷ nlon, 4)               # 0-based quadrant
    iq   = i0 - q * (nlon ÷ 4)                # 0-based within the quadrant
    return j, q + 1, iq + 1
end

# `getcell` is ring-indexed (matching field data layout). Corners are emitted
# counter-clockwise (E, N, W, S) — the convex-clipping intersection kernel needs
# CCW winding; a clockwise ring clips to empty.
function Trees.getcell(node::OctaHEALPixRootNode, ij::Int)
    j, q, iq = _octa_ring_jqi(ij, node.nside)
    E, S, W, N = _octa_corners_lonlat(node.nside, j, q, iq)
    f = GO.UnitSphereFromGeographic()
    return GI.Polygon(SA[GI.LinearRing(SA[f(E), f(N), f(W), f(S), f(E)])])
end
Trees.getcell(node::OctaHEALPixRootNode) = (Trees.getcell(node, i) for i in 1:Trees.ncells(node))

#=
### Extents

A node's bounding `SphericalCap` is built from its (coarse) pixel corners at
resolution `2^level`. Corners are fed to `circle_from_four_corners` in (N,E,W,S)
order so its great-circle edge midpoints land on the four real diamond edges.
=#

# nested pixel (0-based) at resolution `nside` → (ring j, quadrant q, in-quadrant iq), 1-based
function _octa_nest_jqi(pixel0, nside)
    r, c, q = RingGrids.nest2rcq(pixel0 + 1, nside)   # 1-based nested → matrix (r,c,q)
    return r + c - 1, q, min(nside - r, c - 1) + 1
end

# bounding cap of the cell (j,q,iq) at resolution `nside`
function _octa_cap(nside, j, q, iq)
    E, S, W, N = _octa_corners_lonlat(nside, j, q, iq)
    return Trees.circle_from_four_corners((N, E, W, S), ())
end

function STI.node_extent(node::OctaHEALPixTreeNode)
    nside_c = 1 << node.level                          # 2^level
    j, q, iq = _octa_nest_jqi(node.pixel, nside_c)
    return _octa_cap(nside_c, j, q, iq)
end

function STI.child_indices_extents(node::OctaHEALPixTreeNode)
    STI.isleaf(node) || error("child_indices_extents is only valid for leaf nodes")
    nside = node.nside
    j, q, iq = _octa_nest_jqi(node.pixel, nside)
    nlon = min(4j, 8nside - 4j)
    i = iq + (q - 1) * (nlon ÷ 4)                      # 1-based index within ring
    ring_idx = _octa_ring_start(j, nside) + i - 1      # nested → ring (data) index
    return ((ring_idx, _octa_cap(nside, j, q, iq)),)
end

end
