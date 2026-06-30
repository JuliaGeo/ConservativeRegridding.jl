#=
OctaHEALPix native quadtree.

Reduced HEALPix grids aren't curvilinear (points-per-ring varies), so the
`CellBasedGrid` path used for full grids can't represent them. Instead we walk
OctaHEALPix's own nested hierarchy: 4 octahedral base faces, each pixel splitting
into 4 children (`child = 4·parent + offset`), leaves at `log2(nside)`. This
mirrors `HealpixExt`'s quadtree but derives every corner from OctaHEALPix's
closed-form geometry — per-pixel and allocation-free, no grid-sized vertex matrix.

Three index spaces meet at the `(j, q, iq)` triple, which everything routes through:

    ij      ring-order linear index   (field data layout; what `getcell` takes)
    pixel   nested linear index       (tree hierarchy; child = 4·pixel + offset)
    j,q,iq  ring (N→S) · quadrant (1..4) · index within the quadrant

`ij → (j,q,iq)` and `pixel → (j,q,iq)` are the converters; geometry is then a pure
function of `(j,q,iq)`. The ring-order conversions are closed-form inverses of the
cumulative ring counts (no grid instance, no allocation); the nested conversion
reuses RingGrids' `nest2rcq`.

Each pixel is a diamond whose corners sit on the latitude "rails" of neighbouring
rings — N corner on ring j-1, E/W on ring j, S on ring j+1. A rail at level
v = 0..2nside (0 = N pole, 2nside = S pole) has latitude from z = 1 - (v/nside)²
(Górski et al. 2005 eq. 4 with 3N²→N²); per-ring longitudes are equidistant.
Together these reproduce `RingGrids.get_vertices` exactly.
=#

# Tree nodes and spatial-tree navigation.

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

treeify(::Spherical, grid::RingGrids.OctaHEALPixGrid) = OctaHEALPixRootNode(grid.nlat_half)

best_manifold(::OctaHEALPixRootNode) = Spherical()
best_manifold(::OctaHEALPixTreeNode) = Spherical()

Trees.ncells(node::OctaHEALPixRootNode) = 4 * node.nside^2

STI.isspatialtree(::Type{<:OctaHEALPixRootNode}) = true
STI.isleaf(::OctaHEALPixRootNode) = false
STI.nchild(::OctaHEALPixRootNode) = 4
STI.node_extent(::OctaHEALPixRootNode) =
    GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint(0.0, 0.0, 1.0), Float64(π) |> nextfloat)

STI.isspatialtree(::Type{<:OctaHEALPixTreeNode}) = true
STI.isleaf(node::OctaHEALPixTreeNode) = node.level == node.leaf_level
STI.nchild(node::OctaHEALPixTreeNode) = STI.isleaf(node) ? 0 : 4

# Base faces are the root's children (level 0, pixels 0..3); deeper, child = 4·parent + offset.
STI.getchild(root::OctaHEALPixRootNode, i::Int) =
    OctaHEALPixTreeNode(root.nside, root.leaf_level, 0, i - 1)
STI.getchild(node::OctaHEALPixTreeNode, i::Int) =
    OctaHEALPixTreeNode(node.nside, node.leaf_level, node.level + 1, 4 * node.pixel + (i - 1))

# Index spaces: ring-order ↔ (j, q, iq) ↔ nested.

# cells per quadrant on ring j (the ring holds 4× this many points)
_octa_cells_per_quadrant(j, nside) = min(j, 2nside - j)

# ring index of ring-order point `ij` (closed-form inverse of the cumulative ring counts)
function _octa_whichring(ij, nside)
    if ij <= 2nside * (nside + 1)              # northern hemisphere incl. equator
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
    i0   = ij - _octa_ring_start(j, nside)             # 0-based within the ring
    nlon = 4 * _octa_cells_per_quadrant(j, nside)
    q    = mod(4i0 ÷ nlon, 4)                           # 0-based quadrant
    iq   = i0 - q * (nlon ÷ 4)                          # 0-based within the quadrant
    return j, q + 1, iq + 1
end

# (ring j, quadrant q, in-quadrant iq) → ring-order index (inverse of `_octa_ring_jqi`)
function _octa_jqi_ring(j, q, iq, nside)
    nlon = 4 * _octa_cells_per_quadrant(j, nside)
    i = iq + (q - 1) * (nlon ÷ 4)                       # 1-based index within the ring
    return _octa_ring_start(j, nside) + i - 1
end

# nested pixel (0-based) at resolution `nside` → (ring j, quadrant q, in-quadrant iq), 1-based
function _octa_nest_jqi(pixel0, nside)
    r, c, q = RingGrids.nest2rcq(pixel0 + 1, nside)    # nested → matrix (row, col, quadrant)
    return r + c - 1, q, min(nside - r, c - 1) + 1
end

# Closed-form cell geometry: (j, q, iq) → corners as (lon°, lat°).

# latitude (°) of corner-rail level v = 0..2nside; ring centres share these rails.
# z = 1 - (v/nside)², lat = 90° - acos(z)  (Górski et al. 2005 eq. 4 with 3N²→N²)
function _octa_lat(v, nside)
    v == 0      && return 90.0
    v == 2nside && return -90.0
    return v <= nside ? 90 - acosd(1 - (v / nside)^2) :
                      -(90 - acosd(1 - ((2nside - v) / nside)^2))
end

# Corners (named E/S/W/N) of a cell on a northern or equatorial ring j ≤ nside.
# E/W sit on ring j's rail, N on ring j-1's, S on ring j+1's.
function _octa_north_corners(nside, j, q, iq)
    mj   = _octa_cells_per_quadrant(j, nside)
    clon = 90 * (q - 1) + (2iq - 1) * 45 / mj           # cell-centre longitude
    latj = _octa_lat(j, nside)
    E = (mod(clon + 45 / mj, 360.0), latj)              # centre ± half a cell in longitude
    W = (mod(clon - 45 / mj, 360.0), latj)
    N = j == 1 ? (clon, 90.0) :
        (mod(90 * (q - 1) + (iq - 1) * 90 / _octa_cells_per_quadrant(j - 1, nside), 360.0),
         _octa_lat(j - 1, nside))
    # S sits on ring j+1: equator-ward (more cells, same iq) except across the
    # equator, where j+1 is pole-ward (fewer cells, iq-1). The final ring's S
    # corner is the south pole (only at nside == 1).
    if (j + 1) == 2nside
        S = (clon, -90.0)
    else
        kS = (j + 1) <= nside ? iq : iq - 1
        S = (mod(90 * (q - 1) + kS * 90 / _octa_cells_per_quadrant(j + 1, nside), 360.0),
             _octa_lat(j + 1, nside))
    end
    return (; E, S, W, N)
end

# Corners for any ring j. The south hemisphere is the matching north ring mirrored
# across the equator: longitudes kept, latitudes negated, N/S corners swapped.
function _octa_corners(nside, j, q, iq)
    j <= nside && return _octa_north_corners(nside, j, q, iq)
    n = _octa_north_corners(nside, 2nside - j, q, iq)
    mirror((lon, lat)) = (lon, -lat)
    return (E = mirror(n.E), S = mirror(n.N), W = mirror(n.W), N = mirror(n.S))
end

# Bounding `SphericalCap` of cell (j,q,iq). Corners go to `circle_from_four_corners`
# in (N,E,W,S) order so its great-circle edge midpoints land on the real diamond edges.
function _octa_cap(nside, j, q, iq)
    c = _octa_corners(nside, j, q, iq)
    return Trees.circle_from_four_corners((c.N, c.E, c.W, c.S), ())
end

# Spatial-tree interface (leaf-level cell access).

# `getcell` is ring-indexed to match the field data layout. Corners are emitted
# counter-clockwise (E, N, W, S) — the convex-clipping intersection kernel needs
# CCW winding; a clockwise ring clips to empty.
function Trees.getcell(node::OctaHEALPixRootNode, ij::Int)
    j, q, iq = _octa_ring_jqi(ij, node.nside)
    c = _octa_corners(node.nside, j, q, iq)
    f = GO.UnitSphereFromGeographic()
    return GI.Polygon(SA[GI.LinearRing(SA[f(c.E), f(c.N), f(c.W), f(c.S), f(c.E)])])
end
Trees.getcell(node::OctaHEALPixRootNode) = (Trees.getcell(node, i) for i in 1:Trees.ncells(node))

# A node's extent is the bounding cap of its (coarse) pixel at resolution 2^level.
function STI.node_extent(node::OctaHEALPixTreeNode)
    nside_c = 1 << node.level                           # 2^level
    j, q, iq = _octa_nest_jqi(node.pixel, nside_c)
    return _octa_cap(nside_c, j, q, iq)
end

# A leaf wraps one grid cell: report its ring-order (data) index and bounding cap.
function STI.child_indices_extents(node::OctaHEALPixTreeNode)
    STI.isleaf(node) || error("child_indices_extents is only valid for leaf nodes")
    j, q, iq = _octa_nest_jqi(node.pixel, node.nside)
    ring_idx = _octa_jqi_ring(j, q, iq, node.nside)
    return ((ring_idx, _octa_cap(node.nside, j, q, iq)),)
end
