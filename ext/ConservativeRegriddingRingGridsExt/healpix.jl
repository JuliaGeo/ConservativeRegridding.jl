#=
## Standard HEALPix native quadtree

The 12-face `HEALPixGrid` (Górski et al. 2005, DOI:10.1086/427976). RingGrids
exposes it only through ring-order indexing and analytic coordinates — it has no
nest machinery (unlike `OctaHEALPixGrid`). So instead of a Morton hierarchy we
treat each face as its own `nside × nside` curvilinear grid (`HEALPixFaceGrid`)
and wrap it in a stock `TopDownQuadtreeCursor`; `HEALPixRootNode` ties the 12
faces together as the full sphere. Range subdivision needs no power-of-two
restriction (the nested *index* is what needs it), so one code path covers any
`nlat_half`.

`nside == nlat_half ÷ 2`. Pixel geometry and the `(face, x, y) ↔ ring`
conversions are the standard HEALPix closed forms (ported here so the extension
carries no Healpix.jl dependency); they reproduce Healpix.jl exactly for
power-of-two `nside` and RingGrids' own coordinates otherwise.
=#

# Górski face constants, indexed by 0-based face number.
const _HP_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
const _HP_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)

# Continuous face coordinates (x, y) ∈ [0, 1] on a 0-based face → unit-sphere
# point. This is HEALPix's `xyf2loc`; the `have_sintheta` branch keeps polar
# corners accurate where `1 - z²` would cancel.
function _hp_xyf2point(x, y, face)
    jr = _HP_JRLL[face + 1] - x - y
    sintheta = 0.0; have_st = false
    if jr < 1                                   # north cap
        nr = jr; tmp = nr * nr / 3; z = 1 - tmp
        if z > 0.99; sintheta = sqrt(tmp * (2 - tmp)); have_st = true; end
    elseif jr > 3                               # south cap
        nr = 4 - jr; tmp = nr * nr / 3; z = tmp - 1
        if z < -0.99; sintheta = sqrt(tmp * (2 - tmp)); have_st = true; end
    else                                        # equatorial belt
        nr = 1.0; z = (2 - jr) * (2 / 3)
    end
    t = _HP_JPLL[face + 1] * nr + x - y
    t < 0 && (t += 8); t >= 8 && (t -= 8)
    phi = nr < 1e-15 ? 0.0 : (Float64(π) / 4 * t) / nr
    st = have_st ? sintheta : sqrt((1 - z) * (1 + z))
    return GO.UnitSphericalPoint(st * cos(phi), st * sin(phi), z)
end

# Cell polygon for pixel (ix, iy) on `face` at resolution `nside`. Corners are
# the four `(x, y)` lattice points emitted CCW (the convex-clip kernel needs CCW;
# a CW ring clips to empty). Shared lattice corners ⇒ exact tessellation.
function _hp_pixel_polygon(ix, iy, face, nside)
    n = nside
    p1 = _hp_xyf2point((ix + 1) / n, (iy + 1) / n, face)
    p2 = _hp_xyf2point( ix      / n, (iy + 1) / n, face)
    p3 = _hp_xyf2point( ix      / n,  iy      / n, face)
    p4 = _hp_xyf2point((ix + 1) / n,  iy      / n, face)
    return GI.Polygon(SA[GI.LinearRing(SA[p1, p2, p3, p4, p1])])
end

#=
### Ring-order index ↔ (face, x, y)

Closed forms over the ring layout (`nr = nlon ÷ 4`, `jr` the global ring index
north→south). `_hp_xyf2ring` maps a leaf pixel to its ring data index;
`_hp_pix2xyf` is the inverse used by `getcell`.
=#

_hp_ring_nlon(jr, nside) = jr < nside ? 4jr : (jr <= 3nside ? 4nside : 4 * (4nside - jr))

function _hp_ring_first(jr, nside)              # 1-based first pixel of ring jr
    ncap = 2nside * (nside - 1)
    jr < nside   ? 2jr * (jr - 1) + 1 :
    jr <= 3nside ? ncap + (jr - nside) * 4nside + 1 :
                   (js = 4nside - jr; 12nside^2 - 2js * (js + 1) + 1)
end

function _hp_xyf2ring(ix, iy, face, nside)
    jr = _HP_JRLL[face + 1] * nside - ix - iy - 1
    nr = _hp_ring_nlon(jr, nside) >> 2
    kshift = (jr < nside || jr > 3nside) ? 0 : ((jr + nside) & 1)
    jp = (_HP_JPLL[face + 1] * nr + ix - iy + 1 + kshift) ÷ 2
    jp < 1 && (jp += 4nside)
    return _hp_ring_first(jr, nside) - 1 + jp
end

function _hp_pix2xyf(ipix, nside)
    ncap = 2nside * (nside - 1); npix = 12nside^2; n2 = 2nside
    if ipix <= ncap                             # north cap
        jr = (1 + isqrt(2ipix - 1)) >> 1
        iphi = ipix - 2jr * (jr - 1); kshift = 0; nr = jr; iring = jr
        face = (iphi - 1) ÷ nr
    elseif ipix <= npix - ncap                  # equatorial belt
        ip = ipix - 1 - ncap; tmp = ip ÷ (4nside)
        iring = tmp + nside; iphi = ip - tmp * 4nside + 1
        kshift = (iring + nside) & 1; nr = nside
        ire = iring - nside + 1; irm = n2 + 2 - ire
        ifm = (iphi - ire ÷ 2 + nside - 1) ÷ nside
        ifp = (iphi - irm ÷ 2 + nside - 1) ÷ nside
        face = ifp == ifm ? (ifp | 4) : (ifp < ifm ? ifp : ifm + 8)
    else                                        # south cap
        ip = npix - ipix + 1; jr2 = (1 + isqrt(2ip - 1)) >> 1
        iphi = 4jr2 + 1 - (ip - 2jr2 * (jr2 - 1)); kshift = 0; nr = jr2
        iring = 4nside - jr2; face = 8 + (iphi - 1) ÷ nr
    end
    irt = iring - _HP_JRLL[face + 1] * nside + 1
    ipt = 2iphi - _HP_JPLL[face + 1] * nr - kshift - 1
    ipt >= n2 && (ipt -= 8nside)
    return ((ipt - irt) >> 1, (-ipt - irt) >> 1, face)
end

#=
### Per-face curvilinear grid

A single HEALPix face is an `nside × nside` block of pixels on the `(x, y)`
lattice — exactly an `AbstractCurvilinearGrid`, so it plugs into the stock
`TopDownQuadtreeCursor` for free. Cell/vertex geometry is analytic (no stored
matrix), and the index maps are overridden so the cursor emits *global ring*
indices: `cartesian_to_linear_idx` returns the ring data index of a pixel and
`linear_to_cartesian_idx` inverts it.
=#

struct HEALPixFaceGrid{M <: Manifold} <: Trees.AbstractCurvilinearGrid{M}
    manifold::M
    nside::Int
    face::Int                                   # 0-based
end

manifold(g::HEALPixFaceGrid) = g.manifold
Trees.ncells(g::HEALPixFaceGrid, ::Int) = g.nside

# Cell (1-based i, j) → pixel (i-1, j-1); corners come out CCW (see _hp_pixel_polygon).
Trees.getcell(g::HEALPixFaceGrid, i::Int, j::Int) = _hp_pixel_polygon(i - 1, j - 1, g.face, g.nside)

# Lattice vertex (1-based point index 1:(nside+1)) → continuous coord (i-1)/nside.
# Drives the per-cell leaf caps (`cell_range_extent`) and block cap (`node_extent`).
Trees.getvertex(g::HEALPixFaceGrid, i::Int, j::Int) =
    _hp_xyf2point((i - 1) / g.nside, (j - 1) / g.nside, g.face)

# Index maps target the *global* ring layout (field data order), not face-local.
Trees.cartesian_to_linear_idx(g::HEALPixFaceGrid, idx::CartesianIndex{2}) =
    _hp_xyf2ring(idx[1] - 1, idx[2] - 1, g.face, g.nside)
function Trees.linear_to_cartesian_idx(g::HEALPixFaceGrid, idx::Integer)
    ix, iy, _ = _hp_pix2xyf(idx, g.nside)
    return CartesianIndex(ix + 1, iy + 1)
end

# A HEALPix pixel never bulges outside the cap of its 4 corners + great-circle edge
# midpoints, so we can skip the generic method's inclusion of all perimeter vertices.
function STI.node_extent(q::Trees.TopDownQuadtreeCursor{<: HEALPixFaceGrid})
    g = q.grid
    imin, imax = extrema(q.leafranges[1]); imax += 1
    jmin, jmax = extrema(q.leafranges[2]); jmax += 1
    bl = Trees.getvertex(g, imin, jmin)
    tl = Trees.getvertex(g, imin, jmax)
    br = Trees.getvertex(g, imax, jmin)
    tr = Trees.getvertex(g, imax, jmax)
    return Trees.circle_from_four_corners((bl, tl, br, tr), ())
end

#=
### Toplevel tree
=#

"""
    HEALPixRootNode(manifold, nside)

Entry point for a standard HEALPix spatial tree from SpeedyWeather/RingGrids.
This has the full sphere with all 12 base faces, and decomposes into
[`Trees.TopDownQuadtreeCursor`](@ref)s with specialized `HEALPixFaceGrid`
inner curvilinear grids.
"""
struct HEALPixRootNode{M <: Manifold}
    manifold::M
    nside::Int
end

# HEALPix is inherently spherical; default the manifold for REPL/test construction.
HEALPixRootNode(nside::Integer) = HEALPixRootNode(Spherical(), nside)

treeify(m::Spherical, grid::RingGrids.HEALPixGrid) = HEALPixRootNode(m, grid.nlat_half ÷ 2)
treeify(::Spherical, node::HEALPixRootNode) = node

best_manifold(node::HEALPixRootNode) = node.manifold
Trees.ncells(node::HEALPixRootNode) = 12 * node.nside^2

STI.isspatialtree(::Type{<: HEALPixRootNode}) = true
STI.isleaf(::HEALPixRootNode) = false
STI.nchild(::HEALPixRootNode) = 12
STI.getchild(root::HEALPixRootNode, i::Int) =
    Trees.TopDownQuadtreeCursor(HEALPixFaceGrid(root.manifold, root.nside, i - 1))
STI.node_extent(::HEALPixRootNode) =
    GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint(0.0, 0.0, 1.0), Float64(π) |> nextfloat)

# `getcell` is ring-indexed to match the field data layout.
function Trees.getcell(node::HEALPixRootNode, ipix::Int)
    ix, iy, face = _hp_pix2xyf(ipix, node.nside)
    return _hp_pixel_polygon(ix, iy, face, node.nside)
end
Trees.getcell(node::HEALPixRootNode) = (Trees.getcell(node, i) for i in 1:Trees.ncells(node))
