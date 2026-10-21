#=
## GridInterface providers

RingGrids grids as GridInterface grids, over the closed forms of this extension
(the same vertex arithmetic as the native trees, so the weights agree bit for
bit). Positions are ring (data) order throughout.

- `FullRingGrid`: one lattice face (lon × lat, south to north) with axis hooks.
- `HEALPixRingGrid`: the 12 base faces (`healpix.jl`'s forms).
- `OctaHEALPixRingGrid`: the 4 octahedral quadrants (`octahealpix.jl`'s forms).
- `ReducedRingGrid`: any other (reduced) grid as stored polygons.
=#

# --- full grids: one face -------------------------------------------------------------
"""
    FullRingGrid(grid::RingGrids.AbstractFullGrid)

One lattice face on `Spherical()`: lattice `i` runs east along a ring and `j`
south to north, so position `i + (nlat - j) nlon` is the ring-order data
index (`facecolumnmajor` is `false`). Vertices are the native tree's: cell
centres at `get_lond`/`get_latd`, longitude edges a half cell west of the
centres, latitude edges at the midpoints of neighbouring rings and pinned to
±90° at the poles (O(nlon + nlat) stored). Longitude is periodic, both pole
rows collapse; point location is the axis floor.
"""
struct FullRingGrid <: AbstractStructuredGrid
    nlon::Int
    nlat::Int
    lon_edges::Vector{Float64}       # nlon + 1, ascending
    lat_edges::Vector{Float64}       # nlat + 1, north to south, pole-pinned
end
function FullRingGrid(grid::RingGrids.AbstractFullGrid)
    latd = RingGrids.get_latd(grid); lond = RingGrids.get_lond(grid)
    nlat = length(latd); nlon = length(lond)
    lat_edges = Vector{Float64}(undef, nlat + 1)
    lat_edges[1] = 90.0; lat_edges[end] = -90.0
    @inbounds for j in 1:nlat-1
        lat_edges[j+1] = 0.5 * (latd[j] + latd[j+1])
    end
    Δlon = 360 / nlon
    lon_edges = [lond[1] - Δlon / 2 + (i - 1) * Δlon for i in 1:nlon+1]
    return FullRingGrid(nlon, nlat, lon_edges, lat_edges)
end
best_manifold(::FullRingGrid) = Spherical()
nfaces(::FullRingGrid) = 1
facesize(g::FullRingGrid, f) = (g.nlon, g.nlat)
@inline facevertex(g::FullRingGrid, f, a, b) = @inbounds GO.UnitSphereFromGeographic()((g.lon_edges[a], g.lat_edges[g.nlat+2-b]))
@inline faceposition(g::FullRingGrid, f, i, j) = i + (g.nlat - j) * g.nlon
@inline facecell(g::FullRingGrid, k) = (ring = (k - 1) ÷ g.nlon + 1; (1, (k - 1) % g.nlon + 1, g.nlat + 1 - ring))
const _FULL_SEAMS = ((true, false), (true, true))
@inline facecross(g::FullRingGrid, f, i, j, s) = Structured._seamcross(_FULL_SEAMS[1], _FULL_SEAMS[2], g.nlon, g.nlat, i, j, s)
facefan(g::FullRingGrid, f, a, b) = Structured._facefan(g, _FULL_SEAMS[1], _FULL_SEAMS[2], f, a, b)
GrI.maxneighbors(::FullRingGrid, ::GrI.Vertex) = nothing
facecolumnmajor(::FullRingGrid, f) = false
hasfaceaxes(::FullRingGrid, f) = true
faceaxes(g::FullRingGrid, f) = (g.lon_edges, view(g.lat_edges, g.nlat+1:-1:1))
facecoords(::FullRingGrid, f, p) = GO.GeographicFromUnitSphere()(p)
faceperiod(::FullRingGrid, f) = (360.0, nothing)
GrI.isglobal(::FullRingGrid) = true

# --- HEALPix: 12 faces ----------------------------------------------------------------
"""
    HEALPixRingGrid(grid::RingGrids.HEALPixGrid)

The 12 base faces of a HEALPix grid (`nside = nlat_half ÷ 2`) on `Spherical()`,
positions in ring order. The face lattice is turned by 180° so a cell's lattice
ring is `_hp_pixel_polygon`'s corner order, (x+1, y+1), (x, y+1), (x, y),
(x+1, y): lattice vertex `(a, b)` of face `f` is the pixel-coordinate point
`((nside+1-a)/nside, (nside+1-b)/nside)`. Pixel edges are not geodesics
(`CurvedEdges`); the face seams come from the corner vertices.
"""
struct HEALPixRingGrid <: AbstractStructuredGrid
    nside::Int
    seams::FaceSeams
end
function HEALPixRingGrid(nside::Integer)
    g = HEALPixRingGrid(Int(nside), FaceSeams(Matrix{Tuple{Int,Int}}(undef, 0, 0)))
    return HEALPixRingGrid(Int(nside), FaceSeams(g))
end
HEALPixRingGrid(grid::RingGrids.HEALPixGrid) = HEALPixRingGrid(grid.nlat_half ÷ 2)
best_manifold(::HEALPixRingGrid) = Spherical()
nfaces(::HEALPixRingGrid) = 12
facesize(g::HEALPixRingGrid, f) = (g.nside, g.nside)
@inline facevertex(g::HEALPixRingGrid, f, a, b) = (n = g.nside; _hp_xyf2point((n + 1 - a) / n, (n + 1 - b) / n, f - 1))
@inline faceposition(g::HEALPixRingGrid, f, i, j) = (n = g.nside; _hp_xyf2ring(n - i, n - j, f - 1, n))
@inline facecell(g::HEALPixRingGrid, k) = ((ix, iy, f) = _hp_pix2xyf(k, g.nside); n = g.nside; (f + 1, n - ix, n - iy))
@inline facecross(g::HEALPixRingGrid, f, i, j, s) = seamcross(g.seams, g, f, i, j, s)
facecolumnmajor(::HEALPixRingGrid, f) = false
GrI.isglobal(::HEALPixRingGrid) = true
GrI.edgekind(::HEALPixRingGrid) = GrI.CurvedEdges()
GrI.cellshape(::HEALPixRingGrid) = GrI.SimpleCells()

# --- OctaHEALPix: 4 quadrant faces ----------------------------------------------------
"""
    OctaHEALPixRingGrid(grid::RingGrids.OctaHEALPixGrid)

The 4 octahedral quadrants of an OctaHEALPix grid (`nside = nlat_half`) on
`Spherical()`, positions in ring order. Quadrant `q`'s `(r, c)` diamond matrix
(`octahealpix.jl`) is turned so that a cell's lattice ring is the native
tree's corner order E, N, W, S: lattice cell `(i, j)` is matrix cell
`(r, c) = (j, nside + 1 - i)`, lattice vertex `(a, b)` the rail vertex
`(b, nside + 2 - a)`. Cell rings are the native tree's closed-form corners
(`_octa_corners`, the same values as `RingGrids.get_vertices`), the lattice
vertices (`_octa_vertex_lonlat`, the same points to rounding) drive extents
and seams.
"""
struct OctaHEALPixRingGrid <: AbstractStructuredGrid
    nside::Int
    seams::FaceSeams
end
function OctaHEALPixRingGrid(nside::Integer)
    g = OctaHEALPixRingGrid(Int(nside), FaceSeams(Matrix{Tuple{Int,Int}}(undef, 0, 0)))
    return OctaHEALPixRingGrid(Int(nside), FaceSeams(g))
end
OctaHEALPixRingGrid(grid::RingGrids.OctaHEALPixGrid) = OctaHEALPixRingGrid(grid.nlat_half)
best_manifold(::OctaHEALPixRingGrid) = Spherical()
nfaces(::OctaHEALPixRingGrid) = 4
facesize(g::OctaHEALPixRingGrid, f) = (g.nside, g.nside)
@inline facevertex(g::OctaHEALPixRingGrid, q, a, b) =
    GO.UnitSphereFromGeographic()(_octa_vertex_lonlat(q, b, g.nside + 2 - a, g.nside))
@inline function faceposition(g::OctaHEALPixRingGrid, q, i, j)
    jr, iq = _octa_rc_jqi(j, g.nside + 1 - i, g.nside)
    return _octa_jqi_ring(jr, q, iq, g.nside)
end
@inline function facecell(g::OctaHEALPixRingGrid, k)
    jr, q, iq = _octa_ring_jqi(k, g.nside)
    r, c = _octa_jqi_rc(jr, iq, g.nside)
    return (q, g.nside + 1 - c, r)
end
@inline facecross(g::OctaHEALPixRingGrid, f, i, j, s) = seamcross(g.seams, g, f, i, j, s)
facecolumnmajor(::OctaHEALPixRingGrid, f) = false
GrI.isglobal(::OctaHEALPixRingGrid) = true
GrI.cellshape(::OctaHEALPixRingGrid) = GrI.SimpleCells()
function _octa_ring(g::OctaHEALPixRingGrid, k::Integer)
    jr, q, iq = _octa_ring_jqi(Int(k), g.nside)
    cr = _octa_corners(g.nside, jr, q, iq)
    f = GO.UnitSphereFromGeographic()
    return Structured.Corners((f(cr.E), f(cr.N), f(cr.W), f(cr.S)))
end
GrI.cell_boundary(g::OctaHEALPixRingGrid, k::Integer) = _octa_ring(g, k)
GrI.cell_corners(g::OctaHEALPixRingGrid, k::Integer) = _octa_ring(g, k)

# --- reduced grids: stored polygons ---------------------------------------------------
"""
    ReducedRingGrid(grid::RingGrids.AbstractGrid)

Any RingGrids grid without a face structure (the reduced octahedral Gaussian
and Clenshaw grids) as stored cells: RingGrids' own `get_vertices` is eager,
so the E, N, W, S corners of every cell are kept (4 points per cell, as the
native fallback tree keeps its polygons), in ring order. The tree is the
generic R-tree.
"""
struct ReducedRingGrid <: GrI.AbstractGrid
    rings::Vector{SVector{4,UnitSphericalPoint{Float64}}}     # E, N, W, S (CCW)
end
function ReducedRingGrid(grid::RingGrids.AbstractGrid)
    E, S, W, N = RingGrids.get_vertices(typeof(grid), grid.nlat_half)
    f = GO.UnitSphereFromGeographic()
    rings = [SVector(f((E[1, k], E[2, k])), f((N[1, k], N[2, k])), f((W[1, k], W[2, k])), f((S[1, k], S[2, k])))
             for k in axes(E, 2)]
    return ReducedRingGrid(rings)
end
GrI.ncells(g::ReducedRingGrid) = length(g.rings)
best_manifold(::ReducedRingGrid) = Spherical()
GrI.cellshape(::ReducedRingGrid) = GrI.SimpleCells()
GrI.cell_boundary(g::ReducedRingGrid, k::Integer) = g.rings[k]
GrI.isglobal(::ReducedRingGrid) = true
GrI.cell_geometry_is_expensive(::ReducedRingGrid) = false

"""
    ringgrid(grid::RingGrids.AbstractGrid) -> GridInterface grid

The provider for a RingGrids grid or field: [`FullRingGrid`](@ref),
[`HEALPixRingGrid`](@ref), [`OctaHEALPixRingGrid`](@ref) or [`ReducedRingGrid`](@ref).
"""
ringgrid(grid::RingGrids.AbstractFullGrid) = FullRingGrid(grid)
ringgrid(grid::RingGrids.HEALPixGrid) = HEALPixRingGrid(grid)
ringgrid(grid::RingGrids.OctaHEALPixGrid) = OctaHEALPixRingGrid(grid)
ringgrid(grid::RingGrids.AbstractGrid) = ReducedRingGrid(grid)
ringgrid(field::RingGrids.AbstractField) = ringgrid(field.grid)
