module ConservativeRegriddingOceananigansExt

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode, RightFaceFolded, RightCenterFolded
using Oceananigans.Fields: AbstractField
using Oceananigans.Architectures: CPU
using Oceananigans.Utils: launch!, KernelParameters

using KernelAbstractions: @kernel, @index

using ConservativeRegridding
using ConservativeRegridding: Regridder, ExampleFieldFunction

import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
import Oceananigans.Architectures: on_architecture

import Extents
import GridInterface as GrI
using GridInterface: UnitSphericalPoint, SmallVector
using GridInterface.Structured
import GridInterface.Structured: nfaces, facesize, facevertex, faceposition, facecell, facecross, facefan,
    hasfaceaxes, faceaxes, facecoords, faceperiod, FaceSeams, seamcross

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    compute_cell_matrix(field.grid)
end

function compute_cell_matrix(grid::Oceananigans.Grids.AbstractGrid)
    Nx, Ny, _ = size(grid)
    ℓx, ℓy    = Center(), Center()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    arch = grid.architecture
    FT = eltype(grid)

    cell_matrix = Array{Tuple{FT, FT}}(undef, Nx+1, Ny+1)

    # Not GPU compatible so we need to move the grid on the CPU
    cpu_grid = on_architecture(CPU(), grid)
    _compute_cell_matrix!(cell_matrix, Nx+1, Ny+1, ℓx, ℓy, cpu_grid)

    return on_architecture(arch, cell_matrix)
end

flip(::Face) = Center()
flip(::Center) = Face()

function _compute_cell_matrix!(cell_matrix, Fx, Fy, ℓx, ℓy, grid)
    for j in 1:Fy, i in 1:Fx
        vx = flip(ℓx)
        vy = flip(ℓy)

        xl = ξnode(i, j, 1, grid, vx, vy, nothing)
        yl = ηnode(i, j, 1, grid, vx, vy, nothing)

        @inbounds cell_matrix[i, j] = (xl, yl)
    end
end


const RightCenterFoldedGrid = Oceananigans.Grids.ZRegOrthogonalSphericalShellGrid{<: Number, <: Any, Oceananigans.RightCenterFolded}

#####
##### `regrid!` plumbing for Oceananigans fields.
#####
##### A `Field` is a 3D AbstractArray with halos.  `interior(field)` strips the
##### halos but is non-contiguous in memory (row stride includes halo width), so
##### it cannot be a `DenseVector` and `mul!` cannot operate on it directly.
##### We route through the regridder's dense temp buffers via `copyto!`.

ConservativeRegridding.extract_source_arraylike(::Oceananigans.AbstractField, regridder; kwargs...) = regridder.src_temp
ConservativeRegridding.extract_dest_arraylike(::Oceananigans.AbstractField, regridder;   kwargs...) = regridder.dst_temp

function ConservativeRegridding.initialize_regridding!(regridder, field::Oceananigans.AbstractField, src_arraylike::AbstractVector; kwargs...)
    copyto!(src_arraylike, vec(interior(field)))
    return regridder
end

function ConservativeRegridding.finalize_regridding!(field::Oceananigans.AbstractField, regridder, dst_arraylike::AbstractVector; normalize = true, kwargs...)
    if normalize
        dst_arraylike ./= regridder.dst_areas
    end
    mirror_fold_partners!(dst_arraylike, field.grid)
    copyto!(vec(interior(field)), dst_arraylike)
    return field
end

# Partner slots in a folded grid have zero rows in the intersection matrix, so
# after mul!/normalize they hold 0 (or 0/0 = NaN).  Copy each primary's value
# into its partner so the field shows the same physical value on both copies.
# Real (primary) locals along the fold row are 1..Nquarter and Nhalf+1..Nhalf+Nquarter;
# the partner of local r is Nx+1-r.
mirror_fold_partners!(_, _) = nothing

function mirror_fold_partners!(dst, grid::RightCenterFoldedGrid)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    Nq = Nx ÷ 4
    Nh = Nx ÷ 2
    fold_offset = (Ny - 1) * Nx
    arch = grid.architecture
    _mirror_fold_partners!(Oceananigans.Architectures.device(arch), 16, 2Nq)(
        dst, Nx, Nh, Nq, fold_offset
    )
    return dst
end

@kernel function _mirror_fold_partners!(dst, Nx, Nh, Nq, fold_offset)
    i = @index(Global)
    r = ifelse(i ≤ Nq, i, i + Nh - Nq)
    @inbounds dst[Nx + 1 - r + fold_offset] = dst[r + fold_offset]
end

# Also define which manifold the grid lives on.  This gives us accurate area as well for any simulation
# (e.g. on Mars?!)
GOCore.best_manifold(grid::Oceananigans.RectilinearGrid) = GO.Planar()
GOCore.best_manifold(grid::Oceananigans.LatitudeLongitudeGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(grid::Oceananigans.OrthogonalSphericalShellGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(grid::Oceananigans.ImmersedBoundaryGrid) = GOCore.best_manifold(grid.underlying_grid)

GOCore.best_manifold(field::Oceananigans.Field) = GOCore.best_manifold(field.grid)

# Extend the `on_architecture` method for a `Regridder` object
on_architecture(arch, r::Regridder) =
    Regridder(on_architecture(arch, r.intersections),
              on_architecture(arch, r.dst_areas),
              on_architecture(arch, r.src_areas),
              on_architecture(arch, r.dst_temp),
              on_architecture(arch, r.src_temp))

# Allow to set example data on the field
Oceananigans.set!(field::Oceananigans.Field, f::ExampleFieldFunction) = Oceananigans.set!(field, (lon, lat, z) -> f(lon, lat))


#####
##### GridInterface providers: Oceananigans grids as `GridInterface.Structured`
##### faces. Vertices are the values CR's `compute_cell_matrix` reads through
##### `ξnode`/`ηnode` at (Face, Face): the grid's own λᶠᵃᵃ/φᵃᶠᵃ (lat/lon),
##### xᶠᵃᵃ/yᵃᶠᵃ (rectilinear) and λᶠᶠᵃ/φᶠᶠᵃ (curvilinear, tripolar, cubed-sphere
##### panels), read in place: no vertex matrix is built. Positions are the
##### field layout `i + (j - 1) Nx` (panel-major on the cubed sphere).
##### `asgrid` (what `Regridder` regrids) returns them; a U-pivot tripolar
##### grid's field layout goes through `FoldedLayoutGrid` below.
#####

const _USFG = GO.UnitSphereFromGeographic()
@inline _vx(::GOCore.Spherical, x, y) = _USFG((x, y))
@inline _vx(::GOCore.Planar, x, y) = (Float64(x), Float64(y))
# λᶠᵃᵃ-style vectors and λᶠᶠᵃ-style matrices read the same way
@inline _at(x::AbstractVector, a, b, ::Val{1}) = @inbounds x[a]
@inline _at(x::AbstractVector, a, b, ::Val{2}) = @inbounds x[b]
@inline _at(x::AbstractMatrix, a, b, ::Val) = @inbounds x[a, b]
_isperiodic(TX) = TX === Oceananigans.Grids.Periodic

"""
    OceananigansLatLonGrid(grid::LatitudeLongitudeGrid)
    OceananigansLatLonGrid(grid::RectilinearGrid)

One lattice face with separable axes: the grid's own `λᶠᵃᵃ`/`φᵃᶠᵃ` (on
`Spherical(; radius)`) or `xᶠᵃᵃ`/`yᵃᶠᵃ` (a `RectilinearGrid`, planar). Point
location is the axis floor. Longitude is a seam when the grid's x topology is
`Periodic` and it spans 360°; a latitude row at ±90° is collapsed (a pole
fan). A planar rectilinear grid declares no seams: a periodic topology does
not make its first and last vertex columns the same points. `isglobal` when
spherical, periodic and reaching both poles.
"""
struct OceananigansLatLonGrid{X,Y,M} <: AbstractStructuredGrid
    xs::X
    ys::Y
    manifold::M
    Nx::Int
    Ny::Int
    periodic::NTuple{2,Bool}
    collapsed::NTuple{2,Bool}
    # `sincosd` of the vertex longitudes and colatitudes (spherical; empty when
    # planar): a vertex is then three products, the bits `UnitSphereFromGeographic`
    # gives, where per-vertex trig made the engine's cell sweeps ~9x slower.
    xsc::Vector{NTuple{2,Float64}}
    ysc::Vector{NTuple{2,Float64}}
end
function OceananigansLatLonGrid(grid::Oceananigans.LatitudeLongitudeGrid)
    Nx, Ny, _ = size(grid)
    TX, _, _ = Oceananigans.Grids.topology(grid)
    xs, ys = grid.λᶠᵃᵃ, grid.φᵃᶠᵃ
    periodic = (_isperiodic(TX) && abs((xs[Nx+1] - xs[1]) - 360) <= 1e-9 * 360, false)
    collapsed = (ys[1] <= -90, ys[Ny+1] >= 90)
    xsc = [sincosd(Float64(xs[a])) for a in 1:Nx+1]
    ysc = [sincosd(90 - Float64(ys[b])) for b in 1:Ny+1]
    return OceananigansLatLonGrid(xs, ys, GO.Spherical(; radius=grid.radius), Nx, Ny, periodic, collapsed, xsc, ysc)
end
function OceananigansLatLonGrid(grid::Oceananigans.RectilinearGrid)
    Nx, Ny, _ = size(grid)
    return OceananigansLatLonGrid(grid.xᶠᵃᵃ, grid.yᵃᶠᵃ, GO.Planar(), Nx, Ny, (false, false), (false, false),
        NTuple{2,Float64}[], NTuple{2,Float64}[])
end
GOCore.best_manifold(g::OceananigansLatLonGrid) = g.manifold
nfaces(::OceananigansLatLonGrid) = 1
facesize(g::OceananigansLatLonGrid, f) = (g.Nx, g.Ny)
@inline facevertex(g::OceananigansLatLonGrid, f, a, b) = _vx(g.manifold, _at(g.xs, a, b, Val(1)), _at(g.ys, a, b, Val(2)))
@inline function facevertex(g::OceananigansLatLonGrid{<:Any,<:Any,<:GOCore.Spherical}, f, a, b)
    sinθ, cosθ = g.xsc[a]
    sinϕ, cosϕ = g.ysc[b]
    return UnitSphericalPoint(sinϕ * cosθ, sinϕ * sinθ, cosϕ)
end
@inline faceposition(g::OceananigansLatLonGrid, f, i, j) = i + (j - 1) * g.Nx
@inline facecell(g::OceananigansLatLonGrid, k) = (1, (k - 1) % g.Nx + 1, (k - 1) ÷ g.Nx + 1)
@inline facecross(g::OceananigansLatLonGrid, f, i, j, s) = Structured._seamcross(g.periodic, g.collapsed, g.Nx, g.Ny, i, j, s)
facefan(g::OceananigansLatLonGrid, f, a, b) = Structured._facefan(g, g.periodic, g.collapsed, f, a, b)
GrI.maxneighbors(g::OceananigansLatLonGrid, ::GrI.Vertex) = Structured._vertexmaxneighbors(g.collapsed)
hasfaceaxes(::OceananigansLatLonGrid, f) = true
faceaxes(g::OceananigansLatLonGrid, f) = (view(g.xs, 1:g.Nx+1), view(g.ys, 1:g.Ny+1))
facecoords(g::OceananigansLatLonGrid, f, p) = g.manifold isa GOCore.Spherical ? GO.GeographicFromUnitSphere()(p) :
    (Float64(GI.x(p)), Float64(GI.y(p)))
faceperiod(g::OceananigansLatLonGrid, f) = g.manifold isa GOCore.Spherical ? (360.0, nothing) : (nothing, nothing)
GrI.isglobal(g::OceananigansLatLonGrid) = g.manifold isa GOCore.Spherical && g.periodic[1] && g.collapsed[1] && g.collapsed[2]
function Structured._blockextent(::GOCore.Planar, g::OceananigansLatLonGrid, f, ir::UnitRange{Int}, jr::UnitRange{Int})
    return Extents.Extent(X=(Float64(g.xs[first(ir)]), Float64(g.xs[last(ir)+1])), Y=(Float64(g.ys[first(jr)]), Float64(g.ys[last(jr)+1])))
end
Structured._cellextent(m::GOCore.Planar, g::OceananigansLatLonGrid, f, i, j) = Structured._blockextent(m, g, f, i:i, j:j)
# four axis reads: the planar quadtree's node extents are cheap
STI.node_extent_is_expensive(::Type{<:Structured.TopDownQuadtreeCursor{<:OceananigansLatLonGrid{<:Any,<:Any,<:GOCore.Planar}}}) = false

"""
    OceananigansCurvilinearGrid(grid::OrthogonalSphericalShellGrid)

One lattice face over the grid's own `λᶠᶠᵃ`/`φᶠᶠᵃ` on `Spherical(; radius)`:
a `RotatedLatitudeLongitudeGrid` (periodic x, collapsed poles where the
vertex rows coincide) or a `TripolarGrid`. The tripolar fold is read off the
y topology:

- `RightFaceFolded` (F-pivot): vertex `(a, Ny+1)` is vertex `(Nx+2-a, Ny+1)`
  again, so the north side of fold-row cell `(i, Ny)` is a seam onto cell
  `(Nx+1-i, Ny)` (`facecross`); the two pivots are closed two-cell fans.
- `RightCenterFolded` (U-pivot): vertex `(a, Ny+1)` is vertex `(Nx+2-a, Ny)`,
  so fold-row cell `(i, Ny)` is the same quadrilateral as `(Nx+1-i, Ny)`: the
  field layout holds the row twice. GridInterface has no verb for duplicated
  cells, so this full layout declares `tessellation = Overlapping()`, keeps
  the fold out of `facecross` (no consistent lattice crossing exists) and
  answers `neighbors` by hand on the fold row, mapping each duplicate onto
  its primary (CR's halves: `1:Nx÷4` and `Nx÷2+1:Nx÷2+Nx÷4`); [`unique_cells`](@ref)
  is the `Subset` of primaries, which tessellates, and
  [`FoldedLayoutGrid`](@ref) the full layout with the duplicates as ghosts,
  which is what `Regridder` regrids (`asgrid`).
"""
struct OceananigansCurvilinearGrid{A,M} <: AbstractStructuredGrid
    λ::A
    φ::A
    manifold::M
    Nx::Int
    Ny::Int
    periodic_x::Bool
    fold::Symbol            # :none, :face (RightFaceFolded), :center (RightCenterFolded)
    collapsed::NTuple{2,Bool}
end
function OceananigansCurvilinearGrid(grid::Oceananigans.OrthogonalSphericalShellGrid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = Oceananigans.Grids.topology(grid)
    fold = TY === RightCenterFolded ? :center : TY === RightFaceFolded ? :face : :none
    λ, φ = grid.λᶠᶠᵃ, grid.φᶠᶠᵃ
    m = GO.Spherical(; radius=grid.radius)
    # a pole row: every vertex of the row is the same point (a rotated lat/lon grid reaching its poles)
    samerow(b) = all(a -> GrI._near(_USFG((λ[a, b], φ[a, b])), _USFG((λ[1, b], φ[1, b]))), 2:Nx+1)
    collapsed = fold === :none ? (samerow(1), samerow(Ny + 1)) : (false, false)
    return OceananigansCurvilinearGrid(λ, φ, m, Nx, Ny, _isperiodic(TX), fold, collapsed)
end
GOCore.best_manifold(g::OceananigansCurvilinearGrid) = g.manifold
nfaces(::OceananigansCurvilinearGrid) = 1
facesize(g::OceananigansCurvilinearGrid, f) = (g.Nx, g.Ny)
@inline facevertex(g::OceananigansCurvilinearGrid, f, a, b) = @inbounds _USFG((g.λ[a, b], g.φ[a, b]))
@inline faceposition(g::OceananigansCurvilinearGrid, f, i, j) = i + (j - 1) * g.Nx
@inline facecell(g::OceananigansCurvilinearGrid, k) = (1, (k - 1) % g.Nx + 1, (k - 1) ÷ g.Nx + 1)
@inline function facecross(g::OceananigansCurvilinearGrid, f, i, j, s)
    if s == 3 && g.fold === :face
        return (1, g.Nx + 1 - i, g.Ny, 3)                # the F-pivot fold: a seam onto the mirrored cell
    end
    s == 3 && g.fold === :center && return nothing     # the U-pivot fold: no lattice crossing (see the docstring)
    return Structured._seamcross((g.periodic_x, false), g.collapsed, g.Nx, g.Ny, i, j, s)
end
facefan(g::OceananigansCurvilinearGrid, f, a, b) = Structured._facefan(g, (g.periodic_x, false), g.collapsed, f, a, b)
GrI.maxneighbors(g::OceananigansCurvilinearGrid, ::GrI.Vertex) = Structured._vertexmaxneighbors(g.collapsed)
GrI.isglobal(g::OceananigansCurvilinearGrid) = g.fold === :none && g.periodic_x && g.collapsed[1] && g.collapsed[2]
GrI.tessellation(g::OceananigansCurvilinearGrid) = g.fold === :center ? GrI.Overlapping() : GrI.Tessellating()

# --- the U-pivot fold row ---------------------------------------------------
"Whether column `i` of the fold row is a primary (CR's halves) rather than a duplicate."
@inline _isprimary(g::OceananigansCurvilinearGrid, i) = (Nq = g.Nx ÷ 4; Nh = g.Nx ÷ 2; i <= Nq || Nh < i <= Nh + Nq)
@inline _primary(g::OceananigansCurvilinearGrid, i) = (i = mod1(i, g.Nx); _isprimary(g, i) ? i : g.Nx + 1 - i)

"""
    unique_cells(g::OceananigansCurvilinearGrid) -> Subset

The cells of the field layout that are not duplicates: everything but the
duplicated half of a U-pivot fold row (the whole grid otherwise).
"""
function unique_cells(g::OceananigansCurvilinearGrid)
    g.fold === :center || return GrI.subset(g, 1:GrI.ncells(g))
    Nq, Nh = g.Nx ÷ 4, g.Nx ÷ 2
    off = (g.Ny - 1) * g.Nx
    return GrI.subset(g, GrI.RangeCat([1:off+Nq, off+Nh+1:off+Nh+Nq]))     # lazy: O(1), not O(ncells)
end
unique_cells(g::AbstractStructuredGrid) = GrI.subset(g, 1:GrI.ncells(g))

function GrI.neighbors(g::OceananigansCurvilinearGrid, k::Integer; connectivity::GrI.Connectivity=GrI.Vertex())
    g.fold === :center || return Structured._structured_neighbors(g, Int(k), connectivity)
    return _fold_neighbors(g, Int(k), connectivity)
end
# The structured answer with the fold row's duplicates mapped onto their
# primaries, and the fold row itself by hand: physically, the north side of
# fold-row cell (i, Ny) meets cell (Nx+1-i, Ny-1), the cell under its twin.
function _fold_neighbors(g::OceananigansCurvilinearGrid, k::Int, connectivity)
    Nx, Ny = g.Nx, g.Ny
    _, i, j = facecell(g, k)
    pos(ii, jj) = mod1(ii, Nx) + (jj - 1) * Nx
    if j < Ny
        nb = Structured._structured_neighbors(g, k, connectivity)
        out = SmallVector{8,Int}()
        for q in nb
            _, qi, qj = facecell(g, q)
            qj == Ny && (qi = _primary(g, qi))
            out = Structured._add(out, pos(qi, qj))
        end
        return out
    end
    i = _primary(g, i)                 # a duplicate answers as its primary
    k0 = pos(i, Ny)
    p = Nx + 1 - i                     # the twin's column
    out = SmallVector{8,Int}()
    cand = connectivity isa GrI.Edge ?
        ((i, Ny - 1), (_primary(g, i + 1), Ny), (p, Ny - 1), (_primary(g, i - 1), Ny), (0, 0), (0, 0), (0, 0), (0, 0)) :
        ((i, Ny - 1), (i + 1, Ny - 1), (_primary(g, i + 1), Ny), (p - 1, Ny - 1),
         (p, Ny - 1), (p + 1, Ny - 1), (_primary(g, i - 1), Ny), (i - 1, Ny - 1))
    for (ii, jj) in cand
        1 <= jj <= Ny || continue
        q = pos(ii, jj)
        q == k0 || (out = Structured._add(out, q))
    end
    return out
end

"""
    wet_cells(ibg::ImmersedBoundaryGrid; k = size(ibg, 3)) -> Subset

The active cells of an immersed-boundary grid: the unique cells of the
underlying grid's provider that are not immersed at level `k` (the surface
by default), as a `Subset` whose tree is the provider's quadtree with the
masked cells skipped. O(active cells) storage.
"""
function wet_cells(ibg::Oceananigans.ImmersedBoundaryGrid; k::Integer=size(ibg, 3))
    g = structured_grid(ibg.underlying_grid)
    ib = ibg.immersed_boundary
    keep = Int[]
    for p in GrI.parentpositions(unique_cells(g))
        _, i, j = facecell(g, p)
        Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, ibg.underlying_grid, ib) || push!(keep, p)
    end
    return GrI.subset(g, keep)
end

"""
    OceananigansCubedSphereGrid(grid::ConformalCubedSphereGrid)

Six lattice faces, one per panel, over each panel's own `λᶠᶠᵃ`/`φᶠᶠᵃ` on
`Spherical(; radius)`; positions are panel-major, `i + (j-1) Nx + (f-1) Nx Ny`.
The twelve panel seams are found from the panels' corner vertices
([`FaceSeams`](@ref GridInterface.Structured.FaceSeams)) once at construction.
"""
struct OceananigansCubedSphereGrid{A,M} <: AbstractStructuredGrid
    λ::NTuple{6,A}
    φ::NTuple{6,A}
    manifold::M
    Nx::Int
    Ny::Int
    seams::FaceSeams
end
function OceananigansCubedSphereGrid(grid::Oceananigans.MultiRegion.MultiRegionGrid)
    ps = ntuple(k -> Oceananigans.MultiRegion.getregion(grid, k), 6)
    Nx, Ny, _ = size(ps[1])
    g = OceananigansCubedSphereGrid(ntuple(k -> ps[k].λᶠᶠᵃ, 6), ntuple(k -> ps[k].φᶠᶠᵃ, 6),
        GO.Spherical(; radius=ps[1].radius), Nx, Ny, FaceSeams(Matrix{Tuple{Int,Int}}(undef, 0, 0)))
    return OceananigansCubedSphereGrid(g.λ, g.φ, g.manifold, Nx, Ny, FaceSeams(g))
end
GOCore.best_manifold(g::OceananigansCubedSphereGrid) = g.manifold
nfaces(::OceananigansCubedSphereGrid) = 6
facesize(g::OceananigansCubedSphereGrid, f) = (g.Nx, g.Ny)
@inline facevertex(g::OceananigansCubedSphereGrid, f, a, b) = @inbounds _USFG((g.λ[f][a, b], g.φ[f][a, b]))
@inline faceposition(g::OceananigansCubedSphereGrid, f, i, j) = i + (j - 1) * g.Nx + (f - 1) * g.Nx * g.Ny
@inline function facecell(g::OceananigansCubedSphereGrid, k)
    f, r = divrem(k - 1, g.Nx * g.Ny)
    return (f + 1, r % g.Nx + 1, r ÷ g.Nx + 1)
end
@inline facecross(g::OceananigansCubedSphereGrid, f, i, j, s) = seamcross(g.seams, g, f, i, j, s)
GrI.isglobal(::OceananigansCubedSphereGrid) = true

"""
    structured_grid(grid) -> AbstractStructuredGrid

The provider for a native Oceananigans grid, with every cell of its field
layout: [`OceananigansLatLonGrid`](@ref), [`OceananigansCurvilinearGrid`](@ref)
or [`OceananigansCubedSphereGrid`](@ref); an immersed-boundary grid answers
its underlying grid's. The active cells (`wet_cells`, `unique_cells`) are
subsets of it.
"""
structured_grid(grid::Oceananigans.LatitudeLongitudeGrid) = OceananigansLatLonGrid(grid)
structured_grid(grid::Oceananigans.RectilinearGrid) = OceananigansLatLonGrid(grid)
structured_grid(grid::Oceananigans.OrthogonalSphericalShellGrid) = OceananigansCurvilinearGrid(grid)
structured_grid(grid::Oceananigans.MultiRegion.MultiRegionGrid) = OceananigansCubedSphereGrid(grid)
structured_grid(grid::Oceananigans.ImmersedBoundaryGrid) = structured_grid(grid.underlying_grid)
structured_grid(field::Oceananigans.AbstractField) = structured_grid(field.grid)

#####
##### The U-pivot field layout as the grid `Regridder` regrids.
#####

"""
    FoldedLayoutGrid(g::OceananigansCurvilinearGrid)

The field layout of a U-pivot (`RightCenterFolded`) tripolar grid as the grid
CR regrids: every slot `i + (j - 1) Nx`, the fold row's primaries
(CR's halves, `1:Nx÷4` and `Nx÷2+1:Nx÷2+Nx÷4`) as their cells and its
duplicates as ghosts, a degenerate ring of zero area that the tree never
emits (the tree is the provider's quadtree masked to [`unique_cells`](@ref)),
so a physical cell is counted once and the regridder keeps the layout's
size. `finalize_regridding!` then mirrors each primary's value into its
duplicate ([`mirror_fold_partners!`](@ref)).
"""
struct FoldedLayoutGrid{G<:OceananigansCurvilinearGrid} <: GrI.AbstractGrid
    grid::G
end
@inline function _isghost(g::FoldedLayoutGrid, k::Integer)
    _, i, j = facecell(g.grid, Int(k))
    return j == g.grid.Ny && !_isprimary(g.grid, i)
end
# the one vertex every ghost ring repeats (the fold row's first vertex, as CR's native tree did)
_ghostpoint(g::FoldedLayoutGrid) = facevertex(g.grid, 1, 1, g.grid.Ny)
GrI.ncells(g::FoldedLayoutGrid) = GrI.ncells(g.grid)
GOCore.best_manifold(g::FoldedLayoutGrid) = GOCore.best_manifold(g.grid)
GrI.cellshape(::FoldedLayoutGrid) = GrI.ConvexCells()
GrI.tessellation(::FoldedLayoutGrid) = GrI.Tessellating()
GrI.isglobal(g::FoldedLayoutGrid) = GrI.isglobal(g.grid)
GrI.cell_geometry_is_expensive(g::FoldedLayoutGrid) = GrI.cell_geometry_is_expensive(g.grid)
function GrI.cell_boundary(g::FoldedLayoutGrid, k::Integer)
    _isghost(g, k) || return GrI.cell_boundary(g.grid, k)
    p = _ghostpoint(g)
    # a degenerate three-corner ring at that one point (`Corners(c)` would drop every
    # side as degenerate and leave no ring at all)
    return Structured.Corners{typeof(p)}((p, p, p, p), (1, 2, 3, 0), 3)
end
GrI.cell_corners(g::FoldedLayoutGrid, k::Integer) = GrI.cell_boundary(g, k)
GrI.cell_area(g::FoldedLayoutGrid, k::Integer) = _isghost(g, k) ? 0.0 : GrI.cell_area(g.grid, k)
GrI.cell_extent(g::FoldedLayoutGrid, k::Integer) =
    _isghost(g, k) ? GrI.SphericalCap(_ghostpoint(g), 0.0) : GrI.cell_extent(g.grid, k)
GrI.cell_centroid(g::FoldedLayoutGrid, k::Integer) = _isghost(g, k) ? _ghostpoint(g) : GrI.cell_centroid(g.grid, k)
# a ghost contains nothing (the geometric fallback reads its one-point ring as enclosing every point)
GrI.cell_contains(g::FoldedLayoutGrid, k::Integer, p) = !_isghost(g, k) && GrI.cell_contains(g.grid, k, p)
for H in (GrI.Balanced, GrI.ForSweeps, GrI.ForQueries)
    @eval GrI.leaf_capacity(g::FoldedLayoutGrid, hint::$H) = GrI.leaf_capacity(g.grid, hint)
end
# the provider's quadtree, its leaves skipping the ghosts and emitting the layout's positions
GrI.treeify(g::FoldedLayoutGrid, n::Integer) =
    Structured.MaskedCursor{false}(GrI.treeify(g.grid, n), GrI.parentpositions(unique_cells(g.grid)))
GrI.location_is_analytic(::FoldedLayoutGrid) = false
GrI.neighbors_are_analytic(::FoldedLayoutGrid) = true
GrI.neighbors(g::FoldedLayoutGrid, k::Integer; kw...) = GrI.neighbors(g.grid, k; kw...)
Base.show(io::IO, g::FoldedLayoutGrid) = print(io, "FoldedLayoutGrid(", g.grid.Nx, "×", g.grid.Ny, " U-pivot layout)")

"""
    ConservativeRegridding.asgrid(grid::Oceananigans.Grids.AbstractGrid)
    ConservativeRegridding.asgrid(field::Oceananigans.AbstractField)

The grid `Regridder` regrids for an Oceananigans grid or field: the
provider of its field layout ([`structured_grid`](@ref)), a U-pivot tripolar
grid as its [`FoldedLayoutGrid`](@ref). An immersed-boundary grid answers its
underlying grid's (every cell of the layout, land included, as the field's
data does).
"""
function ConservativeRegridding.asgrid(grid::Oceananigans.Grids.AbstractGrid)
    g = structured_grid(grid)
    return g isa OceananigansCurvilinearGrid && g.fold === :center ? FoldedLayoutGrid(g) : g
end
ConservativeRegridding.asgrid(grid::Oceananigans.MultiRegion.MultiRegionGrid) = structured_grid(grid)
ConservativeRegridding.asgrid(field::Oceananigans.AbstractField) = ConservativeRegridding.asgrid(field.grid)

end
