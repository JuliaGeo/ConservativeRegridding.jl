module ConservativeRegriddingOceananigansExt

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode, RightFaceFolded, RightCenterFolded, inactive_node
using Oceananigans.Fields: AbstractField
using Oceananigans.Architectures: CPU
using Oceananigans.Operators: Δxᶜᶠᵃ, Δyᶠᶜᵃ, Δzᶜᶜᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, Δzᶠᶠᶜ, extrinsic_vector

using ConservativeRegridding
using ConservativeRegridding: Regridder, ExampleFieldFunction
using ConservativeRegridding.Trees

import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
import LibGEOS
import Oceananigans.Architectures: on_architecture
import SparseArrays
using LinearAlgebra
using StaticArrays

instantiate(L) = L()
instantiate(::Type{Nothing}) = Center()

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

# An FPivot Tripolar grid has a `RightFaceFolded` topology: the fold is at `Face` nodes,
# which means there is an extra line of Face nodes at the north boundary.
# The prognostic domain for fields `Center`ed in `y` ends at `Ny-1`.
const FPivotTripolarGrid = Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid{<:Any, <:Any, RightFaceFolded}

function compute_cell_matrix(grid::FPivotTripolarGrid)
    Nx, Ny, _ = size(grid)
    ℓx, ℓy    = Center(), Center()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    arch = grid.architecture
    FT = eltype(grid)

    cell_matrix = Array{Tuple{FT, FT}}(undef, Nx+1, Ny)

    # Not GPU compatible so we need to move the grid on the CPU
    cpu_grid = on_architecture(CPU(), grid)
    _compute_cell_matrix!(cell_matrix, Nx+1, Ny, ℓx, ℓy, cpu_grid)

    return on_architecture(arch, cell_matrix)
end

flip(::Face) = Center()
flip(::Center) = Face()

function _compute_cell_matrix!(cell_matrix, Fx, Fy, ℓx, ℓy, grid)
    for i in 1:Fx, j in 1:Fy
        vx = flip(ℓx)
        vy = flip(ℓy)

        xl = ξnode(i, j, 1, grid, vx, vy, nothing)
        yl = ηnode(i, j, 1, grid, vx, vy, nothing)

        @inbounds cell_matrix[i, j] = (xl, yl)
    end
end

"""
    PaddedTreeWrapper(tree, n_padding, padding_polygon, index_offset)

Wraps a spatial tree and adds `n_padding` ghost cells after the real cells.
Ghost cells return `padding_polygon` from `getcell` and have zero area.
They do NOT participate in the spatial tree traversal (dual DFS won't find them
as candidates), ensuring they don't contribute to intersection computations.

This is used for TripolarGrid fold rows where duplicate cells must be present
for dimension matching but should not contribute to intersections.
"""
struct PaddedTreeWrapper{T, P} <: Trees.AbstractTreeWrapper
    tree::T
    n_padding::Int
    padding_polygon::P
    index_offset::Int
end

Base.parent(w::PaddedTreeWrapper) = w.tree

# Override ncells to include padding
function Trees.ncells(w::PaddedTreeWrapper)
    real = Trees.ncells(parent(w))
    return (real[1] + w.n_padding, real[2])
end

function Trees.ncells(w::PaddedTreeWrapper, dim::Int)
    if dim == 1
        return Trees.ncells(parent(w), 1) + w.n_padding
    else
        return Trees.ncells(parent(w), dim)
    end
end

# Override getcell(w, i) for individual cell access
function Trees.getcell(w::PaddedTreeWrapper, i::Int)
    local_i = i - w.index_offset
    n_real = prod(Trees.ncells(parent(w)))
    if local_i <= n_real
        return Trees.getcell(parent(w), i)
    else
        return w.padding_polygon
    end
end

# Override getcell(w) for iteration over all cells (including ghost)
function Trees.getcell(w::PaddedTreeWrapper)
    real_cells = Trees.getcell(parent(w))
    ghost_cells = (w.padding_polygon for _ in 1:w.n_padding)
    return Iterators.flatten((real_cells, ghost_cells))
end

# Define the ConservativeRegridding interface for Oceananigans grids.

function Trees.treeify(
    manifold::GOCore.Spherical,
    grid::Oceananigans.Grids.ZRegOrthogonalSphericalShellGrid{<: Number, <: Any, Oceananigans.RightCenterFolded}
)
    # Compute the matrix of vertices - for an n×m grid, this is an (n+1)×(m+1) matrix of vertices.
    cells_longlat = compute_cell_matrix(grid)
    cells_unitspherical = GO.UnitSphereFromGeographic().(cells_longlat)

    Nx = size(cells_unitspherical, 1) - 1
    Ny = size(cells_unitspherical, 2) - 1
    Nhalf = Nx ÷ 2
    @assert iseven(Nx) "RightFaceFolded requires even number of cells in x"

    # 1. Rest of grid: all rows except the fold row → Nx × (Ny-1) cells
    rest_vertices = cells_unitspherical[:, 1:Ny]
    rest_grid = Trees.CellBasedGrid(manifold, rest_vertices)
    N_rest = Nx * (Ny - 1)

    # 2. Fold row halves.
    #    In a RightCenterFolded grid, each fold half has a within-half duplication:
    #    cell i and cell (Nhalf+1-i) are the same physical quadrilateral (their
    #    bottom vertices coincide with the other's top vertices due to the fold
    #    geometry).  We keep only the first Nquarter = Nhalf÷2 cells as real
    #    polygons in the spatial tree and add the remaining duplicate cells as
    #    ghost cells via PaddedTreeWrapper.  Ghost cells exist for dimension
    #    matching (the source field has Nhalf cells per fold half) but are never
    #    reached during the dual DFS, so they contribute nothing to intersections.
    Nquarter = Nhalf ÷ 2

    # Build CellBasedGrids with only the unique cells (Nquarter per half)
    left_real_grid = Trees.CellBasedGrid(manifold, cells_unitspherical[1:Nquarter+1, Ny:Ny+1])
    right_real_grid = Trees.CellBasedGrid(manifold, cells_unitspherical[Nhalf+1:Nhalf+Nquarter+1, Ny:Ny+1])

    # Create degenerate polygon for ghost cells (all vertices at the same point)
    p = cells_unitspherical[1, Ny]
    ghost_polygon = GI.Polygon(SA[GI.LinearRing(SA[p, p, p, p, p])])

    # Build quadtree cursors over only the real cells, then wrap with
    # PaddedTreeWrapper to add ghost cells for dimension matching.
    # Ghost cells are never in the spatial tree → never intersection candidates.
    left_cursor = Trees.IndexOffsetQuadtreeCursor(left_real_grid, N_rest)
    left_padded = PaddedTreeWrapper(left_cursor, Nquarter, ghost_polygon, N_rest)

    right_cursor = Trees.IndexOffsetQuadtreeCursor(right_real_grid, N_rest + Nhalf)
    right_padded = PaddedTreeWrapper(right_cursor, Nquarter, ghost_polygon, N_rest + Nhalf)

    # Wrap all sub-trees in KnownFullSphereExtentWrapper so their top-level extents
    # don't cause the dual DFS to incorrectly prune candidate pairs.
    rest_tree = Trees.KnownFullSphereExtentWrapper(Trees.IndexOffsetQuadtreeCursor(rest_grid, 0))
    left_top_tree = Trees.KnownFullSphereExtentWrapper(left_padded)
    right_top_tree = Trees.KnownFullSphereExtentWrapper(right_padded)

    # Combine into a multi-tree; offsets are cumulative cell counts for searchsortedfirst
    tree = Trees.MultiTreeWrapper(
        [rest_tree, left_top_tree, right_top_tree],
        [N_rest, N_rest + Nhalf, N_rest + 2 * Nhalf]
    )

    return Trees.KnownFullSphereExtentWrapper(tree)
end
function Trees.treeify(
    manifold::GOCore.Spherical,
    grid::FPivotTripolarGrid
)
    # compute_cell_matrix for FPivotTripolarGrid returns (Nx+1, Ny) which
    # excludes the diagnostic fold row, giving Nx*(Ny-1) cells matching
    # the interior size of Center-Center fields in released Oceananigans.
    cells_longlat = compute_cell_matrix(grid)
    cells_unitspherical = GO.UnitSphereFromGeographic().(cells_longlat)
    cbg = Trees.CellBasedGrid(manifold, cells_unitspherical)
    tree = Trees.TopDownQuadtreeCursor(cbg)
    return Trees.KnownFullSphereExtentWrapper(tree)
end
Trees.treeify(manifold::GOCore.Spherical, grid::Oceananigans.ImmersedBoundaryGrid) = Trees.treeify(manifold, grid.underlying_grid)
Trees.treeify(manifold::GOCore.Planar, grid::Oceananigans.ImmersedBoundaryGrid) = Trees.treeify(manifold, grid.underlying_grid)
function Trees.treeify(
    manifold::GOCore.Spherical,
    grid::Oceananigans.Grids.AbstractGrid
)
    # Compute the matrix of vertices - for an n×m grid, this is an (n+1)×(m+1) matrix of vertices.
    cells_longlat = compute_cell_matrix(grid) # from oceananigans_common.jl
    # Cells come in long-lat coords - convert to unit spherical.
    # This makes things substantially more efficient at query and intersection time.
    cells_unitspherical = GO.UnitSphereFromGeographic().(cells_longlat)
    # Define the grid, which is the base of the quadtree we will construct
    # on top of it.
    grid = Trees.CellBasedGrid(
        manifold, 
        cells_unitspherical
    )
    # We choose to build a quadtree on top of this grid.
    # To do this we choose a top-down subdividing quadtree.
    tree = Trees.TopDownQuadtreeCursor(grid)
    # Finally, this is slightly unnecessary but makes me feel good,
    # we wrap the tree in a known-extent wrapper.  This should actually not be done
    # for long-lat grids that don't cover the whole sphere - TODO.
    return Trees.KnownFullSphereExtentWrapper(tree)
end
function Trees.treeify(manifold::GOCore.Planar, grid::Oceananigans.RectilinearGrid)
    # Compute the matrix of verti|ces - for an n×m grid, this is an (n+1)×(m+1) matrix of vertices.
    cells = compute_cell_matrix(grid) # from oceananigans_common.jl
    # Define the grid, which is the base of the quadtree we will construct
    # on top of it.
    grid = Trees.CellBasedGrid(manifold, cells)
    tree = Trees.TopDownQuadtreeCursor(grid)
    return tree
end

Trees.treeify(field::Oceananigans.Field) = Trees.treeify(field.grid)
Trees.treeify(field::Oceananigans.AbstractField) = Trees.treeify(field.grid)

Trees.treeify(manifold::GOCore.Manifold, field::Oceananigans.Field) = Trees.treeify(manifold, field.grid)
Trees.treeify(manifold::GOCore.Manifold, field::Oceananigans.AbstractField) = Trees.treeify(manifold, field.grid)

@inline vertical_wet_fraction(i, j, k, grid::Oceananigans.ImmersedBoundaryGrid, ::Center, ::Center, ::Center) =
    clamp(Δzᶜᶜᶜ(i, j, k, grid) / Δzᶜᶜᶜ(i, j, k, grid.underlying_grid), zero(eltype(grid)), one(eltype(grid)))

@inline vertical_wet_fraction(i, j, k, grid::Oceananigans.ImmersedBoundaryGrid, ::Face, ::Center, ::Center) =
    clamp(Δzᶠᶜᶜ(i, j, k, grid) / Δzᶠᶜᶜ(i, j, k, grid.underlying_grid), zero(eltype(grid)), one(eltype(grid)))

@inline vertical_wet_fraction(i, j, k, grid::Oceananigans.ImmersedBoundaryGrid, ::Center, ::Face, ::Center) =
    clamp(Δzᶜᶠᶜ(i, j, k, grid) / Δzᶜᶠᶜ(i, j, k, grid.underlying_grid), zero(eltype(grid)), one(eltype(grid)))

@inline vertical_wet_fraction(i, j, k, grid::Oceananigans.ImmersedBoundaryGrid, ::Face, ::Face, ::Center) =
    clamp(Δzᶠᶠᶜ(i, j, k, grid) / Δzᶠᶠᶜ(i, j, k, grid.underlying_grid), zero(eltype(grid)), one(eltype(grid)))

@inline vertical_wet_fraction(i, j, k, grid::Oceananigans.ImmersedBoundaryGrid, ℓx, ℓy, ℓz) = one(eltype(grid))

function wet_cell_fractions(field::Oceananigans.AbstractField{LX, LY, LZ, <:Oceananigans.ImmersedBoundaryGrid}) where {LX, LY, LZ}
    grid = on_architecture(CPU(), field.grid)
    Nx, Ny, _ = size(field)
    FT = eltype(grid)
    wet_fractions = Vector{FT}(undef, Nx * Ny)
    ℓx, ℓy, ℓz = instantiate(LX), instantiate(LY), instantiate(LZ)

    idx = 1
    for j in 1:Ny, i in 1:Nx
        inactive = inactive_node(i, j, 1, grid, ℓx, ℓy, ℓz)
        peripheral = Oceananigans.ImmersedBoundaries.immersed_peripheral_node(i, j, 1, grid, ℓx, ℓy, ℓz)

        if inactive || peripheral
            wet_fractions[idx] = zero(FT)
        else
            wet_fractions[idx] = vertical_wet_fraction(i, j, 1, grid, ℓx, ℓy, ℓz)
        end

        idx += 1
    end

    return wet_fractions
end

function fit_length(wet_fractions::AbstractVector{FT}, n::Int) where FT
    length(wet_fractions) == n && return wet_fractions

    fitted = zeros(FT, n)
    copyto!(fitted, 1, wet_fractions, 1, min(length(wet_fractions), n))
    return fitted
end

function ConservativeRegridding.areas(manifold::GOCore.Manifold,
                                      field::Oceananigans.AbstractField{<:Any, <:Any, <:Any, <:Oceananigans.ImmersedBoundaryGrid},
                                      tree)
    geometric_areas = ConservativeRegridding.areas(manifold, tree)
    wet_fractions = fit_length(wet_cell_fractions(field), length(geometric_areas))
    return geometric_areas .* wet_fractions
end

#=
This section implements the velocity-specific transport remap requested for
Tripolar -> LatitudeLongitude workflows. The key design choice is to avoid
interpolating face velocities to centers. Instead, we:

1. Treat each source u-face / v-face as a line segment carrying integrated face transport.
2. Intersect those segments with destination face control polygons.
3. Use overlap fractions (partial-face handling) to allocate transport conservatively.
4. Project accumulated source transport vectors onto destination orthogonal components.
5. Divide by destination face area to recover destination face velocity.

This is a first-order (piecewise-constant on source face) conservative remap.
=#

@inline face_linear_index(i, j, Nx) = i + (j - 1) * Nx

@inline function unwrap_longitude_pair(lon1::Real, lon2::Real)
    λ1 = Float64(lon1)
    λ2 = Float64(lon2)

    while λ2 - λ1 > 180
        λ2 -= 360
    end

    while λ2 - λ1 < -180
        λ2 += 360
    end

    return λ1, λ2
end

@inline function midpoint_lonlat(p1::Tuple{<:Real, <:Real}, p2::Tuple{<:Real, <:Real})
    λ1, λ2 = unwrap_longitude_pair(p1[1], p2[1])
    φ1 = Float64(p1[2])
    φ2 = Float64(p2[2])
    return (0.5 * (λ1 + λ2), 0.5 * (φ1 + φ2))
end

function continuous_longitude_ring(points::NTuple{4, Tuple{<:Real, <:Real}})
    unwrapped = Vector{Tuple{Float64, Float64}}(undef, 4)

    λprev = Float64(points[1][1])
    unwrapped[1] = (λprev, Float64(points[1][2]))

    for n in 2:4
        λ = Float64(points[n][1])
        φ = Float64(points[n][2])

        while λ - λprev > 180
            λ -= 360
        end

        while λ - λprev < -180
            λ += 360
        end

        unwrapped[n] = (λ, φ)
        λprev = λ
    end

    return unwrapped
end

@inline geometry_grid(grid::Oceananigans.ImmersedBoundaryGrid) = grid.underlying_grid
@inline geometry_grid(grid) = grid

@inline function center_horizontal_size(grid)
    g = geometry_grid(grid)
    Nx, Ny, _ = size(g)

    if g isa FPivotTripolarGrid
        # RightFaceFolded Tripolar has an extra face row at the fold;
        # center-located indices live on Ny-1.
        return Nx, Ny - 1
    else
        return Nx, Ny
    end
end

@inline function face_basis_u(grid, i, j)
    g = geometry_grid(grid)
    g isa Oceananigans.LatitudeLongitudeGrid && return 1.0, 0.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    uₑ, vₑ = extrinsic_vector(ii, jj, 1, g, 1.0, 0.0)
    return Float64(uₑ), Float64(vₑ)
end

@inline function face_basis_v(grid, i, j)
    g = geometry_grid(grid)
    g isa Oceananigans.LatitudeLongitudeGrid && return 0.0, 1.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    uₑ, vₑ = extrinsic_vector(ii, jj, 1, g, 0.0, 1.0)
    return Float64(uₑ), Float64(vₑ)
end

@inline function face_node_lonlat(grid, i, j)
    g = geometry_grid(grid)
    λ = ξnode(i, j, 1, g, Face(), Face(), nothing)
    φ = ηnode(i, j, 1, g, Face(), Face(), nothing)
    return Float64(λ), Float64(φ)
end

@inline function source_u_segment_endpoints(grid, i, j)
    p1 = face_node_lonlat(grid, i, j)
    p2 = face_node_lonlat(grid, i, j + 1)
    λ1, λ2 = unwrap_longitude_pair(p1[1], p2[1])
    return λ1, p1[2], λ2, p2[2]
end

@inline function source_v_segment_endpoints(grid, i, j)
    p1 = face_node_lonlat(grid, i, j)
    p2 = face_node_lonlat(grid, i + 1, j)
    λ1, λ2 = unwrap_longitude_pair(p1[1], p2[1])
    return λ1, p1[2], λ2, p2[2]
end

function destination_u_control_polygon_points(grid, i, j)
    # The u-face control polygon is the dual quadrilateral around the i,j x-face.
    pᵂˢ = midpoint_lonlat(face_node_lonlat(grid, i - 1, j),     face_node_lonlat(grid, i, j))
    pᴱˢ = midpoint_lonlat(face_node_lonlat(grid, i, j),         face_node_lonlat(grid, i + 1, j))
    pᴱᴺ = midpoint_lonlat(face_node_lonlat(grid, i, j + 1),     face_node_lonlat(grid, i + 1, j + 1))
    pᵂᴺ = midpoint_lonlat(face_node_lonlat(grid, i - 1, j + 1), face_node_lonlat(grid, i, j + 1))
    return continuous_longitude_ring((pᵂˢ, pᴱˢ, pᴱᴺ, pᵂᴺ))
end

function destination_v_control_polygon_points(grid, i, j)
    # The v-face control polygon is the dual quadrilateral around the i,j y-face.
    pˢᵂ = midpoint_lonlat(face_node_lonlat(grid, i, j - 1),     face_node_lonlat(grid, i, j))
    pˢᴱ = midpoint_lonlat(face_node_lonlat(grid, i + 1, j - 1), face_node_lonlat(grid, i + 1, j))
    pᴺᴱ = midpoint_lonlat(face_node_lonlat(grid, i + 1, j),     face_node_lonlat(grid, i + 1, j + 1))
    pᴺᵂ = midpoint_lonlat(face_node_lonlat(grid, i, j),         face_node_lonlat(grid, i, j + 1))
    return continuous_longitude_ring((pˢᵂ, pˢᴱ, pᴺᴱ, pᴺᵂ))
end

function polygon_wkt(points::AbstractVector{<:Tuple{<:Real, <:Real}}, shift::Real)
    io = IOBuffer()
    print(io, "POLYGON((")

    for n in eachindex(points)
        λ = points[n][1] + shift
        φ = points[n][2]
        print(io, λ, " ", φ, ",")
    end

    print(io, points[1][1] + shift, " ", points[1][2], "))")
    return String(take!(io))
end

@inline function polyline_length(coords)
    n = length(coords)
    n < 2 && return 0.0

    ℓ = 0.0
    @inbounds for p in 1:n-1
        x₁ = Float64(coords[p][1])
        y₁ = Float64(coords[p][2])
        x₂ = Float64(coords[p + 1][1])
        y₂ = Float64(coords[p + 1][2])
        ℓ += hypot(x₂ - x₁, y₂ - y₁)
    end

    return ℓ
end

overlap_line_length(geom) = overlap_line_length(geom, GI.geomtrait(geom))

overlap_line_length(geom, ::GI.LineStringTrait) = polyline_length(GI.coordinates(geom))

function overlap_line_length(geom, ::GI.MultiLineStringTrait)
    total = 0.0
    for linecoords in GI.coordinates(geom)
        total += polyline_length(linecoords)
    end
    return total
end

function overlap_line_length(geom, ::GI.GeometryCollectionTrait)
    total = 0.0
    for n in 1:GI.ngeom(geom)
        total += overlap_line_length(GI.getgeom(geom, n))
    end
    return total
end

overlap_line_length(geom, trait) = 0.0

@inline x_face_area(i, j, k, grid) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
@inline y_face_area(i, j, k, grid) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)

struct SourceSegmentMetadata{T}
    source_kind::UInt8 # 0x01 => source u-face, 0x02 => source v-face
    source_face_index::Int
    east_component::T
    north_component::T
    segment_length::T
end

"""
    VelocityLineIntegralRegridder

Internal storage for the line-integral velocity remap operator used by
`ConservativeRegridding.regrid_velocity_transport!`.

The four sparse operators represent the conservative transport mapping:

- `Wuu`: source u-face transport contribution to destination u-face transport
- `Wuv`: source v-face transport contribution to destination u-face transport
- `Wvu`: source u-face transport contribution to destination v-face transport
- `Wvv`: source v-face transport contribution to destination v-face transport

Each matrix entry is assembled from:
1. partial source-face overlap fraction (line-segment ∩ destination face control polygon),
2. vector projection from source intrinsic basis into destination intrinsic basis.
"""
struct VelocityLineIntegralRegridder{M, V}
    Wuu::M
    Wuv::M
    Wvu::M
    Wvv::M
    src_u_size::NTuple{3, Int}
    src_v_size::NTuple{3, Int}
    dst_u_size::NTuple{3, Int}
    dst_v_size::NTuple{3, Int}
    src_u_transport::V
    src_v_transport::V
    dst_u_transport::V
    dst_v_transport::V
end

function build_source_segments(src_u, src_v)
    grid = src_u.grid
    Nxᵤ, Nyᵤ, _ = size(src_u)
    Nxᵥ, Nyᵥ, _ = size(src_v)

    geoms = LibGEOS.LineString[]
    metadata = SourceSegmentMetadata{Float64}[]
    lookup = IdDict{LibGEOS.LineString, Int}()

    for j in 1:Nyᵤ, i in 1:Nxᵤ
        λ₁, φ₁, λ₂, φ₂ = source_u_segment_endpoints(grid, i, j)
        ℓ = hypot(λ₂ - λ₁, φ₂ - φ₁)

        if isfinite(ℓ) && ℓ > eps(Float64)
            seg = LibGEOS.LineString([[λ₁, φ₁], [λ₂, φ₂]])
            east, north = face_basis_u(grid, i, j)
            idx = face_linear_index(i, j, Nxᵤ)

            push!(geoms, seg)
            push!(metadata, SourceSegmentMetadata(0x01, idx, east, north, ℓ))
            lookup[seg] = length(geoms)
        end
    end

    for j in 1:Nyᵥ, i in 1:Nxᵥ
        λ₁, φ₁, λ₂, φ₂ = source_v_segment_endpoints(grid, i, j)
        ℓ = hypot(λ₂ - λ₁, φ₂ - φ₁)

        if isfinite(ℓ) && ℓ > eps(Float64)
            seg = LibGEOS.LineString([[λ₁, φ₁], [λ₂, φ₂]])
            east, north = face_basis_v(grid, i, j)
            idx = face_linear_index(i, j, Nxᵥ)

            push!(geoms, seg)
            push!(metadata, SourceSegmentMetadata(0x02, idx, east, north, ℓ))
            lookup[seg] = length(geoms)
        end
    end

    return geoms, metadata, lookup
end

function accumulate_face_weights!(
    row_uu, col_uu, val_uu,
    row_uv, col_uv, val_uv,
    d,
    dst_east, dst_north,
    points,
    tree,
    source_lookup,
    source_metadata
)
    # We evaluate three longitude shifts so control polygons near 0/360° overlap
    # source segments that may be represented with longitudes outside [0, 360].
    for shift in (-360.0, 0.0, 360.0)
        poly = LibGEOS.readgeom(polygon_wkt(points, shift))
        candidates = LibGEOS.query(tree, poly)

        for seg in candidates
            sidx = source_lookup[seg]
            meta = source_metadata[sidx]

            clipped = LibGEOS.intersection(seg, poly)
            overlap = overlap_line_length(clipped)
            overlap <= 0 && continue

            frac = overlap / max(meta.segment_length, eps(Float64))
            proj = meta.east_component * dst_east + meta.north_component * dst_north
            w = frac * proj

            if meta.source_kind == 0x01
                push!(row_uu, d); push!(col_uu, meta.source_face_index); push!(val_uu, w)
            else
                push!(row_uv, d); push!(col_uv, meta.source_face_index); push!(val_uv, w)
            end
        end
    end

    return nothing
end

"""
Rescale each source-face column so that global destination sums reproduce
the exact source-basis projection target.

This preserves linearity and enforces global transport-component closure:
for arbitrary source transports `F`, the global sum of mapped transport equals
the source global transport projected onto the chosen destination component.
"""
function reconcile_column_targets(W, targets::AbstractVector{<:Real})
    colsum = vec(sum(W; dims = 1))
    scales = zeros(Float64, length(targets))

    @inbounds for j in eachindex(scales)
        den = colsum[j]
        tgt = Float64(targets[j])
        scales[j] = abs(den) <= eps(Float64) ? 0.0 : tgt / den
    end

    return W * Diagonal(scales)
end

"""
    ConservativeRegridding.VelocityLineIntegralRegridder(dst_u, dst_v, src_u, src_v; threaded=True())

Construct a velocity transport regridder that preserves integrated face transport
without center interpolation. This constructor is intended for source grids where
u and v are native C-grid face components (for example Tripolar) and destination
grids where u and v are also native face components (for example LatitudeLongitude).

Implementation details:
- source face transport is treated as piecewise constant over each source face segment;
- partial intersections are handled with overlap fractions from clipped line length;
- source vector transport is projected into destination orthogonal face components;
- sparse operators are precomputed once and reused for all vertical levels.
"""
function ConservativeRegridding.VelocityLineIntegralRegridder(dst_u, dst_v, src_u, src_v; threaded = true)
    src_u.grid === src_v.grid || error("src_u and src_v must live on the same source grid.")
    dst_u.grid === dst_v.grid || error("dst_u and dst_v must live on the same destination grid.")

    src_arch = src_u.grid.architecture
    dst_arch = dst_u.grid.architecture
    src_arch isa CPU || error("VelocityLineIntegralRegridder currently supports CPU source grids only.")
    dst_arch isa CPU || error("VelocityLineIntegralRegridder currently supports CPU destination grids only.")

    Nxˢᵤ, Nyˢᵤ, _ = size(src_u)
    Nxˢᵥ, Nyˢᵥ, _ = size(src_v)
    Nxᵈᵤ, Nyᵈᵤ, _ = size(dst_u)
    Nxᵈᵥ, Nyᵈᵥ, _ = size(dst_v)

    nˢᵤ = Nxˢᵤ * Nyˢᵤ
    nˢᵥ = Nxˢᵥ * Nyˢᵥ
    nᵈᵤ = Nxᵈᵤ * Nyᵈᵤ
    nᵈᵥ = Nxᵈᵥ * Nyᵈᵥ

    segments, source_metadata, source_lookup = build_source_segments(src_u, src_v)
    tree = LibGEOS.STRtree(segments)

    row_uu = Int[]; col_uu = Int[]; val_uu = Float64[]
    row_uv = Int[]; col_uv = Int[]; val_uv = Float64[]
    row_vu = Int[]; col_vu = Int[]; val_vu = Float64[]
    row_vv = Int[]; col_vv = Int[]; val_vv = Float64[]

    source_u_east_target = zeros(Float64, nˢᵤ)
    source_u_north_target = zeros(Float64, nˢᵤ)
    source_v_east_target = zeros(Float64, nˢᵥ)
    source_v_north_target = zeros(Float64, nˢᵥ)

    for meta in source_metadata
        if meta.source_kind == 0x01
            source_u_east_target[meta.source_face_index] = meta.east_component
            source_u_north_target[meta.source_face_index] = meta.north_component
        else
            source_v_east_target[meta.source_face_index] = meta.east_component
            source_v_north_target[meta.source_face_index] = meta.north_component
        end
    end

    for j in 1:Nyᵈᵤ, i in 1:Nxᵈᵤ
        d = face_linear_index(i, j, Nxᵈᵤ)
        dst_east, dst_north = face_basis_u(dst_u.grid, i, j)
        points = destination_u_control_polygon_points(dst_u.grid, i, j)

        accumulate_face_weights!(
            row_uu, col_uu, val_uu,
            row_uv, col_uv, val_uv,
            d,
            dst_east, dst_north,
            points,
            tree,
            source_lookup,
            source_metadata
        )
    end

    for j in 1:Nyᵈᵥ, i in 1:Nxᵈᵥ
        d = face_linear_index(i, j, Nxᵈᵥ)
        dst_east, dst_north = face_basis_v(dst_v.grid, i, j)
        points = destination_v_control_polygon_points(dst_v.grid, i, j)

        accumulate_face_weights!(
            row_vu, col_vu, val_vu,
            row_vv, col_vv, val_vv,
            d,
            dst_east, dst_north,
            points,
            tree,
            source_lookup,
            source_metadata
        )
    end

    Wuu = SparseArrays.sparse(row_uu, col_uu, val_uu, nᵈᵤ, nˢᵤ)
    Wuv = SparseArrays.sparse(row_uv, col_uv, val_uv, nᵈᵤ, nˢᵥ)
    Wvu = SparseArrays.sparse(row_vu, col_vu, val_vu, nᵈᵥ, nˢᵤ)
    Wvv = SparseArrays.sparse(row_vv, col_vv, val_vv, nᵈᵥ, nˢᵥ)

    # Enforce exact global component closure in the destination basis by
    # reconciling per-source-face column sums to the source basis projections.
    Wuu = reconcile_column_targets(Wuu, source_u_east_target)
    Wuv = reconcile_column_targets(Wuv, source_v_east_target)
    Wvu = reconcile_column_targets(Wvu, source_u_north_target)
    Wvv = reconcile_column_targets(Wvv, source_v_north_target)

    src_u_transport = zeros(Float64, nˢᵤ)
    src_v_transport = zeros(Float64, nˢᵥ)
    dst_u_transport = zeros(Float64, nᵈᵤ)
    dst_v_transport = zeros(Float64, nᵈᵥ)

    return VelocityLineIntegralRegridder(
        Wuu, Wuv, Wvu, Wvv,
        size(src_u), size(src_v), size(dst_u), size(dst_v),
        src_u_transport, src_v_transport, dst_u_transport, dst_v_transport
    )
end

"""
    ConservativeRegridding.regrid_velocity_transport!(dst_u, dst_v, R, src_u, src_v)

Apply a precomputed `VelocityLineIntegralRegridder` to remap face velocities by
conserving integrated face transport per vertical level.

For each level `k`:
1. Convert source velocities to source face transports (`u * Aˣ`, `v * Aʸ`).
2. Apply the four sparse coupling operators (`Wuu`, `Wuv`, `Wvu`, `Wvv`).
3. Convert destination face transport back to velocity by dividing by destination face area.

Dry / zero-area destination faces are safely set to zero.
"""
function ConservativeRegridding.regrid_velocity_transport!(dst_u, dst_v, R::VelocityLineIntegralRegridder, src_u, src_v)
    size(src_u) == R.src_u_size || error("src_u size mismatch with VelocityLineIntegralRegridder.")
    size(src_v) == R.src_v_size || error("src_v size mismatch with VelocityLineIntegralRegridder.")
    size(dst_u) == R.dst_u_size || error("dst_u size mismatch with VelocityLineIntegralRegridder.")
    size(dst_v) == R.dst_v_size || error("dst_v size mismatch with VelocityLineIntegralRegridder.")

    src_u_data = interior(src_u)
    src_v_data = interior(src_v)
    dst_u_data = interior(dst_u)
    dst_v_data = interior(dst_v)

    src_grid = src_u.grid
    dst_grid = dst_u.grid

    Nxˢᵤ, Nyˢᵤ, Nz = size(src_u)
    Nxˢᵥ, Nyˢᵥ, _ = size(src_v)
    Nxᵈᵤ, Nyᵈᵤ, _ = size(dst_u)
    Nxᵈᵥ, Nyᵈᵥ, _ = size(dst_v)

    for k in 1:Nz
        n = 1
        for j in 1:Nyˢᵤ, i in 1:Nxˢᵤ
            area = x_face_area(i, j, k, src_grid)
            uᵢ = src_u_data[i, j, k]
            R.src_u_transport[n] = (isfinite(uᵢ) && isfinite(area)) ? uᵢ * area : 0.0
            n += 1
        end

        n = 1
        for j in 1:Nyˢᵥ, i in 1:Nxˢᵥ
            area = y_face_area(i, j, k, src_grid)
            vᵢ = src_v_data[i, j, k]
            R.src_v_transport[n] = (isfinite(vᵢ) && isfinite(area)) ? vᵢ * area : 0.0
            n += 1
        end

        mul!(R.dst_u_transport, R.Wuu, R.src_u_transport)
        mul!(R.dst_u_transport, R.Wuv, R.src_v_transport, 1.0, 1.0)

        mul!(R.dst_v_transport, R.Wvu, R.src_u_transport)
        mul!(R.dst_v_transport, R.Wvv, R.src_v_transport, 1.0, 1.0)

        n = 1
        for j in 1:Nyᵈᵤ, i in 1:Nxᵈᵤ
            area = x_face_area(i, j, k, dst_grid)
            dst_u_data[i, j, k] = (isfinite(area) && area != 0) ? R.dst_u_transport[n] / area : 0.0
            n += 1
        end

        n = 1
        for j in 1:Nyᵈᵥ, i in 1:Nxᵈᵥ
            area = y_face_area(i, j, k, dst_grid)
            dst_v_data[i, j, k] = (isfinite(area) && area != 0) ? R.dst_v_transport[n] / area : 0.0
            n += 1
        end
    end

    Oceananigans.fill_halo_regions!(dst_u)
    Oceananigans.fill_halo_regions!(dst_v)

    return nothing
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

end
