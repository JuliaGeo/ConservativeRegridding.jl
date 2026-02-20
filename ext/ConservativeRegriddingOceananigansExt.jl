module ConservativeRegriddingOceananigansExt

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode, RightFaceFolded, RightCenterFolded
using Oceananigans.Fields: AbstractField
using Oceananigans.Architectures: CPU

using ConservativeRegridding
using ConservativeRegridding: Regridder, ExampleFieldFunction
using ConservativeRegridding.Trees

import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
import Oceananigans.Architectures: on_architecture

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


using StaticArrays

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
    #    We use ExplicitPolygonGrid (not CellBasedGrid) so that each cell's
    #    polygon is independent.
    Nquarter = Nhalf ÷ 2

    # Build ExplicitPolygonGrids with only the unique cells (Nquarter per half)
    left_polys = Matrix{GI.Polygon}(undef, Nquarter, 1)
    for k in 1:Nquarter
        left_polys[k, 1] = GI.Polygon(SA[GI.LinearRing(SA[
            cells_unitspherical[k, Ny], cells_unitspherical[k+1, Ny],
            cells_unitspherical[k+1, Ny+1], cells_unitspherical[k, Ny+1],
            cells_unitspherical[k, Ny]
        ])])
    end
    left_real_grid = Trees.ExplicitPolygonGrid(manifold, left_polys)

    right_polys = Matrix{GI.Polygon}(undef, Nquarter, 1)
    for k in 1:Nquarter
        i = Nhalf + k
        right_polys[k, 1] = GI.Polygon(SA[GI.LinearRing(SA[
            cells_unitspherical[i, Ny], cells_unitspherical[i+1, Ny],
            cells_unitspherical[i+1, Ny+1], cells_unitspherical[i, Ny+1],
            cells_unitspherical[i, Ny]
        ])])
    end
    right_real_grid = Trees.ExplicitPolygonGrid(manifold, right_polys)

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
    Nx, Ny, _ = size(grid)
    # The FPivot-specific compute_cell_matrix returns (Nx+1, Ny) which excludes
    # the fold Face row at j=Ny+1.  Compute the full (Nx+1, Ny+1) vertex matrix
    # so we can build a tree with all Nx×Ny cells.
    FT = eltype(grid)
    cell_matrix = Array{Tuple{FT, FT}}(undef, Nx+1, Ny+1)
    cpu_grid = on_architecture(CPU(), grid)
    _compute_cell_matrix!(cell_matrix, Nx+1, Ny+1, Center(), Center(), cpu_grid)

    cells_unitspherical = GO.UnitSphereFromGeographic().(cell_matrix)

    Nhalf = Nx ÷ 2
    @assert iseven(Nx) "RightFaceFolded requires even number of cells in x"

    # 1. Rest of grid: all rows except the fold row → Nx × (Ny-1) cells
    rest_vertices = cells_unitspherical[:, 1:Ny]
    rest_grid = Trees.CellBasedGrid(manifold, rest_vertices)
    N_rest = Nx * (Ny - 1)

    # 2. Fold row: split into left and right halves so the quadtree descent
    #    doesn't mix cells from opposite sides of the fold.
    #    Unlike RightCenterFolded, RightFaceFolded fold cells are NOT exact
    #    duplicates, so all cells are real (no ghost padding needed).
    left_top_vertices = cells_unitspherical[1:Nhalf+1, Ny:Ny+1]
    left_top_grid = Trees.CellBasedGrid(manifold, left_top_vertices)

    right_top_vertices = cells_unitspherical[Nhalf+1:Nx+1, Ny:Ny+1]
    right_top_grid = Trees.CellBasedGrid(manifold, right_top_vertices)

    # Wrap all sub-trees in KnownFullSphereExtentWrapper
    rest_tree = Trees.KnownFullSphereExtentWrapper(Trees.IndexOffsetQuadtreeCursor(rest_grid, 0))
    left_top_tree = Trees.KnownFullSphereExtentWrapper(Trees.IndexOffsetQuadtreeCursor(left_top_grid, N_rest))
    right_top_tree = Trees.KnownFullSphereExtentWrapper(Trees.IndexOffsetQuadtreeCursor(right_top_grid, N_rest + Nhalf))

    tree = Trees.MultiTreeWrapper(
        [rest_tree, left_top_tree, right_top_tree],
        [N_rest, N_rest + Nhalf, N_rest + 2 * Nhalf]
    )

    return Trees.KnownFullSphereExtentWrapper(tree)
end
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
