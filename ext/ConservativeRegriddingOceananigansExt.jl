module ConservativeRegriddingOceananigansExt

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using Oceananigans.Architectures: CPU

using ConservativeRegridding
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
    _compute_cell_matrix!(cell_matrix, Nx, Ny, ℓx, ℓy, cpu_grid)

    return on_architecture(arch, cell_matrix)
end

flip(::Face) = Center()
flip(::Center) = Face()

function _compute_cell_matrix!(cell_matrix, Nx, Ny, ℓx, ℓy, grid)
    for i in 1:Nx+1, j in 1:Ny+1
        vx = flip(ℓx)
        vy = flip(ℓy)

        xl = ξnode(i, j, 1, grid, vx, vy, nothing)
        yl = ηnode(i, j, 1, grid, vx, vy, nothing)

        @inbounds cell_matrix[i, j] = (xl, yl)
    end
end


# Define the ConservativeRegridding interface for Oceananigans grids.
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

GOCore.best_manifold(field::Oceananigans.Field) = GOCore.best_manifold(field.grid)

# Extend the `on_architecture` method for a `Regridder` object
on_architecture(arch, r::ConservativeRegridding.Regridder) =
    ConservativeRegridding.Regridder(
        on_architecture(arch, r.intersections),
        on_architecture(arch, r.dst_areas),
        on_architecture(arch, r.src_areas),
        on_architecture(arch, r.dst_temp),
        on_architecture(arch, r.src_temp)
    )


# Allow to set example data on the field
Oceananigans.set!(field::Oceananigans.Field, f::ExampleFieldFunction) = Oceananigans.set!(field, (lon, lat, z) -> f(lon, lat))

end
