using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
using Statistics
using Test

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    Nx, Ny, _ = size(field.grid)
    ℓx, ℓy    = Center(), Center()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    grid = field.grid
    arch = grid.architecture
    FT = eltype(grid)

    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, Nx+1, Ny+1)

    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, (Nx+1, Ny+1), _compute_cell_matrix!, cell_matrix, Nx, ℓx, ℓy, grid)

    return cell_matrix
end

flip(::Face) = Center()
flip(::Center) = Face()

@kernel function _compute_cell_matrix!(cell_matrix, Nx, ℓx, ℓy, grid)
    i, j = @index(Global, NTuple)

    vx = flip(ℓx)
    vy = flip(ℓy)

    xl = ξnode(i, j, 1, grid, vx, vy, nothing)
    yl = ηnode(i, j, 1, grid, vx, vy, nothing)

    @inbounds cell_matrix[i, j] = (xl, yl)
end


# Define the ConservativeRegridding interface for Oceananigans grids.
function Trees.treeify(
    manifold::GOCore.Spherical,
    field::Oceananigans.Field{T1, T2, T3, T4, GridT}
) where {T1, T2, T3, T4, GridT}
    # Compute the matrix of vertices - for an n×m grid, this is an (n+1)×(m+1) matrix of vertices.
    cells_longlat = compute_cell_matrix(field) # from oceananigans_common.jl
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
function Trees.treeify(manifold::GOCore.Planar, field::Oceananigans.Field{T1, T2, T3, T4, GridT}) where {T1, T2, T3, T4, GridT <: Oceananigans.RectilinearGrid}
    # Compute the matrix of verti|ces - for an n×m grid, this is an (n+1)×(m+1) matrix of vertices.
    cells = compute_cell_matrix(field) # from oceananigans_common.jl
    # Define the grid, which is the base of the quadtree we will construct
    # on top of it.
    grid = Trees.CellBasedGrid(manifold, cells)
    tree = Trees.TopDownQuadtreeCursor(grid)
    return tree
end

# Also define which manifold the grid lives on.  This gives us accurate area as well for any simulation
# (e.g. on Mars?!)
GOCore.best_manifold(grid::Oceananigans.RectilinearGrid) = GO.Planar()
GOCore.best_manifold(grid::Oceananigans.LatitudeLongitudeGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(grid::Oceananigans.TripolarGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(field::Oceananigans.Field) = GOCore.best_manifold(field.grid)


coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

dst = CenterField(coarse_grid)
src = CenterField(fine_grid)

set!(src, (x, y, z) -> x)

regridder = ConservativeRegridding.Regridder(dst, src)

ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

@test mean(interior(dst)) == mean(interior(src))

set!(dst, (x, y, z) -> rand())

ConservativeRegridding.regrid!(vec(interior(src)), transpose(regridder), vec(interior(dst)))

@test mean(dst) ≈ mean(src) rtol=1e-5

large_domain_grid = RectilinearGrid(size=(100, 100), x=(0, 2), y=(0, 2), topology=(Periodic, Periodic, Flat))
small_domain_grid = RectilinearGrid(size=(200, 200), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))

src = CenterField(small_domain_grid)
dst = CenterField(large_domain_grid)

set!(src, 1)

regridder = ConservativeRegridding.Regridder(dst, src)

ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

# Compute the integral and make sure it is the same as the original field
dst_int = Field(Integral(dst))
src_int = Field(Integral(src))

compute!(dst_int)
compute!(src_int)

@test only(dst_int) ≈ only(src_int)
