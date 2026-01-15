using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
using Statistics
using GLMakie

include("oceananigans_common.jl")

function Trees.treeify(
        manifold::GOCore.Spherical,
        field::Oceananigans.Field{T1, T2, T3, T4, GridT}
    ) where {T1, T2, T3, T4, GridT <: Union{Oceananigans.LatitudeLongitudeGrid, Oceananigans.TripolarGrid}}
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

GOCore.best_manifold(grid::Oceananigans.LatitudeLongitudeGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(grid::Oceananigans.TripolarGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(field::Oceananigans.Field) = GOCore.best_manifold(field.grid)

src_grid = LatitudeLongitudeGrid(size=(100, 100, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = LatitudeLongitudeGrid(size=(200, 200, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)


src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)
set!(src_field, (lon, lat, z) -> lon)

@time regridder = ConservativeRegridding.Regridder(
    dst_field,
    src_field;
    progress = true
)

ConservativeRegridding.regrid!(vec(interior(dst_field)), regridder, vec(interior(src_field)))

heatmap(interior(dst_field, :, :, 1))

areas_dst = ConservativeRegridding.areas(GO.Spherical(), dst_tree)
areas_src = ConservativeRegridding.areas(GO.Spherical(), src_tree)

sum(areas_dst)
sum(areas_src)
sum(mat)

vec(sum(mat; dims = 1)) |> sum
vec(sum(mat; dims = 2)) |> sum

@test all(sum((dst_field * Oceananigans.Operators.Az)) ≈ sum((src_field * Oceananigans.Operators.Az)))

sum(dst_field * Oceananigans.Operators.Az)
sum(src_field * Oceananigans.Operators.Az)

@test all(vec(sum(mat; dims = 1)) .≈ areas_dst)
@test all(vec(sum(mat; dims = 2)) .≈ areas_src)
