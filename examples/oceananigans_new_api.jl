using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
using Statistics
using GLMakie, GeoMakie
import LibGEOS

include("oceananigans_common.jl")

# Define the ConservativeRegridding interface for Oceananigans grids.
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
# Also define which manifold the grid lives on.  This gives us accurate area as well for any simulation
# (e.g. on Mars?!)
GOCore.best_manifold(grid::Oceananigans.LatitudeLongitudeGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(grid::Oceananigans.TripolarGrid) = GO.Spherical(; radius = grid.radius)
GOCore.best_manifold(field::Oceananigans.Field) = GOCore.best_manifold(field.grid)

# Instantiate some grids
lonlat_test_grid = LatitudeLongitudeGrid(size=(36, 18, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
lonlat_coarse_grid = LatitudeLongitudeGrid(size=(100, 100, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
lonlat_fine_grid = LatitudeLongitudeGrid(size=(200, 200, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

tripolar_test_grid = TripolarGrid(size=(36, 18, 1), fold_topology = RightFaceFolded)
tripolar_fine_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)

# Select which grid you want to use
src_grid = lonlat_fine_grid
dst_grid = tripolar_fine_grid
# Construct fields from those grids
src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)
# Set the field to some test data
# In this case, we set it to the longitude of the cell
set!(src_field, VortexField(; lat0 = 80))

@time regridder = ConservativeRegridding.Regridder(
    dst_field,
    src_field;
    progress = true
)

ConservativeRegridding.regrid!(vec(interior(dst_field)), regridder, vec(interior(src_field)))

f, a, p = heatmap(interior(src_field, :, :, 1); colorrange = extrema(interior(src_field)), highclip = :red)
cb = Colorbar(f[1, 2], p)

src_polys = collect(Trees.getcell(Trees.treeify(src_field))) |>  x-> GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), x) .|> GI.convert(LibGEOS) |> vec
dst_polys = collect(Trees.getcell(Trees.treeify(dst_field))) |>  x-> GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), x) .|> GI.convert(LibGEOS) |> vec

f, a, p = poly(src_polys; color = vec(interior(src_field)), axis = (; type = GlobeAxis), colorrange = extrema(interior(src_field)), highclip = :red)
f, a, p = poly(dst_polys; color = vec(interior(dst_field)), axis = (; type = GlobeAxis), colorrange = extrema(interior(src_field)), highclip = :red)

lines!(a, GeoMakie.coastlines(); zlevel = 100_000, color = :orange)

# ## Metrics
# First, set up the analytical destination field.
analytical_dst_field = CenterField(dst_grid)
set!(analytical_dst_field, VortexField(; lat0 = 80))
areas_dst = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(dst_field))
# Then, compute metrics.  The global misfit field is nice to show:
misfit = abs.((interior(analytical_dst_field) .- interior(dst_field)) ./ interior(analytical_dst_field))
# and these are the numerical metrics:
mean_misfit = mean(misfit)
max_misfit = maximum(misfit)
rms_misfit = sqrt(mean(misfit.^2))
L_min = (minimum(interior(analytical_dst_field)) - minimum(interior(dst_field))) / maximum(abs, interior(analytical_dst_field))
L_max = (maximum(interior(analytical_dst_field)) - maximum(interior(dst_field))) / maximum(abs, interior(analytical_dst_field))
target_global_conservation = abs(sum(areas_dst .* vec(interior(dst_field))) - sum(areas_dst .* vec(interior(analytical_dst_field)))) / sum(areas_dst .* vec(interior(analytical_dst_field)))


f, a, p = poly(dst_polys; color = vec(misfit), axis = (; type = GlobeAxis))
lines!(a, GeoMakie.coastlines(); zlevel = 100_000, color = :orange, transparency = true, alpha = 0.5)
cb = Colorbar(f[1, 2], p; label = "Misfit")
f
# TODO: how to enforce conservative regridding on a tripolar grid,
# that has an open hole at the south pole??

# areas_dst = ConservativeRegridding.areas(GO.Spherical(), dst_tree)
# areas_src = ConservativeRegridding.areas(GO.Spherical(), src_tree)

# sum(areas_dst)
# sum(areas_src)
# sum(mat)

# vec(sum(mat; dims = 1)) |> sum
# vec(sum(mat; dims = 2)) |> sum

# @test all(sum((dst_field * Oceananigans.Operators.Az)) ≈ sum((src_field * Oceananigans.Operators.Az)))

# sum(dst_field * Oceananigans.Operators.Az)
# sum(src_field * Oceananigans.Operators.Az)

# @test all(vec(sum(mat; dims = 1)) .≈ areas_dst)
# @test all(vec(sum(mat; dims = 2)) .≈ areas_src)
