using GLMakie, GeoMakie
# include("quadtree_geoms.jl")

using ConservativeRegridding.Trees
import GeometryOps: SpatialTreeInterface as STI
import GeometryOps as GO, GeoInterface as GI

# qt = CellBasedQuadtree(dst_cells)
qt = Trees.RegularGridQuadtree(LinRange(-180, 180, 500), LinRange(-90, 90, 500))
qtc = Trees.QuadtreeCursor(qt) |> Trees.KnownFullSphereExtentWrapper

qtc = STI.getchild(qtc, 1)

# c = STI.node_extent(STI.getchild(qtc, 1))
f, a, p = meshimage(-180..180, -90..90, reshape([colorant"white"], 1, 1); axis = (; type = GlobeAxis), zlevel = -300_000)
poly!(a, vec(collect(Trees.getcell(STI.getchild(qtc, 1)))); transparency = true, color = :transparent, strokewidth = 1, strokecolor = Makie.Cycled(1))
poly!(a, vec(collect(Trees.getcell(STI.getchild(qtc, 2)))); transparency = true, color = :transparent, strokewidth = 1, strokecolor = Makie.Cycled(2))
poly!(a, vec(collect(Trees.getcell(STI.getchild(qtc, 3)))); transparency = true, color = :transparent, strokewidth = 1, strokecolor = Makie.Cycled(3))
poly!(a, vec(collect(Trees.getcell(STI.getchild(qtc, 4)))); transparency = true, color = :transparent, strokewidth = 1, strokecolor = Makie.Cycled(4))

# poly!(a, c; color = :red, alpha = 0.5, zlevel = 100_000, transparency = true)

poly!(a, vec(collect(Trees.getcell.((qt,), CartesianIndices(Trees.leaf_idxs(qtc))))))

p1 = poly!(a, STI.node_extent(STI.getchild(qtc, 1)); color = Cycled(1), alpha = .7, transparency=true)
p2 = poly!(a, STI.node_extent(STI.getchild(qtc, 2)); color = Cycled(2), alpha = .7, transparency=true)
p3 = poly!(a, STI.node_extent(STI.getchild(qtc, 3)); color = Cycled(3), alpha = .7, transparency=true)
p4 = poly!(a, STI.node_extent(STI.getchild(qtc, 4)); color = Cycled(4), alpha = .7, transparency=true)
l1 = lines!(a, STI.node_extent(STI.getchild(qtc, 1)); color = Cycled(1), alpha = .7, transparency=false)
l2 = lines!(a, STI.node_extent(STI.getchild(qtc, 2)); color = Cycled(2), alpha = .7, transparency=false)
l3 = lines!(a, STI.node_extent(STI.getchild(qtc, 3)); color = Cycled(3), alpha = .7, transparency=false)
l4 = lines!(a, STI.node_extent(STI.getchild(qtc, 4)); color = Cycled(4), alpha = .7, transparency=false)



using GeoMakie, GLMakie
f, a, p = lines(vec(polygons); axis = (; type = GlobeAxis), transparency = true)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
meshimage!(a, -180..180, -90..90, reshape([colorant"white"], 1, 1); zlevel = -300_000)

poly!(a, c; color = :steelblue, alpha = 0.7, transparency = true, zlevel = 0)