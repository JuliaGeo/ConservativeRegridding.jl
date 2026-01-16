using ConservativeRegridding: Trees
using GeometryOps: SpatialTreeInterface as STI
import GeometryOps as GO
import GeoInterface as GI

y = range(-90.0, 90.0, length=100)
x = range(-180.0, 180.0, length=100)

tree = Trees.RegularGrid(GO.Spherical(), x, y)
tree_cursor = Trees.QuadtreeCursor(tree) |> Trees.KnownFullSphereExtentWrapper
tree_cursor = Trees.TopDownQuadtreeCursor(tree) |> Trees.KnownFullSphereExtentWrapper

idxs = Tuple{Int, Int}[]
log = STI.dual_depth_first_search(GO.UnitSpherical._intersects, tree_cursor, tree_cursor) do i1, i2
    push!(idxs, (i1, i2))
end


STI.getchild(tree_cursor, 1)

fidxs2 = filter(x -> x[1] == x[2], idxs) 

bools = fill(false, size(getcell(tree)))
for (i1, i2) in fidxs2
    bools[i1] = true
end

all(bools)

# using GLMakie
# heatmap(bools)

function napply(f, arg, n)
    res = arg
    for i in 1:n
        res = f(res)
    end
    return res
end

leaf1 = napply(x -> STI.getchild(x, 1), tree_cursor, 5)

c1 = STI.node_extent(leaf1)
GO.UnitSpherical._intersects(c1, c1)

chitr = STI.child_indices_extents(leaf1)

c1 = STI.getchild(leaf1, 1)
p1 = Trees.getcell(c1, 1)
up1 = GO.transform(GO.UnitSphereFromGeographic(), p1)

ue1 = GI.getexterior(up1)
GO.UnitSpherical.slerp(GI.getpoint(ue1, 1), GI.getpoint(ue1, 2), 0.5)





c2 = STI.getchild(tree_cursor, 2)
STI.node_extent(c2)

GO.UnitSpherical._intersects(STI.node_extent(c2), STI.node_extent(c2))

using GeoMakie
f, a, p = meshimage(-180..180, -90..90, reshape([colorant"white"], 1, 1); axis = (; type = GlobeAxis), zlevel = -300_000)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
poly!(a, STI.node_extent(c2); color = :red, alpha = 0.5, zlevel = 100_000, transparency = true)

idxs = Tuple{Int, Int}[]
log = STI.dual_depth_first_search(GO.UnitSpherical._intersects, c2, c2) do i1, i2
    push!(idxs, (i1, i2))
end
idxs

c2c = collect(STI.getchild(c2))


c2 = STI.getchild(tree_cursor, 4) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1)
chillins = STI.getchild(c2) |> collect



f, a, p = meshimage(-180..180, -90..90, reshape([colorant"white"], 1, 1); axis = (; type = GlobeAxis), zlevel = -300_000)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
# poly!(a, STI.node_extent(c2); color = :red, alpha = 0.5, zlevel = 100_000, transparency = true)
# poly!(a, vec(collect(Trees.getcell(c2))); strokecolor = :red, strokewidth = 1, alpha = 0.5, zlevel = 100_000, transparency = true, color = :transparent)
poly!(a, STI.node_extent(chillins[1]); color = Cycled(1), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[2]); color = Cycled(2), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[3]); color = Cycled(3), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[4]); color = Cycled(4), alpha = 0.5, zlevel = 100_000, transparency = true)

lines!(a, vec(collect(Trees.getcell(chillins[1]))); color = Cycled(1), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[2]))); color = Cycled(2), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[3]))); color = Cycled(3), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[4]))); color = Cycled(4), alpha = 0.5, zlevel = 100_000, transparency = true)

c2 = STI.getchild(c2, 1)