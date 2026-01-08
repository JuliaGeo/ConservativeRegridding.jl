include("quadtree_geoms.jl")
nrows, ncols = 8, 9
polygons = [
    GI.Polygon([
        GI.LinearRing([
            (Float64(j - 1), Float64(i - 1)),  # bottom-left
            (Float64(j),     Float64(i - 1)),  # bottom-right
            (Float64(j),     Float64(i)),      # top-right
            (Float64(j - 1), Float64(i)),      # top-left
            (Float64(j - 1), Float64(i - 1))   # close the ring
        ])
    ])
    for i in 1:nrows, j in 1:ncols
]
    
# Create the quadtree
qt = ImplicitQuadtree(polygons)
qtc = ImplicitQuadtreeCursor(qt)

using GeoMakie, GLMakie
f, a, p = lines(vec(polygons); axis = (; type = GlobeAxis), transparency = true)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
meshimage!(a, -180..180, -90..90, reshape([colorant"white"], 1, 1); zlevel = -300_000)

co = Observable{GO.UnitSpherical.SphericalCap{Float64}}(GO.minimum_bounding_circle(GO.Welzl(GO.Spherical()), vec(polygons)))

poly!(a, co; color = :steelblue, alpha = 0.7, transparency = true, zlevel = 0)
p1 = poly!(a, STI.node_extent(STI.getchild(qtc, 1)); color = Cycled(1), alpha = .7, transparency=true)
p2 = poly!(a, STI.node_extent(STI.getchild(qtc, 2)); color = Cycled(2), alpha = .7, transparency=true)
p3 = poly!(a, STI.node_extent(STI.getchild(qtc, 3)); color = Cycled(3), alpha = .7, transparency=true)
p4 = poly!(a, STI.node_extent(STI.getchild(qtc, 4)); color = Cycled(4), alpha = .7, transparency=true)
l1 = lines!(a, STI.node_extent(STI.getchild(qtc, 1)); color = Cycled(1), alpha = .7, transparency=false)
l2 = lines!(a, STI.node_extent(STI.getchild(qtc, 2)); color = Cycled(2), alpha = .7, transparency=false)
l3 = lines!(a, STI.node_extent(STI.getchild(qtc, 3)); color = Cycled(3), alpha = .7, transparency=false)
l4 = lines!(a, STI.node_extent(STI.getchild(qtc, 4)); color = Cycled(4), alpha = .7, transparency=false)



poly!(a, vec(polygons[1:end, 1:end-1]); color = getindex.((LinearIndices((1:size(polygons, 1), 1:size(polygons, 2)-1)),), vec(CartesianIndices((1:size(polygons, 1), 1:size(polygons, 2)-1)))), alpha = .7, transparency=true)

quadtree_lats = -180:179
quadtree_lons = -90:89
quadtree_polys = [
    GI.Polygon([GI.LinearRing([
        (quadtree_lons[j], quadtree_lats[i]),
        (quadtree_lons[j+1], quadtree_lats[i]),
        (quadtree_lons[j+1], quadtree_lats[i+1]),
        (quadtree_lons[j], quadtree_lats[i+1]),
        (quadtree_lons[j], quadtree_lats[i])
    ])])
    for i in 1:length(quadtree_lats)-1, j in 1:length(quadtree_lons)-1
]

iqt = ImplicitQuadtree(quadtree_polys)
iqtc = ImplicitQuadtreeCursor(iqt)

STI.node_extent(iqtc)
