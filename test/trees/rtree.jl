using ConservativeRegridding.Trees
using Test
import ConservativeRegridding
import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using GeometryOps: FlexibleRTrees

# A vector of `nx × ny` axis-aligned boxes tiling the rectangle `x × y`.
function box_vector(nx, ny; x = (0.0, 1.0), y = (0.0, 1.0))
    xs = range(x...; length = nx + 1)
    ys = range(y...; length = ny + 1)
    return vec([
        GI.Polygon([GI.LinearRing([(xs[i], ys[j]), (xs[i+1], ys[j]), (xs[i+1], ys[j+1]), (xs[i], ys[j+1]), (xs[i], ys[j])])])
        for i in 1:nx, j in 1:ny
    ])
end

function leaf_count(node)
    STI.isleaf(node) && return length(collect(STI.child_indices_extents(node)))
    return sum(leaf_count, STI.getchild(node))
end

@testset "treeify builds an RTree over a vector of polygons" begin
    polys = box_vector(20, 17)
    tree = Trees.treeify(GO.Planar(), polys)
    @test tree isa Trees.GeometryMaintainingTreeWrapper
    @test tree.tree isa FlexibleRTrees.RTree{FlexibleRTrees.STR}
    @test tree.geoms === polys
    @test Trees.getcell(tree, 5) === polys[5]
    @test Trees.ncells(tree) == Trees.cell_index_count(tree) == length(polys)

    # mixed polygons and multipolygons
    mixed = [GI.MultiPolygon([polys[1], polys[2]]), polys[3]]
    @test Trees.treeify(GO.Planar(), mixed) isa Trees.GeometryMaintainingTreeWrapper
    # other iterables are unchanged
    @test Trees.treeify(GO.Planar(), (p for p in polys)) isa STI.FlatNoTree

    @testset "$(nameof(typeof(algorithm))), nodecapacity = $nodecapacity" for algorithm in (FlexibleRTrees.STR(), FlexibleRTrees.HPR(), FlexibleRTrees.Unsorted()), nodecapacity in (2, 4, 16)
        tree = Trees.treeify(GO.Planar(), polys; algorithm, nodecapacity)
        @test tree.tree.algorithm === algorithm
        # `ncells` counts the leaves below each node; `cell_index_count` is always global.
        stack = collect(STI.getchild(tree))
        while !isempty(stack)
            node = pop!(stack)
            @test Trees.ncells(node) == leaf_count(node)
            @test Trees.cell_index_count(node) == length(polys)
            STI.isleaf(node) || append!(stack, collect(STI.getchild(node)))
        end
    end
end

@testset "RTree regridding matches FlatNoTree" begin
    src, dst = box_vector(13, 11), box_vector(7, 9)
    reference = ConservativeRegridding.Regridder(GO.Planar(), STI.FlatNoTree(dst), STI.FlatNoTree(src); normalize = false)
    for algorithm in (FlexibleRTrees.STR(), FlexibleRTrees.HPR(), FlexibleRTrees.Unsorted())
        dst_tree = Trees.treeify(GO.Planar(), dst; algorithm)
        r = ConservativeRegridding.Regridder(GO.Planar(), dst_tree, src; normalize = false)
        @test r.intersections ≈ reference.intersections
    end

    # On the sphere the RTree indexes 3D Cartesian boxes, paired against caps or boxes.
    to_unit_sphere(ps) = [GO.transform(GO.UnitSphereFromGeographic(), p) for p in ps]
    src = to_unit_sphere(box_vector(36, 18; x = (-180.0, 180.0), y = (-90.0, 90.0)))
    dst = to_unit_sphere(box_vector(24, 12; x = (-180.0, 180.0), y = (-90.0, 90.0)))
    @test Trees.treeify(GO.Spherical(), dst).tree.extent isa GO.Extents.Extent{(:X, :Y, :Z)}
    r = ConservativeRegridding.Regridder(GO.Spherical(), dst, src; normalize = false)
    reference = ConservativeRegridding.Regridder(GO.Spherical(), STI.FlatNoTree(dst), STI.FlatNoTree(src); normalize = false)
    @test r.intersections ≈ reference.intersections
end
