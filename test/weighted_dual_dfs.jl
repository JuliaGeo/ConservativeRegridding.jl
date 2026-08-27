using Test
using ConservativeRegridding
using ConservativeRegridding.Trees

import ConservativeRegridding as CR
import ConservativeRegridding.WeightedDualDepthFirstSearch as WDFS
import Extents
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI

const _spherical_intersects = Extents.intersects

function pairs_generic(predicate, tree1, tree2)
    pairs = Tuple{Int,Int}[]
    STI.dual_depth_first_search(predicate, tree1, tree2) do i1, i2
        push!(pairs, (i1, i2))
    end
    return pairs
end

function pairs_weighted(predicate, tree1, tree2)
    pairs = Tuple{Int,Int}[]
    WDFS.weighted_dual_depth_first_search(predicate, tree1, tree2) do i1, i2
        push!(pairs, (i1, i2))
    end
    return pairs
end

spherical_regular_grid(nx, ny) = RegularGrid(
    GO.Spherical(),
    collect(range(-180.0, 180.0, length = nx + 1)),
    collect(range(-90.0, 90.0, length = ny + 1)),
)

function regional_spherical_grid(
    nx, ny; lon = (-20.0, 40.0), lat = (0.0, 50.0),
)
    return RegularGrid(
        GO.Spherical(),
        collect(range(lon[1], lon[2], length = nx + 1)),
        collect(range(lat[1], lat[2], length = ny + 1)),
    )
end

planar_regular_grid(nx, ny) = RegularGrid(
    GO.Planar(),
    collect(range(0.0, 10.0, length = nx + 1)),
    collect(range(0.0, 10.0, length = ny + 1)),
)

function spherical_cellbased_grid(nx, ny)
    lons = range(-180, 180, length = nx + 1)
    lats = range(-90, 90, length = ny + 1)
    points = [
        GO.UnitSpherical.UnitSphereFromGeographic()((lon, lat))
        for lon in lons, lat in lats
    ]
    return CellBasedGrid(GO.Spherical(), points)
end

# Count construction of a leaf's inline entries without changing the wrapped tree's
# work estimate.  The traversal should carry a fixed leaf rather than reconstruct it.
mutable struct LeafEntryCounter
    n::Int
end

struct CountedLeafEntriesNode{N}
    node::N
    counter::LeafEntryCounter
end

STI.isspatialtree(::Type{<:CountedLeafEntriesNode}) = true
STI.isleaf(node::CountedLeafEntriesNode) = STI.isleaf(node.node)
STI.nchild(node::CountedLeafEntriesNode) = STI.nchild(node.node)
STI.getchild(node::CountedLeafEntriesNode) =
    (CountedLeafEntriesNode(child, node.counter) for child in STI.getchild(node.node))
STI.getchild(node::CountedLeafEntriesNode, i::Int) =
    CountedLeafEntriesNode(STI.getchild(node.node, i), node.counter)
STI.node_extent(node::CountedLeafEntriesNode) = STI.node_extent(node.node)
function STI.child_indices_extents(node::CountedLeafEntriesNode)
    node.counter.n += 1
    return STI.child_indices_extents(node.node)
end
CR.Trees.ncells(node::CountedLeafEntriesNode) = CR.Trees.ncells(node.node)
CR.Trees.split_weight(node::CountedLeafEntriesNode) = CR.Trees.split_weight(node.node)

@testset "weighted_dual_depth_first_search" begin
    @testset "same candidate set, deterministic weighted order" begin
        cases = Any[]
        for (nx, ny) in ((4, 4), (16, 16), (13, 17))
            grid = regional_spherical_grid(nx, ny)
            push!(cases, ("QuadtreeCursor spherical $(nx)x$(ny) self",
                          _spherical_intersects, QuadtreeCursor(grid), QuadtreeCursor(grid)))
            push!(cases, ("TopDownQuadtreeCursor spherical $(nx)x$(ny) self",
                          _spherical_intersects,
                          TopDownQuadtreeCursor(grid), TopDownQuadtreeCursor(grid)))
        end
        push!(cases, (
            "spherical regional 16x8 vs 7x11",
            _spherical_intersects,
            QuadtreeCursor(regional_spherical_grid(16, 8)),
            QuadtreeCursor(regional_spherical_grid(7, 11)),
        ))
        push!(cases, (
            "spherical regional offset overlap",
            _spherical_intersects,
            TopDownQuadtreeCursor(regional_spherical_grid(21, 13)),
            TopDownQuadtreeCursor(regional_spherical_grid(
                9, 25; lon = (10.0, 70.0), lat = (20.0, 80.0))),
        ))
        push!(cases, (
            "spherical cell-based vs regular",
            _spherical_intersects,
            QuadtreeCursor(spherical_cellbased_grid(12, 9)),
            TopDownQuadtreeCursor(spherical_regular_grid(9, 12)),
        ))
        push!(cases, (
            "planar regular 12x12 self",
            Extents.intersects,
            QuadtreeCursor(planar_regular_grid(12, 12)),
            QuadtreeCursor(planar_regular_grid(12, 12)),
        ))

        polygons = [
            GI.Polygon([GI.LinearRing([
                (x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y),
            ])])
            for x in 0:4, y in 0:4
        ]
        push!(cases, (
            "FlatNoTree polygons",
            Extents.intersects,
            STI.FlatNoTree(vec(polygons)),
            STI.FlatNoTree(vec(polygons)),
        ))

        for (name, predicate, tree1, tree2) in cases
            @testset "$name" begin
                generic = pairs_generic(predicate, tree1, tree2)
                weighted = pairs_weighted(predicate, tree1, tree2)
                @test sort(weighted) == sort(generic)
                @test pairs_weighted(predicate, tree1, tree2) == weighted
            end
        end
    end

    @testset "a fixed leaf is constructed once in either orientation" begin
        deep = QuadtreeCursor(planar_regular_grid(16, 16))
        covering = GI.Polygon([GI.LinearRing([
            (-1.0, -1.0), (11.0, -1.0), (11.0, 11.0), (-1.0, 11.0), (-1.0, -1.0),
        ])])

        counter1 = LeafEntryCounter(0)
        shallow1 = CountedLeafEntriesNode(STI.FlatNoTree([covering]), counter1)
        @test !isempty(pairs_weighted(Extents.intersects, shallow1, deep))
        @test counter1.n == 1

        counter2 = LeafEntryCounter(0)
        shallow2 = CountedLeafEntriesNode(STI.FlatNoTree([covering]), counter2)
        @test !isempty(pairs_weighted(Extents.intersects, deep, shallow2))
        @test counter2.n == 1
    end

    @testset "cache machinery is absent" begin
        @test !isdefined(WDFS, :_extent_stack)
        @test !isdefined(WDFS, :children_extent_type)
        @test !isdefined(CR, :cached_dual_depth_first_search)
    end

    @testset "an Action still steers the deterministic traversal" begin
        tree1 = QuadtreeCursor(planar_regular_grid(8, 8))
        tree2 = QuadtreeCursor(planar_regular_grid(8, 8))
        expected = pairs_weighted(Extents.intersects, tree1, tree2)
        seen = Tuple{Int,Int}[]
        result = WDFS.weighted_dual_depth_first_search(
            Extents.intersects, tree1, tree2,
        ) do i1, i2
            push!(seen, (i1, i2))
            return length(seen) >= 3 ? GO.LoopStateMachine.Action(:full_return, true) : nothing
        end
        @test seen == expected[1:3]
        @test result isa GO.LoopStateMachine.Action
        @test result.name === :full_return
    end
end
