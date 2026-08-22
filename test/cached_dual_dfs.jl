using Test
using ConservativeRegridding
using ConservativeRegridding.Trees
import ConservativeRegridding as CR
import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
import Extents

import ConservativeRegridding.CachedDualDepthFirstSearch as CDFS
using ConservativeRegridding.CachedDualDepthFirstSearch: cached_dual_depth_first_search, children_extent_type

#=
`cached_dual_depth_first_search` is a drop-in for `STI.dual_depth_first_search`: the
contract is that it reports the same `(i1, i2)` pairs, in the same order, for every tree.
Almost everything below is that equality, checked on trees that take each arm of the
caching decision - and then the mechanics of the decision itself.
=#

# Every pair, in traversal order - order is part of the contract, so this does not sort.
function pairs_generic(predicate, t1, t2)
    out = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate, t1, t2) do i1, i2
        push!(out, (i1, i2))
    end
    return out
end

function pairs_cached(predicate, t1, t2)
    out = Tuple{Int, Int}[]
    cached_dual_depth_first_search(predicate, t1, t2) do i1, i2
        push!(out, (i1, i2))
    end
    return out
end

function spherical_regular_grid(nx, ny)
    RegularGrid(GO.Spherical(), collect(range(-180.0, 180.0, length = nx + 1)),
                collect(range(-90.0, 90.0, length = ny + 1)))
end

# A grid that does not span the globe.  The bounding cap of a whole-sphere point set is
# the all-NaN one, and a NaN cap intersects nothing, so a global grid pair can prune to
# no candidates at all - true of both searches, but it makes for a vacuous comparison.
function regional_spherical_grid(nx, ny; lon = (-20.0, 40.0), lat = (0.0, 50.0))
    RegularGrid(GO.Spherical(), collect(range(lon[1], lon[2], length = nx + 1)),
                collect(range(lat[1], lat[2], length = ny + 1)))
end

function planar_regular_grid(nx, ny)
    RegularGrid(GO.Planar(), collect(range(0.0, 10.0, length = nx + 1)),
                collect(range(0.0, 10.0, length = ny + 1)))
end

function spherical_cellbased_grid(nx, ny)
    lons = range(-180, 180, length = nx + 1)
    lats = range(-90, 90, length = ny + 1)
    pts = [GO.UnitSpherical.UnitSphereFromGeographic()((lon, lat)) for lon in lons, lat in lats]
    return CellBasedGrid(GO.Spherical(), pts)
end

const _sph = Extents.intersects

# Counts `node_extent` calls, and inherits the caching decision of the tree it wraps, so
# the two searches see the same tree and differ only in how often they ask for an extent.
mutable struct ExtentCounter
    n::Int
end

# `Expensive` is the trait, so the same tree can be run through both arms of the caching
# decision and the two `node_extent` counts compared directly.
struct CountedNode{Expensive, N}
    node::N
    counter::ExtentCounter
end
CountedNode{E}(node::N, counter) where {E, N} = CountedNode{E, N}(node, counter)
STI.isspatialtree(::Type{<: CountedNode}) = true
STI.node_extent_is_expensive(::Type{<: CountedNode{E}}) where {E} = E
STI.isleaf(n::CountedNode) = STI.isleaf(n.node)
STI.nchild(n::CountedNode) = STI.nchild(n.node)
STI.getchild(n::CountedNode{E}) where {E} = (CountedNode{E}(c, n.counter) for c in STI.getchild(n.node))
STI.getchild(n::CountedNode{E}, i::Int) where {E} = CountedNode{E}(STI.getchild(n.node, i), n.counter)
STI.child_indices_extents(n::CountedNode) = STI.child_indices_extents(n.node)
function STI.node_extent(n::CountedNode)
    n.counter.n += 1
    return STI.node_extent(n.node)
end

# An extent-like type that is not an `Extents.Extent`, so a tree using both genuinely
# changes extent type with depth - the case one tree-wide stack cannot serve.
struct BoxExtent
    X::Tuple{Float64, Float64}
    Y::Tuple{Float64, Float64}
end
_asbox(e) = BoxExtent(e.X, e.Y)
_asextent(b::BoxExtent) = Extents.Extent(X = b.X, Y = b.Y)
_asextent(e) = e
_boxy_intersects(a, b) = Extents.intersects(_asextent(a), _asextent(b))

# Odd levels report a `BoxExtent`, even levels an `Extent`.  Siblings agree, levels do not.
struct AlternatingExtentNode{N}
    node::N
    level::Int
end
AlternatingExtentNode(node) = AlternatingExtentNode(node, 0)
STI.isspatialtree(::Type{<: AlternatingExtentNode}) = true
STI.node_extent_is_expensive(::Type{<: AlternatingExtentNode}) = true
CDFS.children_extent_type(::Type{<: AlternatingExtentNode}) = Union{Extents.Extent{(:X, :Y), NTuple{2, Tuple{Float64, Float64}}}, BoxExtent}
STI.isleaf(n::AlternatingExtentNode) = STI.isleaf(n.node)
STI.nchild(n::AlternatingExtentNode) = STI.nchild(n.node)
STI.getchild(n::AlternatingExtentNode) = (AlternatingExtentNode(c, n.level + 1) for c in STI.getchild(n.node))
STI.getchild(n::AlternatingExtentNode, i::Int) = AlternatingExtentNode(STI.getchild(n.node, i), n.level + 1)
STI.child_indices_extents(n::AlternatingExtentNode) = STI.child_indices_extents(n.node)
STI.node_extent(n::AlternatingExtentNode) =
    iseven(n.level) ? STI.node_extent(n.node) : _asbox(STI.node_extent(n.node))

# Same idea, but the trait is spelled on the node rather than on its type.
struct InstanceTraitNode{N}
    node::N
end
STI.isspatialtree(::Type{<: InstanceTraitNode}) = true
STI.node_extent_is_expensive(::Type{<: InstanceTraitNode}) = true
CDFS.children_extent_type(::InstanceTraitNode) = GO.UnitSpherical.SphericalCap{Float64}
STI.isleaf(n::InstanceTraitNode) = STI.isleaf(n.node)
STI.nchild(n::InstanceTraitNode) = STI.nchild(n.node)
STI.getchild(n::InstanceTraitNode) = (InstanceTraitNode(c) for c in STI.getchild(n.node))
STI.getchild(n::InstanceTraitNode, i::Int) = InstanceTraitNode(STI.getchild(n.node, i))
STI.child_indices_extents(n::InstanceTraitNode) = STI.child_indices_extents(n.node)
STI.node_extent(n::InstanceTraitNode) = STI.node_extent(n.node)

@testset "cached_dual_depth_first_search" begin

    @testset "the cursors that need it opt in, the one that does not stays out" begin
        @test STI.node_extent_is_expensive(QuadtreeCursor(spherical_regular_grid(8, 8)))
        @test STI.node_extent_is_expensive(TopDownQuadtreeCursor(spherical_regular_grid(8, 8)))
        @test STI.node_extent_is_expensive(QuadtreeCursor(spherical_cellbased_grid(8, 8)))
        # planar regular: `cell_range_extent` is four array reads, so caching buys nothing
        @test !STI.node_extent_is_expensive(QuadtreeCursor(planar_regular_grid(8, 8)))
        @test !STI.node_extent_is_expensive(TopDownQuadtreeCursor(planar_regular_grid(8, 8)))
        # a wrapper that forwards `node_extent` forwards the trait with it...
        cheap = QuadtreeCursor(planar_regular_grid(8, 8))
        costly = QuadtreeCursor(spherical_regular_grid(8, 8))
        @test STI.node_extent_is_expensive(Trees.IndexLocalizerRewrapperTree(costly, 0))
        @test !STI.node_extent_is_expensive(Trees.IndexLocalizerRewrapperTree(cheap, 0))
        # ...but one that overrides it with an O(1) answer does not
        @test !STI.node_extent_is_expensive(Trees.KnownFullSphereExtentWrapper(costly))
    end

    @testset "same pairs, same order, as the generic search" begin
        cases = Any[]
        for (nx, ny) in ((4, 4), (16, 16), (13, 17))
            g = spherical_regular_grid(nx, ny)
            push!(cases, ("QuadtreeCursor spherical $(nx)x$(ny) self", _sph,
                          QuadtreeCursor(g), QuadtreeCursor(g)))
            push!(cases, ("TopDownQuadtreeCursor spherical $(nx)x$(ny) self", _sph,
                          TopDownQuadtreeCursor(g), TopDownQuadtreeCursor(g)))
        end
        # different resolutions, so the two descents are genuinely unbalanced
        push!(cases, ("spherical regional 16x8 vs 7x11", _sph,
                      QuadtreeCursor(regional_spherical_grid(16, 8)),
                      QuadtreeCursor(regional_spherical_grid(7, 11))))
        push!(cases, ("spherical regional, offset overlap", _sph,
                      TopDownQuadtreeCursor(regional_spherical_grid(21, 13)),
                      TopDownQuadtreeCursor(regional_spherical_grid(9, 25; lon = (10.0, 70.0), lat = (20.0, 80.0)))))
        push!(cases, ("spherical cell-based vs regular", _sph,
                      QuadtreeCursor(spherical_cellbased_grid(12, 9)),
                      TopDownQuadtreeCursor(spherical_regular_grid(9, 12))))
        # the cheap arm: identical code path, so this is a regression guard
        push!(cases, ("planar regular 12x12 self", Extents.intersects,
                      QuadtreeCursor(planar_regular_grid(12, 12)),
                      QuadtreeCursor(planar_regular_grid(12, 12))))
        # a tree that stores its extents - nothing to cache at all
        polys = [GI.Polygon([GI.LinearRing([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)])])
                 for x in 0:4, y in 0:4]
        push!(cases, ("FlatNoTree of polygons", Extents.intersects,
                      STI.FlatNoTree(vec(polys)), STI.FlatNoTree(vec(polys))))

        for (name, predicate, t1, t2) in cases
            @testset "$name" begin
                expected = pairs_generic(predicate, t1, t2)
                @test !isempty(expected)
                @test pairs_cached(predicate, t1, t2) == expected
            end
        end
    end

    @testset "the whole regridder agrees, pair for pair" begin
        # `get_all_candidate_pairs` is what the caching search was wired into; both its
        # threaded and serial arms must still report what the generic search reports.
        # `TopDownQuadtreeCursor` on both sides: `Trees.split_weight`, which the threaded
        # frontier needs, goes through `ncells(cursor)`, and `QuadtreeCursor` has no
        # no-argument `ncells` method (pre-existing, unrelated to the caching search).
        src = TopDownQuadtreeCursor(regional_spherical_grid(24, 12))
        dst = TopDownQuadtreeCursor(regional_spherical_grid(11, 17))
        expected = pairs_generic(_sph, src, dst)
        @test !isempty(expected)
        @test CR.get_all_candidate_pairs(GO.False(), _sph, src, dst) == expected
        # the threaded frontier reassembles in DFS pre-order, so this is order-exact too
        @test CR.get_all_candidate_pairs(GO.True(), _sph, src, dst) == expected
    end

    @testset "an opted-in tree derives each child extent once" begin
        g = regional_spherical_grid(16, 16)
        counters = ExtentCounter[]
        function run(::Val{E}, search) where {E}
            c1 = ExtentCounter(0); c2 = ExtentCounter(0)
            push!(counters, c1, c2)
            out = search(_sph, CountedNode{E}(QuadtreeCursor(g), c1),
                         CountedNode{E}(QuadtreeCursor(g), c2))
            return out, c1.n + c2.n
        end
        opted_out, calls_out = run(Val(false), pairs_cached)
        opted_in, calls_in = run(Val(true), pairs_cached)
        @test !isempty(opted_in)
        @test opted_in == opted_out                        # caching cannot change the answer
        @test calls_in < calls_out                         # ...it just asks far less often
        # GeometryOps' generic search caches too when the trait is set - it just pays a
        # fresh vector for every visited node pair, where this one reuses a single stack.
        generic, calls_generic = run(Val(true), pairs_generic)
        @test generic == opted_in
        @test calls_in == calls_generic
    end

    @testset "the caching strategy is a compile-time choice" begin
        costly = QuadtreeCursor(spherical_regular_grid(8, 8))
        cheap = QuadtreeCursor(planar_regular_grid(8, 8))
        ecostly = STI.node_extent(costly)
        echeap = STI.node_extent(cheap)
        @test (@inferred CDFS._extent_stack(nothing, cheap, echeap)) === nothing
        @test (@inferred CDFS._extent_stack(nothing, costly, ecostly)) isa
              Vector{GO.UnitSpherical.SphericalCap{Float64}}
        # an existing stack of the right type is reused rather than replaced
        stack = typeof(ecostly)[]
        @test (@inferred CDFS._extent_stack(stack, costly, ecostly)) === stack
    end

    @testset "the descent allocates one stack, not one vector per node pair" begin
        # The leaves' `child_indices_extents` allocates in either search, so this is a
        # comparison rather than an absolute bound: what the stack removes is GeometryOps'
        # per-visited-node-pair `[node_extent(child) for child in getchild(node)]`.
        g = regional_spherical_grid(32, 32)
        a() = QuadtreeCursor(g)
        noop(i, j) = nothing
        cached_dual_depth_first_search(noop, _sph, a(), a())     # compile
        STI.dual_depth_first_search(noop, _sph, a(), a())        # compile
        cached_alloc = @allocated cached_dual_depth_first_search(noop, _sph, a(), a())
        generic_alloc = @allocated STI.dual_depth_first_search(noop, _sph, a(), a())
        @test cached_alloc < generic_alloc
        # a tree that opts out allocates no stack at all
        planar = QuadtreeCursor(planar_regular_grid(16, 16))
        cached_dual_depth_first_search(noop, Extents.intersects, planar, planar)  # compile
        @test (@allocated cached_dual_depth_first_search(noop, Extents.intersects, planar, planar)) ==
              (@allocated STI.dual_depth_first_search(noop, Extents.intersects, planar, planar))
    end

    @testset "children_extent_type is asked of the node, in either spelling" begin
        g = spherical_regular_grid(8, 8)
        onnode = InstanceTraitNode(QuadtreeCursor(g))
        @test children_extent_type(onnode) === GO.UnitSpherical.SphericalCap{Float64}
        @test children_extent_type(QuadtreeCursor(g)) === nothing   # says nothing, so the default
        @test (@inferred CDFS._extent_stack(nothing, onnode, STI.node_extent(onnode))) isa
              Vector{GO.UnitSpherical.SphericalCap{Float64}}
        @test pairs_cached(_sph, onnode, InstanceTraitNode(QuadtreeCursor(g))) ==
              pairs_generic(_sph, QuadtreeCursor(g), QuadtreeCursor(g))
    end

    @testset "children_extent_type carries a per-level extent type" begin
        g = planar_regular_grid(8, 8)
        a = AlternatingExtentNode(QuadtreeCursor(g))
        b = AlternatingExtentNode(QuadtreeCursor(g))
        @test STI.node_extent_is_expensive(a)
        @test children_extent_type(a) isa Type
        # a stack of the declared child type, not of the node's own extent type
        @test (@inferred CDFS._extent_stack(nothing, a, STI.node_extent(a))) isa
              Vector{children_extent_type(a)}
        # and the answers are the ones the plain tree gives
        @test pairs_cached(_boxy_intersects, a, b) ==
              pairs_generic(Extents.intersects, QuadtreeCursor(g), QuadtreeCursor(g))
    end

    @testset "an Action still steers the traversal" begin
        g = spherical_regular_grid(8, 8)
        a = QuadtreeCursor(g); b = QuadtreeCursor(g)
        seen = Tuple{Int, Int}[]
        ret = cached_dual_depth_first_search(_sph, a, b) do i1, i2
            push!(seen, (i1, i2))
            return length(seen) >= 3 ? GO.LoopStateMachine.Action(:full_return, true) : nothing
        end
        @test length(seen) == 3
        @test ret isa GO.LoopStateMachine.Action
        @test ret.name === :full_return
        # and the pairs it did see are the first three the generic search reports
        @test seen == pairs_generic(_sph, a, b)[1:3]
    end
end
