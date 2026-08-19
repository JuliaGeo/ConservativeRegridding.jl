module MultithreadedDualDepthFirstSearch

using GeometryOps: SpatialTreeInterface as STI
import GeometryOps as GO
import StableTasks

import ..Trees: split_weight

#=
The dual-tree search is parallelized by a *budget frontier*: a short serial
descent splits the root pair into at least `nchunks` independent node pairs,
then one task per pair runs the plain serial dual DFS on it.

Phase 1 keeps a max-heap of node pairs keyed by `pair_weight` (an estimate of
the work under the pair) and repeatedly splits the heaviest pair, so the
frontier is sized by estimated work rather than by tree depth. It stops once
`nchunks` pairs exist and none of them still claims more than a `1/nchunks`
share of the estimated work, which keeps a pair the estimate cannot tell apart
from its neighbours - but that really holds a quarter of the traversal - from
surviving as a single task. Phase 2 spawns the tasks and concatenates their
results in DFS pre-order, which makes the output bit-identical - and
order-identical - to a full serial dual DFS.
=#

# A cap that covers the whole sphere, including the all-NaN cap GeometryOps
# returns for a point set with no usable centre - a grid spanning both poles
# has one. It localises nothing, and NaN must not reach the weights: a NaN
# never compares greater, so a NaN-weighted pair sinks to the bottom of the
# max-heap and is never split.
@inline _is_global(c::GO.UnitSpherical.SphericalCap) = !(c.radius < π)

@inline _cap_area(c::GO.UnitSpherical.SphericalCap) =
    _is_global(c) ? 4π : 2π * (1 - cos(c.radius))

# Area of the overlap of two caps: exact when they are disjoint or nested,
# linearly ramped in between. Only feeds `pair_weight`, so it may be rough.
@inline function _overlap_area(a::GO.UnitSpherical.SphericalCap, b::GO.UnitSpherical.SphericalCap)
    _is_global(a) && return _cap_area(b)      # a whole-sphere cap contains the other
    _is_global(b) && return _cap_area(a)
    d = GO.UnitSpherical.spherical_distance(a.point, b.point)
    d >= a.radius + b.radius && return 0.0
    small = a.radius <= b.radius ? a : b
    d <= abs(a.radius - b.radius) && return _cap_area(small)
    return _cap_area(small) * clamp((a.radius + b.radius - d) / (2 * min(a.radius, b.radius)), 0.0, 1.0)
end

# Work estimate for a node pair: the area both nodes cover, times the cell
# density each contributes there. The generic fallback has no notion of
# overlap, so it just multiplies the two subtree sizes.
@inline function pair_weight(n1, e1::GO.UnitSpherical.SphericalCap, n2, e2::GO.UnitSpherical.SphericalCap)
    A = _overlap_area(e1, e2)
    A <= 0 && return 0.0
    d1 = split_weight(n1) / max(_cap_area(e1), eps())
    d2 = split_weight(n2) / max(_cap_area(e2), eps())
    return A * (sqrt(d1) + sqrt(d2))^2
end
@inline pair_weight(n1, e1, n2, e2) = float(split_weight(n1)) * float(split_weight(n2))

"""
    MAX_EXTRA_PAIRS

How many pairs beyond `nchunks` [`frontier`](@ref) may split into while its
work estimate still calls the frontier unbalanced. Extents coarse enough to tie
genuinely unequal pairs (a lat-lon block's bounding cap can be a whole
hemisphere) leave the frontier no choice but to overshoot, and this bounds what
the overshoot costs in spawns.
"""
const MAX_EXTRA_PAIRS = 512

# Binary max-heap over (weight, item), typed on the weight so the comparisons
# stay concrete even though the items are heterogeneous node pairs.
@inline function _heap_up!(ws::Vector{Float64}, xs::Vector{Any}, i::Int)
    while i > 1
        p = i >> 1
        ws[p] >= ws[i] && break
        ws[p], ws[i] = ws[i], ws[p]; xs[p], xs[i] = xs[i], xs[p]
        i = p
    end
end
@inline function _heap_down!(ws::Vector{Float64}, xs::Vector{Any}, i::Int)
    n = length(ws)
    while true
        l = 2i; r = l + 1; m = i
        l <= n && ws[l] > ws[m] && (m = l)
        r <= n && ws[r] > ws[m] && (m = r)
        m == i && return
        ws[m], ws[i] = ws[i], ws[m]; xs[m], xs[i] = xs[i], xs[m]
        i = m
    end
end
@inline function _heap_push!(ws::Vector{Float64}, xs::Vector{Any}, w::Float64, x)
    push!(ws, w); push!(xs, x); _heap_up!(ws, xs, length(ws))
end
@inline function _heap_pop!(ws::Vector{Float64}, xs::Vector{Any})
    x = xs[1]
    ws[1] = ws[end]; xs[1] = xs[end]; pop!(ws); pop!(xs)
    isempty(ws) || _heap_down!(ws, xs, 1)
    return x
end

# Lexicographic order on split keys, i.e. DFS pre-order on the pair tree.
@inline _keyless(a::Vector{Int}, b::Vector{Int}) = begin
    @inbounds for k in 1:min(length(a), length(b))
        a[k] == b[k] || return a[k] < b[k]
    end
    length(a) < length(b)
end

"""
    frontier(predicate, root1, root2; nchunks) -> Vector

Split the root node pair into at least `nchunks` independent node pairs by
repeatedly splitting the heaviest one, and return them in DFS pre-order.

Splitting continues past `nchunks` while the heaviest pair still claims more
than a `1/nchunks` share of the frontier's estimated work, and stops for good
at `nchunks + MAX_EXTRA_PAIRS` pairs. `pair_weight` only orders the queue, so
a coarse extent can tie pairs whose real work differs by orders of magnitude;
the share test keeps the budget on the pairs that are still too big instead
of spending it on whichever tied pair the heap happened to surface.

A pair is splittable unless BOTH sides are leaves, so a leaf facing a large
subtree is handled by descending the large side only. Children failing
`predicate` are dropped exactly as the serial dual DFS would prune them, so
running the serial search on every returned pair visits each candidate pair
exactly once.
"""
function frontier(predicate::P, root1, root2; nchunks::Int) where {P}
    e1 = STI.node_extent(root1)
    e2 = STI.node_extent(root2)
    ws = Float64[]; xs = Any[]          # splittable pairs, as a max-heap
    done = Any[]                        # leaf/leaf pairs, cannot split further
    root = (root1, e1, root2, e2, Int[])
    total = pair_weight(root1, e1, root2, e2)   # estimated work in the frontier
    if STI.isleaf(root1) && STI.isleaf(root2)
        push!(done, root)
    else
        _heap_push!(ws, xs, total, root)
    end
    maxpairs = nchunks + MAX_EXTRA_PAIRS
    while !isempty(ws)
        npairs = length(ws) + length(done)
        npairs < maxpairs || break
        # Below the chunk count always split; above it, only while the heaviest
        # pair is over its even share of the estimate.
        npairs < nchunks || ws[1] * nchunks > total || break
        total -= ws[1]
        (n1, a1, n2, a2, key) = _heap_pop!(ws, xs)
        if STI.isleaf(n1)                        # descend side 2 only
            k = 0
            for c2 in STI.getchild(n2)
                k += 1
                f2 = STI.node_extent(c2)
                predicate(a1, f2) || continue
                total += _emit!(ws, xs, done, n1, a1, c2, f2, vcat(key, k))
            end
        elseif STI.isleaf(n2)                    # descend side 1 only
            k = 0
            for c1 in STI.getchild(n1)
                k += 1
                f1 = STI.node_extent(c1)
                predicate(f1, a2) || continue
                total += _emit!(ws, xs, done, c1, f1, n2, a2, vcat(key, k))
            end
        else                                     # child cross product, side 1 major
            ch1 = collect(STI.getchild(n1)); ch2 = collect(STI.getchild(n2))
            ce1 = map(STI.node_extent, ch1)      # child extents derived once each,
            ce2 = map(STI.node_extent, ch2)      # not once per pair in the product
            k = 0
            for i in eachindex(ch1), j in eachindex(ch2)
                k += 1
                predicate(ce1[i], ce2[j]) || continue
                total += _emit!(ws, xs, done, ch1[i], ce1[i], ch2[j], ce2[j], vcat(key, k))
            end
        end
    end
    pairs = Any[]
    append!(pairs, xs); append!(pairs, done)
    sort!(pairs, by = p -> p[5], lt = _keyless)
    return pairs
end

# Files the pair on the heap or, for a leaf/leaf pair, on `done`, and returns
# its weight so the caller can keep the frontier's work total up to date.
@inline function _emit!(ws, xs, done, n1, e1, n2, e2, key)
    w = pair_weight(n1, e1, n2, e2)
    if STI.isleaf(n1) && STI.isleaf(n2)
        push!(done, (n1, e1, n2, e2, key))
    else
        _heap_push!(ws, xs, w, (n1, e1, n2, e2, key))
    end
    return w
end

function _inner_dfs_f(predicate::P, node1::N1, node2::N2) where {P, N1, N2}
    ret = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate, node1, node2) do i1, i2
        push!(ret, (i1, i2))
    end
    return ret
end

# The frontier's pairs are heterogeneously typed, so one dynamic dispatch per
# task lands here and the spawned closure is concrete from there on.
function _spawn_pair(inner::IF, predicate::P, node1::N1, node2::N2) where {IF, P, N1, N2}
    return StableTasks.@spawn $inner($predicate, $node1, $node2)
end

"""
    multithreaded_dual_query(predicate, node1, node2; chunks_per_thread = 8) -> Vector{Tuple{Int, Int}}

Find every leaf-index pair `(i1, i2)` whose extents satisfy `predicate`, in
parallel, returning them in the same order as a serial
`STI.dual_depth_first_search` over the same trees.

`chunks_per_thread` sets the task budget: the traversal is split into at
least `Threads.nthreads() * chunks_per_thread` node pairs (see
[`frontier`](@ref)), one task each. Raising it costs more spawns but tolerates
more skew between pairs; lowering it does the reverse. It cannot change the
result.

Per-node work estimates come from `Trees.split_weight`, which is the hook to
define if a tree type balances badly.
"""
function multithreaded_dual_query(
    predicate::P, node1::N1, node2::N2;
    chunks_per_thread::Int = 8,
) where {P, N1, N2}
    nchunks = max(1, Threads.nthreads() * chunks_per_thread)
    pairs = frontier(predicate, node1, node2; nchunks)
    tasks = Any[]
    sizehint!(tasks, length(pairs))
    for p in pairs
        push!(tasks, _spawn_pair(_inner_dfs_f, predicate, p[1], p[3]))
    end
    results = Vector{Tuple{Int, Int}}[fetch(t)::Vector{Tuple{Int, Int}} for t in tasks]
    # `init` would cost `vcat`'s pre-sized specialisation, making this quadratic in the task count.
    return isempty(results) ? Tuple{Int, Int}[] : reduce(vcat, results)
end

# Back-compat: the two parallelize closures are no longer consulted.
multithreaded_dual_query(predicate, parallelize1, parallelize2, node1, node2; kwargs...) =
    multithreaded_dual_query(predicate, node1, node2; kwargs...)

export multithreaded_dual_query
end
