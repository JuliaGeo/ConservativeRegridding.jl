module MultithreadedDualDepthFirstSearch

using GeometryOps: SpatialTreeInterface as STI
import GeometryOps as GO
import StableTasks

import ..Trees: split_weight
import ..CachedDualDepthFirstSearch: cached_dual_depth_first_search

#=
The dual-tree search is parallelized by a *budget frontier*: a serial pass splits the
root pair into at least `nchunks` independent node pairs, always splitting the heaviest
by estimated work, then runs the serial dual DFS on each in its own task.  The results
are concatenated in DFS pre-order, so the output matches the serial search exactly.
=#

# Whole-sphere caps, including the all-NaN cap GeometryOps returns for a pole-spanning
# point set: `!(radius < π)` also catches NaN, which must never reach the weights.
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

# Work under a node pair: the area both cover, times the cell density each brings to it.
# The generic fallback has no notion of overlap, so it multiplies the subtree sizes.
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

How far past `nchunks` [`frontier`](@ref) may split while its estimate still calls the
frontier unbalanced.  Coarse extents can tie genuinely unequal pairs, and this caps what
the resulting overshoot costs in spawns.
"""
const MAX_EXTRA_PAIRS = 512

# Binary max-heap over (weight, item), typed on the weight so comparisons stay concrete
# even though the items are heterogeneous node pairs.
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

Split the root node pair into at least `nchunks` independent node pairs by repeatedly
splitting the heaviest one, and return them in DFS pre-order.

Splitting continues past `nchunks` while the heaviest pair still claims more than a
`1/nchunks` share of the estimated work, and stops at `nchunks + MAX_EXTRA_PAIRS`.

A pair is splittable unless BOTH sides are leaves, so a leaf facing a large subtree
descends the large side only.  Children failing `predicate` are pruned exactly as the
serial dual DFS prunes them, so the serial search over the returned pairs visits every
candidate pair exactly once.

A split never raises the estimate: children whose weights sum past their parent's are
scaled back to it, so `pair_weight` models that saturate under splitting (loose bounding
caps, subtree sizes a tree reports as its whole-grid total) still leave a frontier whose
heaviest pair shrinks with every split instead of stalling the share test.
"""
function frontier(predicate::P, root1, root2; nchunks::Int) where {P}
    e1 = STI.node_extent(root1)
    e2 = STI.node_extent(root2)
    # `Any` because the pairs are heterogeneously typed - not trim-compatible yet.
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
    cw = Float64[]; cx = Any[]          # one split's surviving children, then filed
    while !isempty(ws)
        npairs = length(ws) + length(done)
        npairs < maxpairs || break
        # Below the chunk count always split; above it, only while the heaviest
        # pair is over its even share of the estimate.
        npairs < nchunks || ws[1] * nchunks > total || break
        w0 = ws[1]
        (n1, a1, n2, a2, key) = _heap_pop!(ws, xs)
        empty!(cw); empty!(cx)
        if STI.isleaf(n1)                        # descend side 2 only
            k = 0
            for c2 in STI.getchild(n2)
                k += 1
                f2 = STI.node_extent(c2)
                predicate(a1, f2) || continue
                _gather!(cw, cx, n1, a1, c2, f2, vcat(key, k))
            end
        elseif STI.isleaf(n2)                    # descend side 1 only
            k = 0
            for c1 in STI.getchild(n1)
                k += 1
                f1 = STI.node_extent(c1)
                predicate(f1, a2) || continue
                _gather!(cw, cx, c1, f1, n2, a2, vcat(key, k))
            end
        else                                     # child cross product, side 1 major
            ch1 = collect(STI.getchild(n1)); ch2 = collect(STI.getchild(n2))
            ce1 = map(STI.node_extent, ch1)      # child extents derived once each,
            ce2 = map(STI.node_extent, ch2)      # not once per pair in the product
            k = 0
            for i in eachindex(ch1), j in eachindex(ch2)
                k += 1
                predicate(ce1[i], ce2[j]) || continue
                _gather!(cw, cx, ch1[i], ce1[i], ch2[j], ce2[j], vcat(key, k))
            end
        end
        # Conserve the estimate: a split may only redistribute its parent's weight.
        # A zero-weight parent is the estimate being wrong, so its children keep theirs.
        sw = sum(cw; init = 0.0)
        scale = sw > w0 > 0 ? w0 / sw : 1.0
        total -= w0
        for i in eachindex(cw)
            w = cw[i] * scale
            total += w
            p = cx[i]
            if STI.isleaf(p[1]) && STI.isleaf(p[3])
                push!(done, p)
            else
                _heap_push!(ws, xs, w, p)
            end
        end
    end
    pairs = Any[]
    append!(pairs, xs); append!(pairs, done)
    sort!(pairs, by = p -> p[5], lt = _keyless)
    return pairs
end

# Files one surviving child pair and its raw weight for the caller to scale and heap.
@inline function _gather!(cw, cx, n1, e1, n2, e2, key)
    push!(cw, pair_weight(n1, e1, n2, e2))
    push!(cx, (n1, e1, n2, e2, key))
    return nothing
end

function _inner_dfs_f(predicate::P, node1::N1, node2::N2) where {P, N1, N2}
    ret = Tuple{Int, Int}[]
    # Each frontier pair is descended by its own task, so each gets its own scratch
    # stack; nothing is shared across tasks.
    cached_dual_depth_first_search(predicate, node1, node2) do i1, i2
        push!(ret, (i1, i2))
    end
    return ret
end

# One dynamic dispatch per task lands here; the spawned closure is concrete from there on.
function _spawn_pair(inner::IF, predicate::P, node1::N1, node2::N2) where {IF, P, N1, N2}
    return StableTasks.@spawn $inner($predicate, $node1, $node2)
end

"""
    multithreaded_dual_query!(result, predicate, node1, node2; chunks_per_thread = 8)

Find every leaf-index pair `(i1, i2)` whose extents satisfy `predicate`, in
parallel, writing them to an emptied `result` in the same order as a serial
`STI.dual_depth_first_search` over the same trees.

`chunks_per_thread` sets the task budget: the traversal is split into at least
`Threads.nthreads() * chunks_per_thread` node pairs (see [`frontier`](@ref)), one task
each.  Raising it costs more spawns but tolerates more skew between pairs; it cannot
change the result.  Work estimates come from `Trees.split_weight`.
"""
function multithreaded_dual_query!(
    result::Vector{Tuple{Int, Int}},
    predicate::P, node1::N1, node2::N2;
    chunks_per_thread::Int = 8,
) where {P, N1, N2}
    empty!(result)
    nchunks = max(1, Threads.nthreads() * chunks_per_thread)
    pairs = frontier(predicate, node1, node2; nchunks)
    tasks = map(pairs) do pair
        _spawn_pair(_inner_dfs_f, predicate, pair[1], pair[3])
    end
    # Fetch and append in frontier order, preserving the serial DFS order.  The caller
    # may retain `result` in task-local scratch across repeated block builds.
    for task in tasks
        append!(result, fetch(task)::Vector{Tuple{Int, Int}})
    end
    return result
end

function multithreaded_dual_query(
    predicate::P, node1::N1, node2::N2;
    chunks_per_thread::Int = 8,
) where {P, N1, N2}
    return multithreaded_dual_query!(
        Tuple{Int, Int}[], predicate, node1, node2; chunks_per_thread)
end

# Back-compat: the two parallelize closures are no longer consulted.
multithreaded_dual_query(predicate, parallelize1, parallelize2, node1, node2; kwargs...) =
    multithreaded_dual_query(predicate, node1, node2; kwargs...)

export multithreaded_dual_query, multithreaded_dual_query!
end
