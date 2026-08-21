"""
    CachedDualDepthFirstSearch

A dual depth-first search that caches the child extents it derives, for trees whose
`STI.node_extent` *computes* an extent instead of reading a stored one.

## Why this lives here and not in GeometryOps

`STI.dual_depth_first_search` is deliberately generic: it is written for trees where
`node_extent` is a field read, which is the case for every tree GeometryOps ships
(RTree, STRtree, `NaturalIndex`, `FlatNoTree`).  For those, caching is pure overhead,
and the generic search is right to stay as it is.

Regridding is where the other kind of tree shows up.  A cursor over a curvilinear grid
(and, downstream, a DGGS cell cursor) has no extents to read: every
`node_extent` call walks the node's boundary cells and folds them into a bounding cap.
The plain dual search calls `node_extent` on each of node2's children once per child of
node1, so a `4 x 4` node pair derives 4 extents 4 times over. This module derives them
once, from a scratch stack that the whole descent shares:

- a node that opts into [`STI.node_extent_is_expensive`](@ref) has its children's extents
  pushed once, read `nchild(node1)` times, and popped on the way back out;
- the stack is allocated at most once per level whose child-extent type changes, and
  reused at every node below that, so a descent allocates a bounded amount rather than
  one vector per visited node pair;
- a tree that stores its extents opts out by *doing nothing*: the trait defaults to
  false, the branch folds at compile time, and the search compiles to exactly the
  uncached loop.

The traversal order, the pruning, and the `(i1, i2)` pairs it reports are identical to
`STI.dual_depth_first_search`'s in every case; only the number of `node_extent` calls
differs.

The public entry point is [`cached_dual_depth_first_search`](@ref), and
[`children_extent_type`](@ref) is the one trait a tree may need to define beyond the
SpatialTreeInterface ones.
"""
module CachedDualDepthFirstSearch

import GeometryOps as GO
using GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: Action, @controlflow

export cached_dual_depth_first_search

"""
    children_extent_type(node)::Union{Type, Nothing}

Return the type `STI.node_extent` gives for `node`'s *children*, or `nothing` — the
default — to say it is the same type as `node`'s own extent.

Only [`cached_dual_depth_first_search`](@ref), on a tree that opts into
`STI.node_extent_is_expensive`, consults this: it caches a node's child extents in a
typed scratch stack, and needs the element type before it touches a child.  Siblings
must agree, but a tree whose extent type changes with depth can say so here and still
be traversed.

## Implementation notes

Define it on the node — `children_extent_type(node::MyNode) = MyChildExtent` — or on its
type if that reads better, `children_extent_type(::Type{MyNode}) = MyChildExtent`; the
node method falls back to the type one.  Either way the answer is a constant per node
type, so it folds away.
"""
children_extent_type(node) = children_extent_type(typeof(node))
children_extent_type(::Type{<:Any}) = nothing

# The descent's scratch stack of derived child extents, or `nothing` to mean "call
# `node_extent` in the loop".  `node_extent_is_expensive` is a function of `N` alone,
# so the branch folds and only one arm is compiled.
@inline _extent_stack(stack, node::N, ::E) where {N, E} =
    STI.node_extent_is_expensive(N) ? _as_stack(stack, _child_extent_type(node, E), node) : nothing

# Siblings share a type, so one stack serves a node's children; a level whose children
# type differs from the stack it inherited starts its own.  Dispatch picks the arm, so
# the reuse path never calls `nchild` - it can be costly.
@inline _as_stack(stack::Vector{E}, ::Type{E}, node) where {E} = stack
@inline _as_stack(stack, ::Type{E}, node) where {E} = sizehint!(Vector{E}(), STI.nchild(node))

# An inherited stack is appended to above whatever the ancestors left on it; a stack
# this level started for itself begins empty.
@inline _stack_base(::Nothing, stack) = 0
@inline _stack_base(stack2, stack) = stack2 === stack ? length(stack2) : 0

# asked of the node, not its type - the answer is constant per type either way
@inline _child_extent_type(node, ::Type{E}) where {E} = something(children_extent_type(node), E)

# Hand an ancestor's stack on through levels that do not need one themselves.
@inline _carry(::Nothing, stack) = stack
@inline _carry(stack, _) = stack

@inline _fill_child_extents!(::Nothing, node) = nothing
@inline function _fill_child_extents!(stack, node)
    for child in STI.getchild(node)
        push!(stack, STI.node_extent(child))
    end
    return nothing
end

@inline _child_extent(::Nothing, base, child, i) = STI.node_extent(child)
@inline _child_extent(stack, base, child, i) = stack[base + i]

"""
    cached_dual_depth_first_search(f, predicate, tree1, tree2)

Run a dual depth-first search over `tree1` and `tree2`, calling `f(i1, i2)` for every
leaf-level index pair whose extents satisfy `predicate`.

This reports exactly the pairs `STI.dual_depth_first_search(f, predicate, tree1, tree2)`
reports, in exactly that order, and prunes exactly the same branches.  The difference is
that a node opting into `STI.node_extent_is_expensive` has its children's extents derived
once and cached on a scratch stack shared by the whole descent, instead of re-derived once
per opposing child.  See the [`CachedDualDepthFirstSearch`](@ref) module docstring for why
this variant lives in ConservativeRegridding rather than in GeometryOps.

`f` may return an `Action` to steer the traversal, exactly as in
`STI.dual_depth_first_search`.

Trees whose child extents are of a different type than their own must say so via
[`children_extent_type`](@ref).
"""
function cached_dual_depth_first_search(f::F, predicate::P, node1::N1, node2::N2) where {F, P, N1, N2}
    return cached_dual_depth_first_search(
        f, predicate, node1, STI.node_extent(node1), node2, STI.node_extent(node2))
end

# `extent1` and `extent2` are a precondition, not a hint: they must equal
# `node_extent(node1)` and `node_extent(node2)`.
function cached_dual_depth_first_search(
    f::F, predicate::P, node1::N1, extent1::E1, node2::N2, extent2::E2
) where {F, P, N1, E1, N2, E2}
    return cached_dual_depth_first_search(f, predicate, node1, extent1, node2, extent2, nothing)
end

function cached_dual_depth_first_search(
    f::F, predicate::P, node1::N1, extent1::E1, node2::N2, extent2::E2, stack::S
) where {F, P, N1, E1, N2, E2, S}
    leaf1 = STI.isleaf(node1)
    leaf2 = STI.isleaf(node2)
    if leaf1 && leaf2
        # bound once each - `cie_2` would otherwise be rebuilt per cell of node1
        cie_1 = STI.child_indices_extents(node1)
        cie_2 = STI.child_indices_extents(node2)
        for (i1, cell_extent1) in cie_1
            for (i2, cell_extent2) in cie_2
                if predicate(cell_extent1, cell_extent2)
                    @controlflow f(i1, i2)
                end
            end
        end
    elseif leaf1 # node2 is not a leaf, node1 is - recurse further into node2
        for child in STI.getchild(node2)
            child_extent = STI.node_extent(child)
            if predicate(extent1, child_extent)
                @controlflow cached_dual_depth_first_search(
                    f, predicate, node1, extent1, child, child_extent, stack)
            end
        end
    elseif leaf2 # node1 is not a leaf, node2 is - recurse further into node1
        for child in STI.getchild(node1)
            child_extent = STI.node_extent(child)
            if predicate(child_extent, extent2)
                @controlflow cached_dual_depth_first_search(
                    f, predicate, child, child_extent, node2, extent2, stack)
            end
        end
    else # neither node is a leaf, recurse into both children
        # node2's child extents go on the shared stack and come off again on the way
        # out, so the descent allocates no buffer per visited node pair.  node1's are
        # derived once each by the outer loop already, so they need no cache.
        stack2 = _extent_stack(stack, node2, extent2)
        base = _stack_base(stack2, stack)
        _fill_child_extents!(stack2, node2)
        child_stack = _carry(stack2, stack)
        for child1 in STI.getchild(node1)
            child_extent1 = STI.node_extent(child1)
            i2 = 0
            for child2 in STI.getchild(node2)
                i2 += 1
                child_extent2 = _child_extent(stack2, base, child2, i2)
                if predicate(child_extent1, child_extent2)
                    @controlflow cached_dual_depth_first_search(
                        f, predicate, child1, child_extent1, child2, child_extent2, child_stack)
                end
            end
        end
        stack2 === nothing || resize!(stack2, base)
    end
    # never the scratch stack `resize!` hands back - only a propagating `Action` leaves
    # this function with a value
    return nothing
end

end
