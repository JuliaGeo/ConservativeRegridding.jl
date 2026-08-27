"""
    WeightedDualDepthFirstSearch

A deterministic dual-tree traversal that descends one tree at a time.

When both nodes can be split, the side with the greater [`Trees.split_weight`](@ref)
is opened; ties consistently open the first side.  Opening only one side avoids the
child Cartesian product used by the generic dual-tree walk.  More importantly, the
first leaf reached is held fixed while the opposing tree finishes descending, so its
`child_indices_extents` value is constructed once and carried through that descent.

There is no retained cache or scratch extent stack.  A traversal holds only its
recursive state and, after reaching a leaf, that leaf's inline entries.  The reported
candidate set is the same as `STI.dual_depth_first_search`, but its deterministic order
may differ.
"""
module WeightedDualDepthFirstSearch

import GeometryOps as GO
using GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: @controlflow

import ..Trees: split_weight

export weighted_dual_depth_first_search

# Both the serial traversal and the task frontier use this exact rule.  That makes
# concatenating task results reproduce the serial weighted traversal for every task
# budget.  A leaf can never be selected for descent.
@inline function _split_first(node1, node2)
    leaf1 = STI.isleaf(node1)
    leaf2 = STI.isleaf(node2)
    leaf1 && return false
    leaf2 && return true
    return split_weight(node1) >= split_weight(node2)
end

# GeometryOps uses a radius-NaN cap as the conservative whole-sphere fallback for
# point sets whose mean direction is undefined.  Such a node cannot prune anything.
# The `!(r < π)` spelling catches both that representation and an explicit full sphere.
@inline _is_unprunable_extent(::Any) = false
@inline _is_unprunable_extent(cap::GO.UnitSpherical.SphericalCap) = !(cap.radius < π)
@inline _may_intersect(predicate, extent1, extent2) =
    _is_unprunable_extent(extent1) || _is_unprunable_extent(extent2) ||
    predicate(extent1, extent2)

"""
    weighted_dual_depth_first_search(f, predicate, tree1, tree2)

Call `f(i1, i2)` for every intersecting leaf-index pair in a deterministic,
weight-guided order.

Only one tree descends at each recursive step.  The tree with more estimated leaf work
opens first, according to [`Trees.split_weight`](@ref), and the first side wins ties.
Once either side reaches a leaf, its `child_indices_extents` value is constructed once
and carried while the other side descends.  No grid-sized or traversal-sized extent
cache is retained.

The candidate set and pruning predicate are the same as for
`STI.dual_depth_first_search`; only enumeration order may differ.  `f` may return a
GeometryOps `Action` to steer the traversal.
"""
function weighted_dual_depth_first_search(
    f::F, predicate::P, node1::N1, node2::N2,
) where {F, P, N1, N2}
    return weighted_dual_depth_first_search(
        f, predicate, node1, STI.node_extent(node1), node2, STI.node_extent(node2))
end

# `extent1` and `extent2` are a precondition: each must equal `node_extent(node)`.
function weighted_dual_depth_first_search(
    f::F, predicate::P, node1::N1, extent1::E1, node2::N2, extent2::E2,
) where {F, P, N1, E1, N2, E2}
    return weighted_dual_depth_first_search(
        f, predicate, node1, extent1, node2, extent2, nothing, nothing)
end

function weighted_dual_depth_first_search(
    f::F, predicate::P,
    node1::N1, extent1::E1, node2::N2, extent2::E2,
    entries1::C1, entries2::C2,
) where {F, P, N1, E1, N2, E2, C1, C2}
    leaf1 = STI.isleaf(node1)
    leaf2 = STI.isleaf(node2)

    if leaf1 && leaf2
        # `@something` is lazy: carried entries suppress reconstruction completely.
        cie1 = @something entries1 STI.child_indices_extents(node1)
        cie2 = @something entries2 STI.child_indices_extents(node2)
        for (i1, cell_extent1) in cie1
            for (i2, cell_extent2) in cie2
                predicate(cell_extent1, cell_extent2) && @controlflow f(i1, i2)
            end
        end
    elseif _split_first(node1, node2)
        # If node2 is already a leaf, derive its entries before opening node1 and carry
        # the value through every surviving child.  Otherwise `entries2` is `nothing`.
        cie2 = leaf2 ? (@something entries2 STI.child_indices_extents(node2)) : nothing
        for child1 in STI.getchild(node1)
            child_extent1 = STI.node_extent(child1)
            if _may_intersect(predicate, child_extent1, extent2)
                @controlflow weighted_dual_depth_first_search(
                    f, predicate, child1, child_extent1, node2, extent2, nothing, cie2)
            end
        end
    else
        cie1 = leaf1 ? (@something entries1 STI.child_indices_extents(node1)) : nothing
        for child2 in STI.getchild(node2)
            child_extent2 = STI.node_extent(child2)
            if _may_intersect(predicate, extent1, child_extent2)
                @controlflow weighted_dual_depth_first_search(
                    f, predicate, node1, extent1, child2, child_extent2, cie1, nothing)
            end
        end
    end
    return nothing
end

end
