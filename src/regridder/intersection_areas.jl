import GeometryOps as GO
using GeometryOps: SpatialTreeInterface as STI

# Intersection-operator interface: the operator drives assembly via one trait
# (`IntersectionReturnStyle`) + two hooks (`work_items`, `output_matrix_size`),
# each with a default reproducing the built-in area computation.

"""
    abstract type IntersectionReturnStyle

Trait describing how an intersection operator delivers its contribution for a
single unit of work during sparse-matrix assembly.  The trait is resolved once,
at the top of [`intersection_areas`](@ref) via the accessor
`IntersectionReturnStyle(op)`, and the resulting value is threaded through the
parallel assembly so the trait lookup never happens in the hot loop.

Subtypes:
- [`OutOfPlaceSingleResult`](@ref): the kernel is `op(src_cell, dst_cell) -> area`
  and the driver stores the COO triplet.
- [`InPlace`](@ref): the kernel is
  `op(rows, cols, vals, item, src_tree, dst_tree) -> nothing` and pushes its own
  COO contributions.

The default accessor returns [`OutOfPlaceSingleResult`](@ref), so any operator
that does not override it behaves like [`DefaultIntersectionOperator`](@ref).
"""
abstract type IntersectionReturnStyle end

"""
    OutOfPlaceSingleResult <: IntersectionReturnStyle

Return style for operators that compute one scalar per candidate pair.  The
kernel is `op(src_cell, dst_cell) -> area::Real`; the driver stores the COO
triplet `(dst_index, src_index, area)` whenever `area > 0`.  This is the default
style and the one used by [`DefaultIntersectionOperator`](@ref).
"""
struct OutOfPlaceSingleResult <: IntersectionReturnStyle end

"""
    InPlace <: IntersectionReturnStyle

Return style for operators that push a block of COO contributions per work item.
The kernel is `op(rows, cols, vals, item, src_tree, dst_tree) -> nothing` and is
responsible for `push!`ing onto `rows`/`cols`/`vals` itself.  Used by
block-emitting operators such as the ClimaCore spectral-element assemblers.
"""
struct InPlace <: IntersectionReturnStyle end

"""
    IntersectionReturnStyle(op) -> IntersectionReturnStyle

Return the [`IntersectionReturnStyle`](@ref) of intersection operator `op`.
Defaults to [`OutOfPlaceSingleResult`](@ref); operators that assemble blocks in
place override this to return [`InPlace`](@ref).
"""
IntersectionReturnStyle(::Any) = OutOfPlaceSingleResult()

"""
    work_items(op, candidate_pairs) -> items

Map the candidate `(src_index, dst_index)` pairs to the collection of work units
that the parallel assembly iterates over.  Each item is handed to the operator:
destructured as `(src_index, dst_index)` for [`OutOfPlaceSingleResult`](@ref), or
passed through verbatim for [`InPlace`](@ref).

Defaults to one work unit per candidate pair.  Override it to change the parallel
granularity — e.g. to group all of an element's candidate cells into a single
work unit.
"""
work_items(::Any, candidate_pairs) = candidate_pairs

"""
    output_matrix_size(op, src_tree, dst_tree) -> (nrows, ncols)

Return the `(nrows, ncols)` shape of the sparse matrix that
[`intersection_areas`](@ref) assembles for operator `op`.  Defaults to
`(prod(ncells(dst_tree)), prod(ncells(src_tree)))` — destination cells as rows,
source cells as columns.  Operators whose row/column counts differ from the cell
counts (e.g. spectral-element node counts) override this.
"""
output_matrix_size(::Any, src_tree, dst_tree) =
    (prod(Trees.ncells(dst_tree)), prod(Trees.ncells(src_tree)))

# If the root tree is a `WithParallelizePolicy`, route the dual-DFS's
# `(node, extent)` query through the user policy; otherwise fall back to the
# default `should_parallelize` dispatch. The wrapper is *not* a dispatch axis
# on `should_parallelize` — detecting it here keeps the dispatch graph simple.
@inline function _build_parallelize_closure(tree)
    if tree isa Trees.WithParallelizePolicy
        let inner = tree.tree, p = tree.policy
            return (node, extent) -> p(inner, node, extent)
        end
    else
        return (node, extent) -> Trees.should_parallelize(node, extent)
    end
end

function get_all_candidate_pairs(threaded::True, predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    # TODO: Threaded dual dfs via chunking.
    # For now this is just serial, and is the big bottleneck for larger grids.
    # First, run the dual depth first search to get all candidate pairs of
    # cells that may intersect.
    par_src = _build_parallelize_closure(src_tree)
    par_dst = _build_parallelize_closure(dst_tree)
    candidate_idxs = multithreaded_dual_query(predicate_f, par_src, par_dst, src_tree, dst_tree) # from utils/MultithreadedDualDepthFirstSearch.jl
    return candidate_idxs
end

function get_all_candidate_pairs(threaded::False, predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    candidate_idxs = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate_f, src_tree, dst_tree) do i1, i2
        push!(candidate_idxs, (i1, i2))
    end
    return candidate_idxs
end

# Shared parallel COO assembly. `style` (the resolved `IntersectionReturnStyle`)
# is threaded through so the trait is never looked up in the hot loop.

# Run the operator for one work item and store its COO contribution(s).
@inline function _run_and_store!(::OutOfPlaceSingleResult, op, rows, cols, vals, (i1, i2), src_tree, dst_tree)
    p1 = Trees.getcell(src_tree, i1)
    p2 = Trees.getcell(dst_tree, i2)
    area_of_intersection = op(p1, p2)
    if area_of_intersection > 0
        push!(rows, i2)   # row = destination index
        push!(cols, i1)   # col = source index
        push!(vals, area_of_intersection)
    end
    return nothing
end

@inline function _run_and_store!(::InPlace, op, rows, cols, vals, item, src_tree, dst_tree)
    op(rows, cols, vals, item, src_tree, dst_tree)   # the operator stores in place
    return nothing
end

# One chunk of work items → its COO triplets. `style` is passed in, not re-resolved.
function _assemble_chunk(style, op, items, src_tree, dst_tree)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for item in items
        _run_and_store!(style, op, rows, cols, vals, item, src_tree, dst_tree)
    end
    return rows, cols, vals
end

# `True` chunks/spawns, `False` runs one chunk. `$`-interpolation keeps the
# spawned tasks type-stable (concrete `style`/`op`/trees, no boxing).
function _parallel_coo(style, op, items, src_tree, dst_tree, ::True; npartitions, progress)
    # Partition the list of work items,
    partitions = ChunkSplitters.chunks(items; n = npartitions)
    if progress
        progress_meter = ProgressMeter.Progress(length(partitions); desc = "Computing intersection areas")
    end
    # and assemble the COO triplets for each partition in parallel.
    # This is a bit oversubscribed though I guess.  But Julia's dynamic
    # scheduler should handle it fine.
    result_tasks = [
        StableTasks.@spawn begin
            ret = _assemble_chunk($style, $op, partition, $src_tree, $dst_tree)
            $(progress ? :(ProgressMeter.next!(progress_meter)) : :())
            ret
        end
        for partition in partitions
    ]
    # Fetch the results of `result_tasks`
    all_results = map(fetch, result_tasks)
    # Concatenate the per-chunk COO vectors into single vectors, in partition order.
    rows = reduce(vcat, getindex.(all_results, 1))
    cols = reduce(vcat, getindex.(all_results, 2))
    vals = reduce(vcat, getindex.(all_results, 3))
    return rows, cols, vals
end

_parallel_coo(style, op, items, src_tree, dst_tree, ::False; kwargs...) =
    _assemble_chunk(style, op, items, src_tree, dst_tree)

"""
    intersection_areas(manifold, threaded, dst_tree, src_tree;
                       intersection_operator = DefaultIntersectionOperator(manifold),
                       npartitions = Threads.nthreads() * 4, progress = false)

Assemble the sparse intersection matrix between `src_tree` and `dst_tree` on
`manifold`.

This is the lower-level assembly entry that intersection operators plug into;
most users go through [`Regridder`](@ref)`(…; intersection_operator = …)`, which
calls this.

The build is driven by three dispatched seams on `intersection_operator`, each
with a default reproducing the built-in area computation:
- [`IntersectionReturnStyle`](@ref)`(op)` — how each work item's contribution is stored.
- [`work_items`](@ref)`(op, candidate_pairs)` — the units of work the parallel loop iterates.
- [`output_matrix_size`](@ref)`(op, src_tree, dst_tree)` — the `(nrows, ncols)` matrix shape.

`threaded` is a `GeometryOpsCore.BoolsAsTypes` (`True()`/`False()`); pass
`booltype(::Bool)` to convert.  When threaded, candidate work items are
partitioned into `npartitions` chunks assembled on separate tasks.
"""
function intersection_areas(
        manifold::M, threaded::BoolsAsTypes, dst_tree, src_tree;
        intersection_operator = DefaultIntersectionOperator(manifold),
        npartitions::Int = Threads.nthreads() * 4,
        progress = false,
    ) where {M <: Manifold}

    # Resolve the return-style trait once, here, and thread `style` through everything.
    style = IntersectionReturnStyle(intersection_operator)

    predicate_f = if M <: Spherical
        GO.UnitSpherical._intersects
    else
        Extents.intersects
    end

    # First, run the dual depth first search to get all candidate pairs of
    # cells that may intersect.
    candidate_pairs = get_all_candidate_pairs(threaded, predicate_f, src_tree, dst_tree)

    # Map candidate pairs → work units (default: one pair per unit), assemble the
    # COO triplets (in parallel or serially), and build the sparse matrix.
    items = work_items(intersection_operator, candidate_pairs)
    nrows, ncols = output_matrix_size(intersection_operator, src_tree, dst_tree)
    rows, cols, vals = _parallel_coo(
        style, intersection_operator, items, src_tree, dst_tree, threaded;
        npartitions, progress,
    )
    return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
end
