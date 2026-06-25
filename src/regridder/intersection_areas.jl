import GeometryOps as GO
using GeometryOps: SpatialTreeInterface as STI

# Intersection-operator interface: the operator drives assembly via one trait
# (`IntersectionReturnStyle`) + two hooks (`work_items`, `output_matrix_size`),
# each with a default reproducing the built-in area computation.

"""
    abstract type IntersectionReturnStyle

Trait for how an intersection operator delivers its contribution per work item
during sparse-matrix assembly. Resolved once via `IntersectionReturnStyle(op)`
at the top of [`intersection_areas`](@ref) and threaded through the parallel
assembly, so the lookup never happens in the hot loop.

Subtypes:
- [`OutOfPlaceSingleResult`](@ref): kernel `op(src_cell, dst_cell) -> area`; driver stores the COO triplet.
- [`InPlace`](@ref): kernel `op(rows, cols, vals, item, src_tree, dst_tree) -> nothing` pushes its own COO.

Defaults to [`OutOfPlaceSingleResult`](@ref), matching [`DefaultIntersectionOperator`](@ref).
"""
abstract type IntersectionReturnStyle end

"""
    OutOfPlaceSingleResult <: IntersectionReturnStyle

In this [`IntersectionReturnStyle`](@ref), the operator is a function 
`op(src_cell, dst_cell) -> result`.

The harness around it will store the result in a COO triplet `(dst_index, src_index, result)`,
so the operator can remain relatively pure.

The operator must also implement the
[`should_store_result(op, result) -> Bool`](@ref should_store_result) method, 
which determines whether the result should be stored in the sparse matrix.
"""
struct OutOfPlaceSingleResult <: IntersectionReturnStyle end

"""
    InPlace <: IntersectionReturnStyle

In this [`IntersectionReturnStyle`](@ref), the operator is a function 
`op(rows, cols, vals, item, src_tree, dst_tree)`, pushing to the vectors
`rows`, `cols`, and `vals` in place.

Here, `item` is a single entry from the list returned by [`work_items`](@ref).
Usually, that's a single candidate pair `(src_index, dst_index)`.

This provides maximum flexibility, at the cost of a slightly more complex implementation
being required.
"""
struct InPlace <: IntersectionReturnStyle end

"""
    IntersectionReturnStyle(op) -> IntersectionReturnStyle

Return the [`IntersectionReturnStyle`](@ref) of operator `op`. Defaults to
[`OutOfPlaceSingleResult`](@ref); block-assembling operators override to [`InPlace`](@ref).
"""
IntersectionReturnStyle(op) = OutOfPlaceSingleResult()

"""
    work_items(op, candidate_pairs) -> items

Map candidate `(src_index, dst_index)` pairs to the input you want to pass to the 
intersection operator.  This can be used to e.g. run a grouping pass on the source index
over the candidates, as is done for the spectral element regridder.

By default, this is a no-op and returns the candidate pairs as is.
"""
work_items(op, candidate_pairs) = candidate_pairs

"""
    output_matrix_size(op, src_tree, dst_tree) -> (nrows, ncols)

Shape of the sparse matrix [`intersection_areas`](@ref) assembles for `op`.

Defaults to `(prod(ncells(dst_tree)), prod(ncells(src_tree)))` — dst cells as
rows, src cells as columns. Operators whose counts differ from cell counts
(e.g. spectral-element node counts) may override this.
"""
output_matrix_size(op, src_tree, dst_tree) =
    (prod(Trees.ncells(dst_tree)), prod(Trees.ncells(src_tree)))

"""
    output_eltype(op, [src_tree, dst_tree]) -> eltype

Element type of the sparse matrix [`intersection_areas`](@ref) assembles for `op`.

Defaults to `Float64`. Operators may override this to e.g. return a matrix of intersection polygons,
rather than just areas.
"""
function output_eltype end
output_eltype(op, src_tree, dst_tree) = output_eltype(op)
output_eltype(op) = Float64

"""
    should_store_result(op, result) -> Bool

Determine whether the result should be stored in the sparse matrix, after it has been computed.

There is a default implementation for `::Number` results across all operators,
which is simply `result > 0`.  All other combinations of operator and result type
**must** have explicit dispatches implemented.
"""
function should_store_result end
should_store_result(op, result) = should_store_result(result)
should_store_result(result::T) where T <: Number = result > zero(T)
should_store_result(result) = error("""
    `should_store_result` is not implemented for type $(typeof(result)).
    You must implement `should_store_result(op, result) -> Bool` for your operator.
    """)

# If the root tree is a `WithParallelizePolicy`, route the dual-DFS's
# `(node, extent)` query through the user policy; otherwise fall back to the
# default `should_parallelize` dispatch. The wrapper is *not* a dispatch axis
# on `should_parallelize` — detecting it here keeps the dispatch graph simple.
@inline function _build_parallelize_closure(tree::T) where T
    if tree isa Trees.WithParallelizePolicy
        let inner = tree.tree, p = tree.policy
            return (node, extent) -> p(inner, node, extent)
        end
    else
        return (node, extent) -> Trees.should_parallelize(node, extent)
    end
end

function get_all_candidate_pairs(threaded::True, predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
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
@inline function _run_and_store!(::OutOfPlaceSingleResult, op::O, rows::R, cols::C, vals::V, (i1, i2), src_tree::T1, dst_tree::T2) where {O, R, C, V, T1, T2}
    p1 = Trees.getcell(src_tree, i1)
    p2 = Trees.getcell(dst_tree, i2)
    result = op(p1, p2) # usually an area of intersection, by default
    if should_store_result(op, result)
        push!(rows, i2)   # row = destination index
        push!(cols, i1)   # col = source index
        push!(vals, result)
    end
    return nothing
end

@inline function _run_and_store!(::InPlace, op::O, rows::R, cols::C, vals::V, item::I, src_tree::T1, dst_tree::T2) where {O, R, C, V, I, T1, T2}
    op(rows, cols, vals, item, src_tree, dst_tree)   # the operator stores in place
    return nothing
end

# One chunk of work items → its COO triplets. `style` is passed in, not re-resolved.
function _assemble_chunk(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::Type{ValType}) where {S, O, I, T1, T2, ValType}
    rows = Int[]
    cols = Int[]
    vals = ValType[]
    for item in items
        _run_and_store!(style, op, rows, cols, vals, item, src_tree, dst_tree)
    end
    return rows, cols, vals
end

# `True` chunks/spawns, `False` runs one chunk. `$`-interpolation keeps the
# spawned tasks type-stable (concrete `style`/`op`/trees, no boxing).
function assemble_sparse_matrix_coo(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::True; npartitions, progress) where {S, O, I, T1, T2}
    # Partition the list of work items,
    partitions = ChunkSplitters.chunks(items; n = npartitions)
    if progress
        progress_meter = ProgressMeter.Progress(length(partitions); desc = "Computing intersection areas")
    end
    ValType = output_eltype(op, src_tree, dst_tree)
    # and assemble the COO triplets for each partition in parallel.
    # This is a bit oversubscribed though I guess.  But Julia's dynamic
    # scheduler should handle it fine.
    result_tasks = [
        StableTasks.@spawn begin
            ret = _assemble_chunk($style, $op, partition, $src_tree, $dst_tree, $ValType)
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
# Non-threaded version
function assemble_sparse_matrix_coo(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::False; kwargs...) where {S, O, I, T1, T2}
    _assemble_chunk(style, op, items, src_tree, dst_tree, output_eltype(op, src_tree, dst_tree))
end

"""
    intersection_areas(manifold, threaded, dst_tree, src_tree;
                       intersection_operator = DefaultIntersectionOperator(manifold),
                       npartitions = Threads.nthreads() * 4, progress = false)

Assemble the sparse intersection matrix between `src_tree` and `dst_tree` on
`manifold`.  Returns a `SparseMatrixCSC` of the output of the intersection operator.

This is more of a developer level function, which pulls together the intersection operator interface.
Users should go through [`Regridder`](@ref)`(…; intersection_operator = …)`.

This calls out to four functions, which dispatch on `intersection_operator`:

- [`IntersectionReturnStyle(intersection_operator)`](@ref IntersectionReturnStyle): return an
  [`IntersectionReturnStyle`](@ref) trait object, defining how the operator wants to receive and store
  the results of its computation.  This is usually either [`OutOfPlaceSingleResult`](@ref) or
  [`InPlace`](@ref).
- [`work_items(intersection_operator, candidate_pairs)`](@ref work_items): return a vector of
  "work items".  For the regular area-of-intersection operator, this is a vector of
  `(src_index, dst_index)` pairs.  But it can also be more complex, as in the spectral element
  regridder.
- [`output_matrix_size(intersection_operator, src_tree, dst_tree)`](@ref output_matrix_size):
  return the `(nrows, ncols)` shape of the sparse matrix.  For the regular operator, this is
  `(ncells(dst_tree), ncells(src_tree))`.  For the spectral element regridder, this is more
  complex.
- [`output_eltype(intersection_operator, src_tree, dst_tree)`](@ref output_eltype): return the
  element type of the sparse matrix.  This is usually `Float64`, but may be different, especially
  if you wish to build up e.g. a matrix of intersection _polygons_, rather than just areas.

`threaded` is a `GeometryOpsCore.BoolsAsTypes` (`True()`/`False()`; convert via
`booltype(::Bool)`). When threaded, work items are partitioned into `npartitions`
chunks assembled on separate tasks via ChunkSplitters.jl.
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
    rows, cols, vals = assemble_sparse_matrix_coo(
        style, intersection_operator, items, src_tree, dst_tree, threaded;
        npartitions, progress,
    )
    return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
end
