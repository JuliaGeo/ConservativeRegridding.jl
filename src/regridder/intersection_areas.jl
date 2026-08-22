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
    task_local_operator(op) -> op′

Return the operator instance which the current assembly task should use. 

This is mainly to enable intersection operators to materialize task-local caches.
Repeated calls on one task may return operators backed by the same reusable cache;
different tasks must never share mutable cache storage.

Defaults to returning `op` unchanged.  For any threadsafe operator, this is a no-op.
"""
task_local_operator(op) = op

"""
    output_matrix_size(op, src_tree, dst_tree) -> (nrows, ncols)

Shape of the sparse matrix [`intersection_areas`](@ref) assembles for `op`.

Defaults to `(cell_index_count(dst_tree), cell_index_count(src_tree))` — the
dense global cell-index domains for dst rows and src columns. Operators whose
counts differ from cell index counts (e.g. spectral-element node counts) may
override this.
"""
output_matrix_size(op, src_tree, dst_tree) =
    (Trees.cell_index_count(dst_tree), Trees.cell_index_count(src_tree))

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

function get_all_candidate_pairs!(candidate_idxs::Vector{Tuple{Int, Int}}, threaded::True,
        predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    # from utils/MultithreadedDualDepthFirstSearch.jl
    return multithreaded_dual_query!(candidate_idxs, predicate_f, src_tree, dst_tree)
end

function get_all_candidate_pairs!(candidate_idxs::Vector{Tuple{Int, Int}}, threaded::False,
        predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    empty!(candidate_idxs)
    # from utils/CachedDualDepthFirstSearch.jl - the same pairs in the same order as
    # `STI.dual_depth_first_search`, with the child extents of expensive-extent trees
    # derived once rather than once per opposing child.
    cached_dual_depth_first_search(predicate_f, src_tree, dst_tree) do i1, i2
        push!(candidate_idxs, (i1, i2))
    end
    return candidate_idxs
end

get_all_candidate_pairs(threaded, predicate_f, src_tree, dst_tree) =
    get_all_candidate_pairs!(Tuple{Int, Int}[], threaded, predicate_f, src_tree, dst_tree)

# Repeated serial builds in an outer worker keep these buffers on that task.  A task
# may migrate between scheduler threads, so thread-indexed storage would be unsafe.
mutable struct _AssemblyScratch{T}
    candidate_pairs::Vector{Tuple{Int, Int}}
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
    in_use::Bool
end

_AssemblyScratch(::Type{T}) where {T} =
    _AssemblyScratch(Tuple{Int, Int}[], Int[], Int[], T[], false)

# The key is a module-private type object, unique for every COO value type without
# allocating a composite key on each lookup.
struct _AssemblyScratchKey{T} end

function _acquire_assembly_scratch(::Type{T}) where {T}
    tls = task_local_storage()
    scratch = get(tls, _AssemblyScratchKey{T}, nothing)
    if scratch isa _AssemblyScratch{T} && scratch.in_use
        # Reentrant use on one task gets an unregistered fallback; the outer build
        # remains the buffer retained for the next top-level block.
        scratch = _AssemblyScratch(T)
    elseif !(scratch isa _AssemblyScratch{T})
        scratch = _AssemblyScratch(T)
        tls[_AssemblyScratchKey{T}] = scratch
    end
    scratch.in_use = true
    empty!(scratch.candidate_pairs)
    empty!(scratch.rows)
    empty!(scratch.cols)
    empty!(scratch.vals)
    return scratch
end

function _release_assembly_scratch!(scratch::_AssemblyScratch)
    empty!(scratch.candidate_pairs)
    empty!(scratch.rows)
    empty!(scratch.cols)
    empty!(scratch.vals)
    scratch.in_use = false
    return nothing
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
# `task_local_operator`'s return type may not be inferrable, so resolve the per-task
# operator here and pay one dynamic dispatch per chunk into the type-stable kernel
# below — never one per item.
function _assemble_chunk(style, op, items, src_tree, dst_tree, ValType)
    chunk_op = task_local_operator(op)
    return _assemble_chunk_kernel(style, chunk_op, items, src_tree, dst_tree, ValType)
end

function _assemble_chunk_kernel(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::Type{ValType}) where {S, O, I, T1, T2, ValType}
    rows = Int[]
    cols = Int[]
    vals = ValType[]
    return _assemble_chunk_kernel!(style, op, items, src_tree, dst_tree, rows, cols, vals)
end

function _assemble_chunk_kernel!(style::S, op::O, items::I, src_tree::T1, dst_tree::T2,
        rows::Vector{Int}, cols::Vector{Int}, vals::Vector{ValType}) where {S, O, I, T1, T2, ValType}
    for item in items
        _run_and_store!(style, op, rows, cols, vals, item, src_tree, dst_tree)
    end
    return rows, cols, vals
end

# `True` chunks/spawns, `False` runs one chunk. `$`-interpolation keeps the
# spawned tasks type-stable (concrete `style`/`op`/trees, no boxing).
function _assemble_chunks(style::S, op::O, items::I, src_tree::T1, dst_tree::T2; npartitions, progress) where {S, O, I, T1, T2}
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
    return map(fetch, result_tasks)
end

# Concatenate the per-chunk COO vectors into single vectors, in partition order.
# `init` would cost `vcat`'s pre-sized specialisation, making this quadratic in the partition count.
_concat_chunks(chunks) = (reduce(vcat, getindex.(chunks, 1)),
                          reduce(vcat, getindex.(chunks, 2)),
                          reduce(vcat, getindex.(chunks, 3)))

function assemble_sparse_matrix_coo(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::True; npartitions, progress) where {S, O, I, T1, T2}
    chunks = _assemble_chunks(style, op, items, src_tree, dst_tree; npartitions, progress)
    isempty(chunks) && return Int[], Int[], output_eltype(op, src_tree, dst_tree)[]
    return _concat_chunks(chunks)
end
# Non-threaded version
function assemble_sparse_matrix_coo(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::False; kwargs...) where {S, O, I, T1, T2}
    _assemble_chunk(style, op, items, src_tree, dst_tree, output_eltype(op, src_tree, dst_tree))
end

# Coarse column blocks the window chooser histograms into; windows cut on block boundaries.
const _COLUMN_BLOCKS = 4096

# Below this many entries a window cannot pay for its own `O(nrows)` counting sort.
const _MIN_WINDOW_ENTRIES = 1 << 16

@inline _blockof(col::Int, blockwidth::Int) = (col - 1) ÷ blockwidth + 1

function _column_blocks(ncols::Int)
    ncols < 2 && return 1, max(ncols, 1)
    blockwidth = cld(ncols, min(_COLUMN_BLOCKS, ncols))
    return cld(ncols, blockwidth), blockwidth
end

# Bounds-checked on purpose: `_bucket_chunk!` indexes `windowof` with these block numbers
# under `@inbounds`, so a column far outside `1:ncols` has to be caught here.
function _count_blocks!(hist::Matrix{Int}, c::Int, cols::Vector{Int}, blockwidth::Int)
    for col in cols
        hist[_blockof(col, blockwidth), c] += 1
    end
    return nothing
end

# Cut the coarse blocks into `nwindows` runs of near-equal entry count. The clamp keeps
# the bounds strictly increasing, so every window owns at least one block of columns.
function _window_bounds(blockcount::Vector{Int}, nwindows::Int)
    nblocks = length(blockcount)
    cum = cumsum(blockcount)
    bnd = Vector{Int}(undef, nwindows + 1)
    bnd[1] = 1
    bnd[nwindows + 1] = nblocks + 1
    for w in 2:nwindows
        k = searchsortedfirst(cum, (w - 1) * cum[end] / nwindows)
        bnd[w] = clamp(k + 1, bnd[w - 1] + 1, nblocks - nwindows + w)
    end
    return bnd
end

# `sparse` rejects ragged triplets; the windowed route would read `rows`/`vals` unchecked.
function _check_chunk_lengths(chunks)
    for (rows, cols, vals) in chunks
        length(rows) == length(cols) == length(vals) || throw(ArgumentError(
            "the COO vectors of a chunk must have equal lengths, got $(length(rows)), $(length(cols)) and $(length(vals))"))
    end
    return nothing
end

# Scatter one chunk's triplets into the window they belong to, columns shifted to be
# window-local. `cursor[w]` is this chunk's write position in window `w`.
function _bucket_chunk!(wrows::Vector{Vector{Int}}, wcols::Vector{Vector{Int}}, wvals::Vector{Vector{Tv}},
        chunk, cursor::Vector{Int}, windowof::Vector{Int}, colbase::Vector{Int}, blockwidth::Int) where {Tv}
    rows, cols, vals = chunk
    @inbounds for j in eachindex(cols)
        w = windowof[_blockof(cols[j], blockwidth)]
        p = cursor[w] + 1
        cursor[w] = p
        wrows[w][p] = rows[j]
        wcols[w][p] = cols[j] - colbase[w]
        wvals[w][p] = vals[j]
    end
    return nothing
end

# `sparse`'s own combine defaults, so a window folds duplicates exactly as `sparse` would.
_coocombine(::AbstractVector{Bool}) = |
_coocombine(::AbstractVector) = +

# One window's COO to CSC. `sparse!` is allowed to take the CSC's storage from the COO
# arrays once it is done reading them, and the window owns those arrays outright.
function _window_csc(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Tv}, nrows::Int, ncols::Int) where {Tv}
    n = length(rows)
    return SparseArrays.sparse!(rows, cols, vals, nrows, ncols, _coocombine(vals),
        Vector{Int}(undef, ncols), Vector{Int}(undef, nrows + 1),
        Vector{Int}(undef, n), Vector{Tv}(undef, n), cols, rows, vals)
end

# Copy one window's CSC into the whole matrix's arrays; its columns are contiguous there.
function _splice_window!(colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{Tv},
        mat::SparseArrays.SparseMatrixCSC, base::Int, colbase::Int) where {Tv}
    cp = SparseArrays.getcolptr(mat)
    @inbounds for c in 1:size(mat, 2)
        colptr[colbase + c] = base + cp[c]
    end
    copyto!(rowval, base + 1, SparseArrays.rowvals(mat), 1, SparseArrays.nnz(mat))
    copyto!(nzval, base + 1, SparseArrays.nonzeros(mat), 1, SparseArrays.nnz(mat))
    return nothing
end

"""
    _sparse_from_chunks(chunks, nrows, ncols) -> SparseMatrixCSC

Build the CSC from the per-task COO triplets without concatenating them first.

A coarse column histogram cuts the columns into entry-balanced windows, one pass buckets
the triplets into their window, and each window is built by `SparseArrays.sparse!` on its
own task. Every entry of a column reaches exactly one window, in the order it would have
in the concatenation, and `sparse`'s output for a column is a function of that column's
entries and their order alone — so the result is identical, bit for bit, to `sparse` over
the concatenated triplets.
"""
function _sparse_from_chunks(chunks, nrows::Int, ncols::Int)
    ntotal = sum(chunk -> length(chunk[1]), chunks)
    # One window per thread, capped so the windows' `O(nrows)` scratch stays within the
    # size of the entries they sort.
    nwindows = min(Threads.nthreads(), ntotal ÷ max(nrows, _MIN_WINDOW_ENTRIES))
    return _sparse_from_chunks(chunks, nrows, ncols, nwindows)
end

function _sparse_from_chunks(chunks, nrows::Int, ncols::Int, nwindows::Int)
    _check_chunk_lengths(chunks)
    nblocks, blockwidth = _column_blocks(ncols)
    nwindows = min(nwindows, nblocks)
    if nwindows < 2
        rows, cols, vals = _concat_chunks(chunks)
        return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
    end

    nchunks = length(chunks)
    hist = zeros(Int, nblocks, nchunks)
    hist_tasks = map(1:nchunks) do c
        cols = chunks[c][2]
        StableTasks.@spawn _count_blocks!($hist, $c, $cols, $blockwidth)
    end
    foreach(fetch, hist_tasks)

    bnd = _window_bounds(vec(sum(hist; dims = 2)), nwindows)
    windowof = Vector{Int}(undef, nblocks)
    for w in 1:nwindows, k in bnd[w]:(bnd[w + 1] - 1)
        windowof[k] = w
    end
    colbase = [(bnd[w] - 1) * blockwidth for w in 1:nwindows]

    # Where each (chunk, window) pair writes in its window's arrays: within a window the
    # chunks keep their partition order, so a column's entries keep their global order.
    starts = Matrix{Int}(undef, nchunks, nwindows)
    windowlen = Vector{Int}(undef, nwindows)
    for w in 1:nwindows
        s = 0
        for c in 1:nchunks
            starts[c, w] = s
            for k in bnd[w]:(bnd[w + 1] - 1)
                s += hist[k, c]
            end
        end
        windowlen[w] = s
    end

    Tv = eltype(chunks[1][3])
    wrows = [Vector{Int}(undef, windowlen[w]) for w in 1:nwindows]
    wcols = [Vector{Int}(undef, windowlen[w]) for w in 1:nwindows]
    wvals = [Vector{Tv}(undef, windowlen[w]) for w in 1:nwindows]
    bucket_tasks = map(1:nchunks) do c
        chunk = chunks[c]
        cursor = starts[c, :]
        StableTasks.@spawn _bucket_chunk!($wrows, $wcols, $wvals, $chunk, $cursor, $windowof, $colbase, $blockwidth)
    end
    foreach(fetch, bucket_tasks)

    window_tasks = map(1:nwindows) do w
        width = min(ncols, (bnd[w + 1] - 1) * blockwidth) - colbase[w]
        StableTasks.@spawn _window_csc($(wrows[w]), $(wcols[w]), $(wvals[w]), $nrows, $width)
    end
    mats = SparseArrays.SparseMatrixCSC{Tv, Int}[fetch(t) for t in window_tasks]

    nzbase = Vector{Int}(undef, nwindows + 1)
    nzbase[1] = 0
    for w in 1:nwindows
        nzbase[w + 1] = nzbase[w] + SparseArrays.nnz(mats[w])
    end
    colptr = Vector{Int}(undef, ncols + 1)
    rowval = Vector{Int}(undef, nzbase[end])
    nzval = Vector{Tv}(undef, nzbase[end])
    splice_tasks = map(1:nwindows) do w
        mat = mats[w]
        base = nzbase[w]
        c0 = colbase[w]
        StableTasks.@spawn _splice_window!($colptr, $rowval, $nzval, $mat, $base, $c0)
    end
    foreach(fetch, splice_tasks)
    colptr[ncols + 1] = nzbase[end] + 1
    return SparseArrays.SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)
end

# Threaded: straight from the per-task triplets to the CSC. Serial: one chunk, one `sparse`.
function _assemble_sparse(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, threaded::True,
        nrows::Int, ncols::Int; npartitions, progress) where {S, O, I, T1, T2}
    chunks = _assemble_chunks(style, op, items, src_tree, dst_tree; npartitions, progress)
    isempty(chunks) && return SparseArrays.sparse(
        Int[], Int[], output_eltype(op, src_tree, dst_tree)[], nrows, ncols)
    return _sparse_from_chunks(chunks, nrows, ncols)
end
function _assemble_sparse(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, threaded::False,
        nrows::Int, ncols::Int; kwargs...) where {S, O, I, T1, T2}
    rows, cols, vals = assemble_sparse_matrix_coo(style, op, items, src_tree, dst_tree, threaded)
    return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
end

function _assemble_sparse(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::False,
        nrows::Int, ncols::Int, scratch::_AssemblyScratch{ValType}; kwargs...
    ) where {S, O, I, T1, T2, ValType}
    chunk_op = task_local_operator(op)
    rows, cols, vals = _assemble_chunk_kernel!(
        style, chunk_op, items, src_tree, dst_tree,
        scratch.rows, scratch.cols, scratch.vals)
    # `sparse` copies its COO inputs.  The caller clears these task-owned vectors only
    # after this returns, leaving the matrix's CSC storage independent of the scratch.
    return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
end

function _assemble_sparse(style::S, op::O, items::I, src_tree::T1, dst_tree::T2, threaded::True,
        nrows::Int, ncols::Int, scratch::_AssemblyScratch; kwargs...
    ) where {S, O, I, T1, T2}
    return _assemble_sparse(
        style, op, items, src_tree, dst_tree, threaded, nrows, ncols; kwargs...)
end

"""
    intersection_areas(manifold, threaded, dst_tree, src_tree;
                       intersection_operator = DefaultIntersectionOperator(manifold),
                       npartitions = Threads.nthreads() * 4, progress = false)

Assemble the sparse intersection matrix between `src_tree` and `dst_tree` on
`manifold`.  Returns a `SparseMatrixCSC` of the output of the intersection operator.

This is more of a developer level function, which pulls together the intersection operator interface.
Users should go through [`Regridder`](@ref)`(…; intersection_operator = …)`.

This calls out to five functions, which dispatch on `intersection_operator`:

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
  `(cell_index_count(dst_tree), cell_index_count(src_tree))`.  For the spectral element
  regridder, this is more complex.
- [`output_eltype(intersection_operator, src_tree, dst_tree)`](@ref output_eltype): return the
  element type of the sparse matrix.  This is usually `Float64`, but may be different, especially
  if you wish to build up e.g. a matrix of intersection _polygons_, rather than just areas.
- [`task_local_operator(intersection_operator)`](@ref task_local_operator): return a private
  operator for each assembly task, for operators that carry mutable state such as caches.

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
    ValType = output_eltype(intersection_operator, src_tree, dst_tree)
    scratch = _acquire_assembly_scratch(ValType)

    try
        # Every spatial tree participates through the public extent protocol.  In
        # particular, spherical searches may pair cap and Cartesian XYZ extents.
        predicate_f = Extents.intersects

        # First, run the dual depth first search to get all candidate pairs of
        # cells that may intersect.
        candidate_pairs = get_all_candidate_pairs!(
            scratch.candidate_pairs, threaded, predicate_f, src_tree, dst_tree)

        # Map candidate pairs → work units (default: one pair per unit), assemble the
        # COO triplets (in parallel or serially), and build the sparse matrix.
        items = work_items(intersection_operator, candidate_pairs)
        nrows, ncols = output_matrix_size(intersection_operator, src_tree, dst_tree)
        return _assemble_sparse(
            style, intersection_operator, items, src_tree, dst_tree, threaded, nrows, ncols,
            scratch; npartitions, progress)
    finally
        _release_assembly_scratch!(scratch)
    end
end
