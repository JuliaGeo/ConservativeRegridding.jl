# A validated batch of COO entries. Chunks preserve the order in which work partitions
# produced their entries; the windowed assembler relies on that order when folding duplicates.
struct COOChunk{Tv}
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{Tv}

    function COOChunk{Tv}(
            rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Tv},
        ) where {Tv}
        length(rows) == length(cols) == length(vals) || throw(ArgumentError(
            "the COO vectors of a chunk must have equal lengths, got " *
            "$(length(rows)), $(length(cols)) and $(length(vals))",
        ))
        return new{Tv}(rows, cols, vals)
    end
end

COOChunk(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Tv}) where {Tv} =
    COOChunk{Tv}(rows, cols, vals)

Base.length(chunk::COOChunk) = length(chunk.rows)
Base.isempty(chunk::COOChunk) = isempty(chunk.rows)

"""
    SparseMatrixAssemblyCache([T = Float64])

Reusable buffers for sparse intersection-matrix assembly. Pass the cache to
[`Regridder`](@ref) or [`intersection_areas`](@ref) with `cache=...`; do not use
one cache concurrently from multiple calls.
"""
mutable struct SparseMatrixAssemblyCache{T}
    candidate_pairs::Vector{Tuple{Int, Int}}
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
end

SparseMatrixAssemblyCache(::Type{T}) where {T} =
    SparseMatrixAssemblyCache(Tuple{Int, Int}[], Int[], Int[], T[])
SparseMatrixAssemblyCache() = SparseMatrixAssemblyCache(Float64)

function Base.empty!(cache::SparseMatrixAssemblyCache)
    empty!(cache.candidate_pairs)
    empty!(cache.rows)
    empty!(cache.cols)
    empty!(cache.vals)
    return cache
end

_assembly_cache(::Type{T}, ::Nothing) where {T} = SparseMatrixAssemblyCache(T)
_assembly_cache(::Type{T}, cache::SparseMatrixAssemblyCache{T}) where {T} = empty!(cache)
function _assembly_cache(::Type{T}, cache::SparseMatrixAssemblyCache) where {T}
    throw(ArgumentError(
        "cache stores $(eltype(cache.vals)) values, but the intersection operator outputs $T"))
end

# Concatenate in chunk order, which is also the input order used to fold duplicates.
function _concat_chunks(chunks::AbstractVector{<:COOChunk{Tv}}) where {Tv}
    isempty(chunks) && return Int[], Int[], Tv[]
    rows = reduce(vcat, map(chunk -> chunk.rows, chunks))
    cols = reduce(vcat, map(chunk -> chunk.cols, chunks))
    vals = reduce(vcat, map(chunk -> chunk.vals, chunks))
    return rows, cols, vals
end

# Everything needed to scatter ordered chunks into contiguous column windows.
struct WindowPlan
    columns::Vector{UnitRange{Int}}
    block_to_window::Vector{Int}
    chunk_offsets::Matrix{Int}
    entry_counts::Vector{Int}
    block_width::Int
end

# Coarse column blocks the window chooser histograms into; windows cut on block boundaries.
const _COLUMN_BLOCKS = 4096

# Below this many entries a window cannot pay for its own `O(nrows)` counting sort.
const _MIN_WINDOW_ENTRIES = 1 << 16

@inline _blockof(col::Int, block_width::Int) = (col - 1) ÷ block_width + 1

function _column_blocks(ncols::Int)
    ncols < 2 && return 1, max(ncols, 1)
    block_width = cld(ncols, min(_COLUMN_BLOCKS, ncols))
    return cld(ncols, block_width), block_width
end

# Bounds-checked on purpose: the scatter indexes `block_to_window` under `@inbounds`,
# so an out-of-range column must be caught while constructing the histogram.
function _count_blocks!(
        histogram::Matrix{Int}, chunk_index::Int, chunk::COOChunk, block_width::Int,
    )
    for col in chunk.cols
        histogram[_blockof(col, block_width), chunk_index] += 1
    end
    return nothing
end

function _column_histogram(chunks, ncols::Int)
    nblocks, block_width = _column_blocks(ncols)
    histogram = zeros(Int, nblocks, length(chunks))
    tasks = map(eachindex(chunks)) do chunk_index
        chunk = chunks[chunk_index]
        StableTasks.@spawn _count_blocks!(
            $histogram, $chunk_index, $chunk, $block_width,
        )
    end
    foreach(fetch, tasks)
    return histogram, block_width
end

# Cut the coarse blocks into runs of near-equal entry count. Bounds are strictly
# increasing, so every window owns at least one block of columns.
function _window_bounds(block_counts::AbstractVector{Int}, nwindows::Int)
    nblocks = length(block_counts)
    cumulative = cumsum(block_counts)
    bounds = Vector{Int}(undef, nwindows + 1)
    bounds[1] = 1
    bounds[end] = nblocks + 1
    for window in 2:nwindows
        block = searchsortedfirst(
            cumulative, (window - 1) * cumulative[end] / nwindows,
        )
        bounds[window] = clamp(
            block + 1, bounds[window - 1] + 1, nblocks - nwindows + window,
        )
    end
    return bounds
end

function _plan_windows(
        histogram::Matrix{Int}, ncols::Int, block_width::Int, nwindows::Int,
    )
    block_counts = vec(sum(histogram; dims = 2))
    bounds = _window_bounds(block_counts, nwindows)

    block_to_window = Vector{Int}(undef, length(block_counts))
    columns = Vector{UnitRange{Int}}(undef, nwindows)
    for window in 1:nwindows
        blocks = bounds[window]:(bounds[window + 1] - 1)
        block_to_window[blocks] .= window
        first_col = (first(blocks) - 1) * block_width + 1
        last_col = min(ncols, last(blocks) * block_width)
        columns[window] = first_col:last_col
    end

    # Each chunk owns a disjoint span of every destination window. The spans follow
    # chunk order, preserving duplicate-fold order within every column.
    chunk_offsets = Matrix{Int}(undef, size(histogram, 2), nwindows)
    entry_counts = Vector{Int}(undef, nwindows)
    for window in 1:nwindows
        entries = 0
        blocks = bounds[window]:(bounds[window + 1] - 1)
        for chunk_index in axes(histogram, 2)
            chunk_offsets[chunk_index, window] = entries
            entries += sum(@view histogram[blocks, chunk_index])
        end
        entry_counts[window] = entries
    end

    return WindowPlan(
        columns, block_to_window, chunk_offsets, entry_counts, block_width,
    )
end

function _scatter_chunk!(
        windows::Vector{COOChunk{Tv}}, chunk::COOChunk{Tv},
        cursor::Vector{Int}, plan::WindowPlan,
    ) where {Tv}
    @inbounds for entry in eachindex(chunk.cols)
        window = plan.block_to_window[_blockof(chunk.cols[entry], plan.block_width)]
        position = cursor[window] + 1
        cursor[window] = position
        windows[window].rows[position] = chunk.rows[entry]
        windows[window].cols[position] =
            chunk.cols[entry] - first(plan.columns[window]) + 1
        windows[window].vals[position] = chunk.vals[entry]
    end
    return nothing
end

function _scatter_to_windows(
        chunks::AbstractVector{<:COOChunk{Tv}}, plan::WindowPlan,
    ) where {Tv}
    windows = [
        COOChunk(
            Vector{Int}(undef, count),
            Vector{Int}(undef, count),
            Vector{Tv}(undef, count),
        )
        for count in plan.entry_counts
    ]

    tasks = map(eachindex(chunks)) do chunk_index
        chunk = chunks[chunk_index]
        cursor = copy(@view plan.chunk_offsets[chunk_index, :])
        StableTasks.@spawn _scatter_chunk!($windows, $chunk, $cursor, $plan)
    end
    foreach(fetch, tasks)
    return windows
end

# `sparse`'s own combine defaults, so each window folds duplicates identically.
_coocombine(::AbstractVector{Bool}) = |
_coocombine(::AbstractVector) = +

# `sparse!` may take its output storage from these window-owned COO vectors.
function _window_csc(chunk::COOChunk{Tv}, nrows::Int, ncols::Int) where {Tv}
    nentries = length(chunk)
    return SparseArrays.sparse!(
        chunk.rows, chunk.cols, chunk.vals, nrows, ncols, _coocombine(chunk.vals),
        Vector{Int}(undef, ncols), Vector{Int}(undef, nrows + 1),
        Vector{Int}(undef, nentries), Vector{Tv}(undef, nentries),
        chunk.cols, chunk.rows, chunk.vals,
    )
end

function _build_window_cscs(
        windows::AbstractVector{<:COOChunk{Tv}}, plan::WindowPlan, nrows::Int,
    ) where {Tv}
    tasks = map(eachindex(windows)) do window
        entries = windows[window]
        ncols = length(plan.columns[window])
        StableTasks.@spawn _window_csc($entries, $nrows, $ncols)
    end
    return SparseArrays.SparseMatrixCSC{Tv, Int}[fetch(task) for task in tasks]
end

function _copy_window!(
        colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{Tv},
        matrix::SparseArrays.SparseMatrixCSC{Tv, Int}, nnz_offset::Int,
        first_col::Int,
    ) where {Tv}
    window_colptr = SparseArrays.getcolptr(matrix)
    @inbounds for local_col in 1:size(matrix, 2)
        global_col = first_col + local_col - 1
        colptr[global_col] = nnz_offset + window_colptr[local_col]
    end
    copyto!(
        rowval, nnz_offset + 1, SparseArrays.rowvals(matrix), 1,
        SparseArrays.nnz(matrix),
    )
    copyto!(
        nzval, nnz_offset + 1, SparseArrays.nonzeros(matrix), 1,
        SparseArrays.nnz(matrix),
    )
    return nothing
end

function _stitch_windows(
        matrices::Vector{SparseArrays.SparseMatrixCSC{Tv, Int}},
        plan::WindowPlan, nrows::Int, ncols::Int,
    ) where {Tv}
    nnz_offsets = Vector{Int}(undef, length(matrices) + 1)
    nnz_offsets[1] = 0
    for window in eachindex(matrices)
        nnz_offsets[window + 1] =
            nnz_offsets[window] + SparseArrays.nnz(matrices[window])
    end

    colptr = Vector{Int}(undef, ncols + 1)
    rowval = Vector{Int}(undef, nnz_offsets[end])
    nzval = Vector{Tv}(undef, nnz_offsets[end])
    tasks = map(eachindex(matrices)) do window
        matrix = matrices[window]
        nnz_offset = nnz_offsets[window]
        first_col = first(plan.columns[window])
        StableTasks.@spawn _copy_window!(
            $colptr, $rowval, $nzval, $matrix, $nnz_offset, $first_col,
        )
    end
    foreach(fetch, tasks)
    colptr[end] = nnz_offsets[end] + 1
    return SparseArrays.SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)
end

function _concatenated_sparse(chunks, nrows::Int, ncols::Int)
    rows, cols, vals = _concat_chunks(chunks)
    return SparseArrays.sparse(rows, cols, vals, nrows, ncols)
end

function _choose_window_count(chunks, nrows::Int, max_windows::Int)
    nentries = sum(length, chunks; init = 0)
    return min(max_windows, nentries ÷ max(nrows, _MIN_WINDOW_ENTRIES))
end

"""
    _sparse_from_chunks(chunks, nrows, ncols; max_windows = Threads.nthreads())

Build a CSC from ordered COO chunks. Large inputs are scattered into contiguous,
entry-balanced column windows and assembled in parallel; small inputs use one `sparse`.
"""
function _sparse_from_chunks(
        chunks::AbstractVector{<:COOChunk}, nrows::Int, ncols::Int;
        max_windows::Int = Threads.nthreads(),
    )
    nwindows = _choose_window_count(chunks, nrows, max_windows)
    return _sparse_from_chunks(chunks, nrows, ncols, nwindows)
end

# The positional window count is retained as a test seam for bit-identity checks.
function _sparse_from_chunks(
        chunks::AbstractVector{<:COOChunk}, nrows::Int, ncols::Int, nwindows::Int,
    )
    nblocks, _ = _column_blocks(ncols)
    nwindows = min(nwindows, nblocks)
    nwindows < 2 && return _concatenated_sparse(chunks, nrows, ncols)

    histogram, block_width = _column_histogram(chunks, ncols)
    plan = _plan_windows(histogram, ncols, block_width, nwindows)
    windows = _scatter_to_windows(chunks, plan)
    matrices = _build_window_cscs(windows, plan, nrows)
    return _stitch_windows(matrices, plan, nrows, ncols)
end

# The serial path fills the cache-owned COO vectors, while the threaded path preserves
# the independent ordered-chunk/window pipeline extracted here by #139.
function _assemble_sparse(
        style::S, op::O, items::I, src_tree::T1, dst_tree::T2, ::False,
        nrows::Int, ncols::Int, scratch::SparseMatrixAssemblyCache{ValType}; kwargs...,
    ) where {S, O, I, T1, T2, ValType}
    chunk_op = task_local_operator(op)
    chunk = _assemble_chunk_kernel!(
        style, chunk_op, items, src_tree, dst_tree,
        scratch.rows, scratch.cols, scratch.vals,
    )
    # The sparse constructor copies the cache-owned COO inputs before the caller clears them.
    return _sparse_from_chunks(COOChunk{ValType}[chunk], nrows, ncols; max_windows = 1)
end

function _assemble_sparse(
        style::S, op::O, items::I, src_tree::T1, dst_tree::T2, threaded::True,
        nrows::Int, ncols::Int, ::SparseMatrixAssemblyCache; kwargs...,
    ) where {S, O, I, T1, T2}
    chunks = _collect_coo_chunks(
        style, op, items, src_tree, dst_tree, threaded; kwargs...,
    )
    return _sparse_from_chunks(chunks, nrows, ncols; max_windows = Threads.nthreads())
end
