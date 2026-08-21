using ConservativeRegridding
using Random
using SparseArrays
using Test

const CR = ConservativeRegridding

chunk(rows, cols, vals) = CR.COOChunk(rows, cols, vals)

function fixture(rng, nrows, ncols, nchunks, nper, ndup)
    map(1:nchunks) do _
        rows = rand(rng, 1:nrows, nper)
        cols = rand(rng, 1:ncols, nper)
        vals = randn(rng, nper)
        # Repeat the head so columns carry duplicates within and across chunks.
        chunk(
            vcat(rows, rows[1:ndup]),
            vcat(cols, cols[1:ndup]),
            vcat(vals, randn(rng, ndup)),
        )
    end
end

@testset "COO chunks validate their parallel arrays" begin
    good = chunk([1, 2], [3, 4], [5.0, 6.0])
    @test length(good) == 2
    @test !isempty(good)
    @test_throws ArgumentError chunk([1, 2], [3], [5.0, 6.0])
end

@testset "Windowed COO assembly is bit-identical to `sparse`" begin
    rng = MersenneTwister(20260820)
    shapes = (
        (200, 5000, 32, 4000),
        (50, 97, 16, 8000),
        (1, 3, 8, 30000),
        (5000, 200, 32, 6000),
        (300, 4097, 7, 20000),
        (300, 100_000, 32, 5000),
    )
    for (nrows, ncols, nchunks, nper) in shapes
        chunks = fixture(rng, nrows, ncols, nchunks, nper, nper ÷ 3)
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, nrows, ncols)
        @test nnz(expected) < length(rows) # the fixture really contains duplicates
        for nwindows in (2, 3, 8)
            got = CR._sparse_from_chunks(chunks, nrows, ncols, nwindows)
            @test got.colptr == expected.colptr
            @test got.rowval == expected.rowval
            @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
        end
    end

    # A skewed distribution exercises a hot block that cannot itself be split.
    let nrows = 200, ncols = 20_000, nchunks = 32, nper = 5000
        chunks = map(1:nchunks) do _
            cols = [
                rand(rng) < 0.9 ? rand(rng, 1:200) : rand(rng, 1:ncols)
                for _ in 1:nper
            ]
            chunk(rand(rng, 1:nrows, nper), cols, randn(rng, nper))
        end
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, nrows, ncols)
        for nwindows in (2, 8)
            got = CR._sparse_from_chunks(chunks, nrows, ncols, nwindows)
            @test got.colptr == expected.colptr
            @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
        end
    end

    # Empty chunks are a normal outcome of the parallel work split.
    let nrows = 100, ncols = 3000, nper = 20_000
        chunks = map(1:16) do index
            nentries = iseven(index) ? 0 : nper
            chunk(
                rand(rng, 1:nrows, nentries),
                rand(rng, 1:ncols, nentries),
                randn(rng, nentries),
            )
        end
        rows, cols, vals = CR._concat_chunks(chunks)
        @test CR._sparse_from_chunks(chunks, nrows, ncols, 4) ==
            sparse(rows, cols, vals, nrows, ncols)
    end
end

@testset "Column-window plans are contiguous and balanced" begin
    cases = (
        fill(100, 64),
        [1000; fill(1, 63)],
        [fill(0, 60); fill(500, 4)],
        collect(1:64),
        reverse(collect(1:64)),
    )
    for block_counts in cases, nwindows in (2, 3, 8, 64)
        histogram = reshape(block_counts, :, 1)
        plan = CR._plan_windows(histogram, length(block_counts), 1, nwindows)
        @test length(plan.columns) == nwindows
        @test first(plan.columns[1]) == 1
        @test last(plan.columns[end]) == length(block_counts)
        @test all(
            last(plan.columns[index]) + 1 == first(plan.columns[index + 1])
            for index in 1:(nwindows - 1)
        )
        @test all(!isempty, plan.columns)
    end

    even_plan = CR._plan_windows(reshape(fill(100, 64), :, 1), 64, 1, 4)
    @test even_plan.columns == [1:16, 17:32, 33:48, 49:64]

    hot_plan = CR._plan_windows(
        reshape([1000; fill(1, 63)], :, 1), 64, 1, 4,
    )
    @test hot_plan.columns == [1:1, 2:2, 3:3, 4:64]
end

@testset "Windowed COO assembly preserves order-sensitive folds" begin
    # `1e16 + 1.0 - 1e16` is 0.0 in input order and 1.0 in either other order.
    cancel = [
        chunk([1], [7], [1.0e16]),
        chunk([1], [7], [1.0]),
        chunk([1], [7], [-1.0e16]),
    ]
    for nwindows in (2, 3, 8)
        @test CR._sparse_from_chunks(cancel, 4, 20_000, nwindows).nzval == [0.0]
    end

    signs = [
        chunk([1, 2], [3, 5], [-0.0, 5.0e-324]),
        chunk([1, 2], [3, 5], [0.0, 5.0e-324]),
    ]
    bools = [
        chunk([1, 2], [3, 5], [true, false]),
        chunk([1, 2], [3, 5], [false, false]),
    ]
    for chunks in (cancel, signs, bools)
        rows, cols, vals = CR._concat_chunks(chunks)
        expected = sparse(rows, cols, vals, 4, 20_000)
        for nwindows in (2, 3, 8)
            got = CR._sparse_from_chunks(chunks, 4, 20_000, nwindows)
            @test got.colptr == expected.colptr
            @test got.rowval == expected.rowval
            @test reinterpret(UInt8, got.nzval) == reinterpret(UInt8, expected.nzval)
        end
    end
end

@testset "Default window selection does not affect output" begin
    nrows, ncols = 64, 30_000
    rng = MersenneTwister(20260821)
    chunks = [
        chunk(
            rand(rng, 1:nrows, 45_000),
            rand(rng, 1:ncols, 45_000),
            randn(rng, 45_000),
        )
        for _ in 1:3
    ]
    rows, cols, vals = CR._concat_chunks(chunks)
    expected = sparse(rows, cols, vals, nrows, ncols)
    got = CR._sparse_from_chunks(chunks, nrows, ncols)
    @test got.colptr == expected.colptr
    @test got.rowval == expected.rowval
    @test reinterpret(UInt64, got.nzval) == reinterpret(UInt64, expected.nzval)
end
