using ConservativeRegridding.Trees
using ConservativeRegridding.Trees: neighbours, has_optimized_neighbour_search
using Test
using Random
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI

# Helper: build a wrapper from x, y vectors and a manifold.
function make_lonlat_wrapper(x, y; manifold = GO.Spherical())
    grid = RegularGrid(manifold, x, y)
    tree = TopDownQuadtreeCursor(grid)
    return LonLatConnectivityWrapper(tree)
end

# Helper: linear index from (i, j).
@inline _lin(i, j, nlon) = i + (j - 1) * nlon

@testset "LonLatConnectivityWrapper inference" begin
    # Global grid, 8 lon × 4 lat cells. Bounds end at the poles, nlon even.
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    @test w.periodic_x       == true
    @test w.pole_top_fold    == true
    @test w.pole_bottom_fold == true
    @test w.nlon == 8
    @test w.nlat == 4
    @test has_optimized_neighbour_search(w) == true
end

@testset "Interior cell — 8 neighbours, exact indices" begin
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    nlon = w.nlon
    # Pick a strict-interior cell: (i, j) = (3, 2) for an 8×4 grid.
    i, j = 3, 2
    idx = _lin(i, j, nlon)
    expected = Set{Int}([
        _lin(i,   j-1, nlon), _lin(i+1, j-1, nlon),
        _lin(i+1, j,   nlon), _lin(i+1, j+1, nlon),
        _lin(i,   j+1, nlon), _lin(i-1, j+1, nlon),
        _lin(i-1, j,   nlon), _lin(i-1, j-1, nlon),
    ])
    nbrs = neighbours(w, idx)
    @test length(nbrs) == 8
    @test Set(nbrs) == expected
end

@testset "i = 1 — 8 neighbours via x-wrap" begin
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    nlon = w.nlon
    i, j = 1, 2
    idx = _lin(i, j, nlon)
    nbrs = neighbours(w, idx)
    @test length(nbrs) == 8
    # The west-side neighbours should wrap to i = nlon.
    @test _lin(nlon, j,   nlon) in nbrs
    @test _lin(nlon, j-1, nlon) in nbrs
    @test _lin(nlon, j+1, nlon) in nbrs
end

@testset "Top row j = nlat — pole fold" begin
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    nlon, nlat = w.nlon, w.nlat
    half = nlon ÷ 2
    i = 4 # not at a wrap edge
    j = nlat
    idx = _lin(i, j, nlon)
    nbrs = neighbours(w, idx)
    @test length(nbrs) == 8
    # 2 same-row + 3 below + 3 across-pole at j = nlat.
    same_row     = (_lin(i-1, j, nlon), _lin(i+1, j, nlon))
    below        = (_lin(i-1, j-1, nlon), _lin(i, j-1, nlon), _lin(i+1, j-1, nlon))
    across_pole_is = mod1.(i .+ (-1:1) .+ half, nlon)
    across_pole  = (_lin(across_pole_is[1], j, nlon),
                    _lin(across_pole_is[2], j, nlon),
                    _lin(across_pole_is[3], j, nlon))
    expected = Set{Int}(vcat(collect(same_row), collect(below), collect(across_pole)))
    @test Set(nbrs) == expected
end

@testset "Top-left corner (1, nlat) — combined wrap + fold" begin
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    nlon, nlat = w.nlon, w.nlat
    i, j = 1, nlat
    idx = _lin(i, j, nlon)
    nbrs = neighbours(w, idx)
    @test length(nbrs) == 8
    # All 8 should be distinct at this nlon (= 8).
    @test length(Set(nbrs)) == 8
end

@testset "Non-global grid: all flags false" begin
    x = collect(range(0.0, 90.0; length = 5))
    y = collect(range(0.0, 45.0; length = 4))
    w = make_lonlat_wrapper(x, y)
    @test w.periodic_x       == false
    @test w.pole_top_fold    == false
    @test w.pole_bottom_fold == false
    nlon = w.nlon
    # Border at (1, 1): only (i+1, 1), (1, j+1), (i+1, j+1) are valid.
    nbrs = neighbours(w, _lin(1, 1, nlon))
    @test length(nbrs) == 3
    @test Set(nbrs) == Set{Int}([
        _lin(2, 1, nlon),
        _lin(1, 2, nlon),
        _lin(2, 2, nlon),
    ])
end

@testset "Odd-nlon global grid: folds forced false" begin
    nlon = 17
    nlat = 8
    x = collect(range(-180.0, 180.0; length = nlon + 1))
    y = collect(range(-90.0,  90.0; length = nlat + 1))
    w = make_lonlat_wrapper(x, y)
    @test w.periodic_x       == true
    @test w.pole_top_fold    == false
    @test w.pole_bottom_fold == false
    # Top row cell — no fold means j = nlat returns 5 neighbours
    # (2 same row + 3 below).
    i = 5
    nbrs = neighbours(w, _lin(i, nlat, nlon))
    @test length(nbrs) == 5
end

@testset "Symmetry round-trip on global grid" begin
    x = collect(range(-180.0, 180.0; length = 9))
    y = collect(range(-90.0,  90.0; length = 5))
    w = make_lonlat_wrapper(x, y)
    ncells = w.nlon * w.nlat
    rng_idxs = vcat(1:ncells)  # exhaustive on this small grid
    for a in rng_idxs
        for b in neighbours(w, a)
            @test a in neighbours(w, b)
        end
    end
end

@testset "Scale test: lon-lat 720×360" begin
    nlon, nlat = 720, 360
    x = collect(range(-180.0, 180.0; length = nlon + 1))
    y = collect(range(-90.0,  90.0; length = nlat + 1))
    w = make_lonlat_wrapper(x, y)
    @test w.periodic_x       == true
    @test w.pole_top_fold    == true
    @test w.pole_bottom_fold == true

    # Sample 1000 random cells. Global + even nlon means all should have 8 neighbours.
    rng = Random.MersenneTwister(42)
    sample = rand(rng, 1:(nlon * nlat), 1000)
    for a in sample
        nbrs = neighbours(w, a)
        @test length(nbrs) == 8
    end
    # Symmetry on the sample.
    for a in sample
        for b in neighbours(w, a)
            @test a in neighbours(w, b)
        end
    end
end
