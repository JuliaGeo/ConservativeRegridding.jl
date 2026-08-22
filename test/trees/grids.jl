import ConservativeRegridding
using ConservativeRegridding.Trees
using ConservativeRegridding.Trees: cell_range_extent
using Test
import GeoInterface as GI
import GeometryOpsCore as GOCore
import Extents

# Helper to build a matrix of lon/lat points covering the globe
function make_lonlat_point_matrix(nx, ny)
    lons = range(-180, 180, length=nx+1)
    lats = range(-90, 90, length=ny+1)
    return [(lon, lat) for lon in lons, lat in lats]
end

# Helper to build a matrix of polygons from points
function make_polygon_matrix(points)
    nx, ny = size(points) .- 1
    polygons = Matrix{GI.Polygon}(undef, nx, ny)
    for i in 1:nx, j in 1:ny
        polygons[i, j] = GI.Polygon([GI.LinearRing([
            points[i, j],
            points[i+1, j],
            points[i+1, j+1],
            points[i, j+1],
            points[i, j]
        ])])
    end
    return polygons
end

@testset "ExplicitPolygonGrid" begin
    @testset "16×16 global grid" begin
        points = make_lonlat_point_matrix(16, 16)
        polygons = make_polygon_matrix(points)
        grid = ExplicitPolygonGrid(polygons)

        @test ncells(grid, 1) == 16
        @test ncells(grid, 2) == 16

        # getcell returns valid polygon
        cell = getcell(grid, 1, 1)
        @test GI.npoint(GI.getexterior(cell)) == 5  # closed ring

        cell_corner = getcell(grid, 16, 16)
        @test GI.npoint(GI.getexterior(cell_corner)) == 5
    end

    @testset "13×17 odd-sized grid" begin
        points = make_lonlat_point_matrix(13, 17)
        polygons = make_polygon_matrix(points)
        grid = ExplicitPolygonGrid(polygons)

        @test ncells(grid, 1) == 13
        @test ncells(grid, 2) == 17

        cell = getcell(grid, 7, 9)
        @test GI.npoint(GI.getexterior(cell)) == 5
    end
end

@testset "CellBasedGrid" begin
    @testset "16×16 global grid" begin
        points = make_lonlat_point_matrix(16, 16)
        grid = CellBasedGrid(points)

        @test ncells(grid, 1) == 16
        @test ncells(grid, 2) == 16

        # getcell returns valid polygon with correct coordinates
        cell = getcell(grid, 1, 1)
        @test GI.npoint(GI.getexterior(cell)) == 5

        # Check first cell is in bottom-left corner (lon=-180, lat=-90)
        ring = GI.getexterior(cell)
        first_point = GI.getpoint(ring, 1)
        @test GI.x(first_point) == -180.0
        @test GI.y(first_point) == -90.0

        cell_corner = getcell(grid, 16, 16)
        @test GI.npoint(GI.getexterior(cell_corner)) == 5
    end

    @testset "13×17 odd-sized grid" begin
        points = make_lonlat_point_matrix(13, 17)
        grid = CellBasedGrid(points)

        @test ncells(grid, 1) == 13
        @test ncells(grid, 2) == 17

        cell = getcell(grid, 7, 9)
        @test GI.npoint(GI.getexterior(cell)) == 5
    end

    @testset "3×5 small grid" begin
        points = make_lonlat_point_matrix(3, 5)
        grid = CellBasedGrid(points)

        @test ncells(grid, 1) == 3
        @test ncells(grid, 2) == 5
    end
end

@testset "RegularGrid" begin
    @testset "16×16 global grid" begin
        lons = collect(range(-180, 180, length=17))
        lats = collect(range(-90, 90, length=17))
        grid = RegularGrid(lons, lats)

        @test ncells(grid, 1) == 16
        @test ncells(grid, 2) == 16

        # getcell returns valid polygon
        cell = getcell(grid, 1, 1)
        @test GI.npoint(GI.getexterior(cell)) == 5

        # Check coordinates
        ring = GI.getexterior(cell)
        first_point = GI.getpoint(ring, 1)
        @test first_point[1] == -180.0
        @test first_point[2] == -90.0

        cell_corner = getcell(grid, 16, 16)
        @test GI.npoint(GI.getexterior(cell_corner)) == 5
    end

    @testset "13×17 odd-sized grid" begin
        lons = collect(range(-180, 180, length=14))
        lats = collect(range(-90, 90, length=18))
        grid = RegularGrid(lons, lats)

        @test ncells(grid, 1) == 13
        @test ncells(grid, 2) == 17

        cell = getcell(grid, 7, 9)
        @test GI.npoint(GI.getexterior(cell)) == 5
    end

    @testset "3×5 small grid" begin
        lons = collect(range(-180, 180, length=4))
        lats = collect(range(-90, 90, length=6))
        grid = RegularGrid(lons, lats)

        @test ncells(grid, 1) == 3
        @test ncells(grid, 2) == 5
    end
end

# Regression test for issue #65: cell_range_extent had swapped mapreduce arguments
# for ExplicitPolygonGrid{Planar}.
@testset "cell_range_extent for Planar ExplicitPolygonGrid (#65)" begin
    # Helper to build a simple planar unit-square grid
    function make_planar_grid(nx, ny)
        polys = Matrix{GI.Polygon}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            x0, x1 = (i - 1) / nx, i / nx
            y0, y1 = (j - 1) / ny, j / ny
            ring = GI.LinearRing([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
            polys[i, j] = GI.Polygon([ring])
        end
        polys
    end

    @testset "2×2 grid full range" begin
        polys = make_planar_grid(2, 2)
        epg = ExplicitPolygonGrid(GOCore.Planar(), polys)
        ext = cell_range_extent(epg, 1:2, 1:2)
        @test ext == Extents.Extent(X=(0.0, 1.0), Y=(0.0, 1.0))
    end

    @testset "2×2 grid partial range" begin
        polys = make_planar_grid(2, 2)
        epg = ExplicitPolygonGrid(GOCore.Planar(), polys)
        # Only the first cell (bottom-left quadrant)
        ext = cell_range_extent(epg, 1:1, 1:1)
        @test ext == Extents.Extent(X=(0.0, 0.5), Y=(0.0, 0.5))
    end

    @testset "4×3 grid full range" begin
        polys = make_planar_grid(4, 3)
        epg = ExplicitPolygonGrid(GOCore.Planar(), polys)
        ext = cell_range_extent(epg, 1:4, 1:3)
        @test ext == Extents.Extent(X=(0.0, 1.0), Y=(0.0, 1.0))
    end

    @testset "4×3 grid subrange" begin
        polys = make_planar_grid(4, 3)
        epg = ExplicitPolygonGrid(GOCore.Planar(), polys)
        # Cells (2:3, 1:2) should cover X=(0.25, 0.75), Y=(0.0, 2/3)
        ext = cell_range_extent(epg, 2:3, 1:2)
        @test ext.X[1] ≈ 0.25
        @test ext.X[2] ≈ 0.75
        @test ext.Y[1] ≈ 0.0
        @test ext.Y[2] ≈ 2 / 3
    end
end

# ===========================================================================
# Spherical bounding-cap machinery: getvertex + CurvilinearGridPerimeterPoints
# (replacing the private `_pt_at` accessor and the `PerimeterPoints` state machine).
# ===========================================================================
import GeometryOps as GO
import ConstructionBase
import LinearAlgebra

# Regional spherical grids used for the ordinary finite-cap characterization.
const _SPH_LONS = collect(range(0.0, 40.0, length = 5))   # 4 cells along i
const _SPH_LATS = collect(range(0.0, 30.0, length = 4))   # 3 cells along j
const _SPH_LONLAT = [(lon, lat) for lon in _SPH_LONS, lat in _SPH_LATS]
const _SPH_PTS = GO.UnitSphereFromGeographic().(_SPH_LONLAT)
_regular_sph()   = RegularGrid(GOCore.Spherical(), _SPH_LONS, _SPH_LATS)
_cellbased_sph() = CellBasedGrid(_SPH_PTS)   # UnitSphericalPoint eltype ⇒ Spherical

function _is_canonical_whole_sphere(cap)
    T = typeof(cap.radius)
    canonical_point = GO.UnitSpherical.UnitSphericalPoint(zero(T), zero(T), one(T))
    return cap.point == canonical_point && cap.radius == nextfloat(T(pi))
end

function _test_cap_shape(cap)
    T = typeof(cap.radius)
    @test isfinite(cap.point)
    @test isfinite(cap.radius)
    @test cap.radius <= T(pi) / T(2) || _is_canonical_whole_sphere(cap)
end

# Check every declared shortest-geodesic cell edge, not just its vertices.
function _test_dense_edge_coverage(cap, points, irange, jrange; subdivisions = 32)
    _test_cap_shape(cap)
    for j in jrange, i in irange
        p1, p2 = points[i, j], points[i + 1, j]
        p3, p4 = points[i + 1, j + 1], points[i, j + 1]
        for (a, b) in ((p1, p2), (p2, p3), (p3, p4), (p4, p1)), k in 0:subdivisions
            p = GO.UnitSpherical.slerp(a, b, k / subdivisions)
            @test GO.spherical_distance(cap.point, p) <= cap.radius
        end
    end
end

@testset "cell_range_extent spherical: finite cap and outward radius" begin
    ir, jr = 2:3, 1:2
    imin, imax = first(ir), last(ir) + 1
    jmin, jmax = first(jr), last(jr) + 1
    expected_center = LinearAlgebra.normalize(
        (_SPH_PTS[imin, jmin] + _SPH_PTS[imax, jmin] +
         _SPH_PTS[imax, jmax] + _SPH_PTS[imin, jmax]) / 4)

    cellbased = _cellbased_sph()
    raw_radius = maximum(p -> GO.spherical_distance(expected_center, p),
        ConservativeRegridding.Trees.CurvilinearGridPerimeterPoints(
            cellbased, imin, imax, jmin, jmax))
    expected_radius = nextfloat(raw_radius * 1.0001)

    for g in (_regular_sph(), cellbased)
        cap = cell_range_extent(g, ir, jr)
        @test isapprox(cap.point, expected_center; atol = 1e-12)
        @test cap.radius == expected_radius
        @test cap.radius > raw_radius * 1.0001
        _test_dense_edge_coverage(cap, _SPH_PTS, ir, jr)
    end
end

_float32_point(p) = GO.UnitSpherical.UnitSphericalPoint(
    Float32(p[1]), Float32(p[2]), Float32(p[3]))

const _SPH_PTS32 = _float32_point.(_SPH_PTS)

@testset "cell_range_extent spherical: Float32 finite cap" begin
    ir, jr = 2:3, 1:2
    imin, imax = first(ir), last(ir) + 1
    jmin, jmax = first(jr), last(jr) + 1

    cellbased = CellBasedGrid(_SPH_PTS32)
    regular = RegularGrid(GOCore.Spherical(), Float32.(_SPH_LONS), Float32.(_SPH_LATS))
    regular_points = [
        ConservativeRegridding.Trees.getvertex(regular, i, j)
        for i in axes(_SPH_PTS32, 1), j in axes(_SPH_PTS32, 2)
    ]
    for (grid, points) in ((regular, regular_points), (cellbased, _SPH_PTS32))
        expected_center = LinearAlgebra.normalize(
            (points[imin, jmin] + points[imax, jmin] +
             points[imax, jmax] + points[imin, jmax]) / 4)
        raw_radius = maximum(p -> GO.spherical_distance(expected_center, p),
            ConservativeRegridding.Trees.CurvilinearGridPerimeterPoints(
                grid, imin, imax, jmin, jmax))
        expected_radius = nextfloat(Float32(raw_radius * Float32(1.0001)))
        cap = cell_range_extent(grid, ir, jr)
        @test cap isa GO.UnitSpherical.SphericalCap{Float32}
        @test cap.radius == expected_radius
        @test cap.radius > Float32(raw_radius * Float32(1.0001))
        _test_dense_edge_coverage(cap, points, ir, jr)
    end
end

# Longitude identifies `u` and latitude is strictly increasing in `v`, so this
# smooth chart is injective. Its sinusoid vanishes at the four corners while
# escaping their finite cap at intermediate boundary vertices.
const _WARPED_SPH_PTS = [
    GO.UnitSphereFromGeographic()((
        -50.0 + 100.0 * u,
        -20.0 + 40.0 * v + 30.0 * sinpi(4.0 * u),
    ))
    for u in range(0.0, 1.0; length = 9), v in range(0.0, 1.0; length = 5)
]

@testset "cell_range_extent spherical: exhaustive 8×4 injective warp" begin
    grid = CellBasedGrid(_WARPED_SPH_PTS)
    nrectangles = 0
    for ilo in 1:8, ihi in ilo:8, jlo in 1:4, jhi in jlo:4
        irange, jrange = ilo:ihi, jlo:jhi
        cap = cell_range_extent(grid, irange, jrange)
        @test cap.radius <= Float64(pi) / 2
        _test_dense_edge_coverage(cap, _WARPED_SPH_PTS, irange, jrange)
        nrectangles += 1
    end
    @test nrectangles == 360

    corners = (_WARPED_SPH_PTS[1, 1], _WARPED_SPH_PTS[9, 1],
        _WARPED_SPH_PTS[9, 5], _WARPED_SPH_PTS[1, 5])
    corner_center = LinearAlgebra.normalize(sum(corners) / 4)
    corner_radius = nextfloat(
        maximum(p -> GO.spherical_distance(corner_center, p), corners) * 1.0001)
    @test count(p -> GO.spherical_distance(corner_center, p) > corner_radius,
        _WARPED_SPH_PTS) > 0
    @test cell_range_extent(grid, 1:8, 1:4).radius > corner_radius
end

const _GLOBAL_SPH_LONS = collect(range(-180.0, 180.0; length = 9))
const _GLOBAL_SPH_LATS = collect(range(-90.0, 90.0; length = 5))
const _GLOBAL_SPH_PTS = [
    GO.UnitSphereFromGeographic()((lon, lat))
    for lon in _GLOBAL_SPH_LONS, lat in _GLOBAL_SPH_LATS
]

@testset "cell_range_extent spherical: global, stripe, and polar ranges" begin
    regular = RegularGrid(GOCore.Spherical(), _GLOBAL_SPH_LONS, _GLOBAL_SPH_LATS)
    cellbased = CellBasedGrid(_GLOBAL_SPH_PTS)

    global_corners = (_GLOBAL_SPH_PTS[1, 1], _GLOBAL_SPH_PTS[9, 1],
        _GLOBAL_SPH_PTS[9, 5], _GLOBAL_SPH_PTS[1, 5])
    global_mean_norm = LinearAlgebra.norm(sum(global_corners) / 4)
    @test global_mean_norm <= eps(Float64)

    for grid in (regular, cellbased)
        global_cap = cell_range_extent(grid, 1:8, 1:4)
        @test _is_canonical_whole_sphere(global_cap)
        _test_dense_edge_coverage(global_cap, _GLOBAL_SPH_PTS, 1:8, 1:4)

        stripe_cap = cell_range_extent(grid, 1:8, 2:3)
        @test _is_canonical_whole_sphere(stripe_cap)
        _test_dense_edge_coverage(stripe_cap, _GLOBAL_SPH_PTS, 1:8, 2:3)

        polar_cap = cell_range_extent(grid, 1:8, 4:4)
        @test polar_cap.radius <= Float64(pi) / 2
        @test !_is_canonical_whole_sphere(polar_cap)
        _test_dense_edge_coverage(polar_cap, _GLOBAL_SPH_PTS, 1:8, 4:4)
    end
end

const _GLOBAL_SPH_PTS32 = _float32_point.(_GLOBAL_SPH_PTS)

@testset "cell_range_extent spherical: Float32 global and cached traversal" begin
    regular = RegularGrid(GOCore.Spherical(),
        Float32.(_GLOBAL_SPH_LONS), Float32.(_GLOBAL_SPH_LATS))
    cellbased = CellBasedGrid(_GLOBAL_SPH_PTS32)
    regular_points = [
        ConservativeRegridding.Trees.getvertex(regular, i, j)
        for i in axes(_GLOBAL_SPH_PTS32, 1), j in axes(_GLOBAL_SPH_PTS32, 2)
    ]

    for (grid, points) in ((regular, regular_points), (cellbased, _GLOBAL_SPH_PTS32))
        global_cap = cell_range_extent(grid, 1:8, 1:4)
        @test global_cap isa GO.UnitSpherical.SphericalCap{Float32}
        @test _is_canonical_whole_sphere(global_cap)
        _test_dense_edge_coverage(global_cap, points, 1:8, 1:4)

        stripe_cap = cell_range_extent(grid, 1:8, 2:3)
        @test stripe_cap isa GO.UnitSpherical.SphericalCap{Float32}
        @test _is_canonical_whole_sphere(stripe_cap)
        _test_dense_edge_coverage(stripe_cap, points, 1:8, 2:3)
    end

    # The root is a whole-sphere fallback while narrower children are finite. The
    # cached serial descent stores child extents in a root-typed vector, so this
    # catches any Float64 fallback mixed into an otherwise Float32 cursor.
    tree = TopDownQuadtreeCursor(cellbased)
    @test GO.SpatialTreeInterface.node_extent(tree) isa
        GO.UnitSpherical.SphericalCap{Float32}
    grandchild_extents = [
        GO.SpatialTreeInterface.node_extent(grandchild)
        for child in GO.SpatialTreeInterface.getchild(tree)
        for grandchild in GO.SpatialTreeInterface.getchild(child)
    ]
    @test all(cap -> cap isa GO.UnitSpherical.SphericalCap{Float32}, grandchild_extents)
    @test any(cap -> cap.radius <= Float32(pi) / 2, grandchild_extents)
    seen = Ref(0)
    result = ConservativeRegridding.cached_dual_depth_first_search(
        Extents.intersects, tree, tree) do i, j
        seen[] += 1
        return nothing
    end
    @test result === nothing
    @test seen[] > 0
end

@testset "cell_range_extent spherical: near-zero corner mean falls back" begin
    δ = eps(Float64) / 2
    p1 = GO.UnitSpherical.UnitSphericalPoint(1.0, 0.0, 0.0)
    p2 = GO.UnitSpherical.UnitSphericalPoint(-1.0, 0.0, 0.0)
    p3 = GO.UnitSpherical.UnitSphericalPoint(0.0, 1.0, 0.0)
    p4 = GO.UnitSpherical.UnitSphericalPoint(δ, -1.0, 0.0)
    points = reshape(typeof(p1)[p1, p2, p4, p3], 2, 2)
    mean_norm = LinearAlgebra.norm((p1 + p2 + p3 + p4) / 4)
    @test 0 < mean_norm <= eps(Float64)
    @test _is_canonical_whole_sphere(
        cell_range_extent(CellBasedGrid(points), 1:1, 1:1))
end

@testset "cell_range_extent spherical: non-finite inputs fall back" begin
    to_sphere = GO.UnitSphereFromGeographic()
    invalid = GO.UnitSpherical.UnitSphericalPoint(NaN, NaN, NaN)

    perimeter_invalid = [
        to_sphere((-20.0 + 20.0 * (i - 1), -10.0 + 20.0 * (j - 1)))
        for i in 1:3, j in 1:2
    ]
    perimeter_invalid[2, 1] = invalid
    @test _is_canonical_whole_sphere(
        cell_range_extent(CellBasedGrid(perimeter_invalid), 1:2, 1:1))

    corner_invalid = copy(perimeter_invalid)
    corner_invalid[2, 1] = to_sphere((0.0, -10.0))
    corner_invalid[1, 1] = invalid
    @test _is_canonical_whole_sphere(
        cell_range_extent(CellBasedGrid(corner_invalid), 1:2, 1:1))
end

@testset "cell_range_extent spherical: wide finite cap can miss an edge" begin
    to_sphere = GO.UnitSphereFromGeographic()
    sw, se, ne, nw = to_sphere.((
        (-110.0, -70.0), (30.0, -40.0), (140.0, 20.0), (-170.0, 80.0),
    ))
    points = reshape(typeof(sw)[sw, se, nw, ne], 2, 2)

    # Reconstruct the former finite cap, including its explicit midpoint samples.
    corners = (sw, se, ne, nw)
    center = LinearAlgebra.normalize(sum(corners) / 4)
    midpoints = ntuple(i -> GO.UnitSpherical.slerp(
        corners[i], corners[mod1(i + 1, 4)], 0.5), 4)
    old_radius = maximum(p -> GO.spherical_distance(center, p),
        (corners..., midpoints...)) * 1.0001
    escaped_edge_point = GO.UnitSpherical.slerp(nw, sw, 0.3275)
    @test Float64(pi) / 2 < old_radius < Float64(pi)
    @test old_radius < GO.spherical_distance(center, escaped_edge_point)

    cap = cell_range_extent(CellBasedGrid(points), 1:1, 1:1)
    @test _is_canonical_whole_sphere(cap)
    _test_dense_edge_coverage(cap, points, 1:1, 1:1)
end

@testset "getvertex returns the vertex at point-index (i,j)" begin
    cbg = _cellbased_sph()
    rg  = _regular_sph()
    gv  = ConservativeRegridding.Trees.getvertex
    # CellBasedGrid stores the points directly
    @test gv(cbg, 1, 1) == _SPH_PTS[1, 1]
    @test gv(cbg, 2, 3) == _SPH_PTS[2, 3]
    @test gv(cbg, 5, 4) == _SPH_PTS[5, 4]
    # RegularGrid converts (lon, lat) → unit sphere on demand
    @test gv(rg, 1, 1) == GO.UnitSphereFromGeographic()((_SPH_LONS[1], _SPH_LATS[1]))
    @test gv(rg, 2, 3) == GO.UnitSphereFromGeographic()((_SPH_LONS[2], _SPH_LATS[3]))
    @test gv(rg, 5, 4) == GO.UnitSphereFromGeographic()((_SPH_LONS[5], _SPH_LATS[4]))
end

# Expected border-ring (i,j) order: west column, east column, then the interiors
# of the south and north rows (so each corner is yielded once, by the columns).
function _expected_ring_ij(imin, imax, jmin, jmax)
    ij = Tuple{Int,Int}[]
    for j in jmin:jmax;         push!(ij, (imin, j)); end   # west column
    for j in jmin:jmax;         push!(ij, (imax, j)); end   # east column
    for i in (imin + 1):(imax - 1); push!(ij, (i, jmin)); end  # south row interior
    for i in (imin + 1):(imax - 1); push!(ij, (i, jmax)); end  # north row interior
    return ij
end

@testset "CurvilinearGridPerimeterPoints yields the border ring in order" begin
    cbg = _cellbased_sph()
    pts = _SPH_PTS
    PP  = ConservativeRegridding.Trees.CurvilinearGridPerimeterPoints
    # full grid, sub-range, single cell, thin-in-i (W=2), thin-in-j (H=2)
    for (imin, imax, jmin, jmax) in
            ((1, 5, 1, 4), (2, 4, 1, 3), (1, 2, 1, 2), (1, 2, 1, 4), (1, 5, 2, 3))
        it = PP(cbg, imin, imax, jmin, jmax)
        expected = [pts[i, j] for (i, j) in _expected_ring_ij(imin, imax, jmin, jmax)]
        @test collect(it) == expected
        @test length(it) == length(expected)
        @test length(it) == 2 * (imax - imin + 1) + 2 * (jmax - jmin + 1) - 4
    end
end

# Declared manifolds.  The WGS84 authalic (equal-area) sphere below is 1.6 m smaller than
# `Spherical()`'s mean radius, so a radius that silently reverted to the default shows up
# in the area invariants.
const _AUTHALIC = GOCore.Spherical(; radius = 6371007.180918474)   # WGS84 R_A

_authalic_corners(nlon, nlat) = [
    GO.UnitSphereFromGeographic()((lon, lat))
    for lon in range(-180, 180; length = nlon), lat in range(-90, 90; length = nlat)
]

@testset "best_manifold reports a grid's declared manifold" begin
    points   = make_lonlat_point_matrix(4, 3)
    polygons = make_polygon_matrix(points)

    # The one-argument constructors keep guessing exactly as they always have.
    @test GOCore.best_manifold(ExplicitPolygonGrid(polygons)) == GOCore.Planar()
    @test GOCore.best_manifold(CellBasedGrid(points)) == GOCore.Planar()
    @test GOCore.best_manifold(CellBasedGrid(_SPH_PTS)) == GOCore.Spherical()
    @test GOCore.best_manifold(RegularGrid(_SPH_LONS, _SPH_LATS)) == GOCore.Planar()

    for grid in (ExplicitPolygonGrid(_AUTHALIC, polygons),
                 CellBasedGrid(_AUTHALIC, _SPH_PTS),
                 RegularGrid(_AUTHALIC, _SPH_LONS, _SPH_LATS))
        @test GOCore.best_manifold(grid) == GOCore.manifold(grid) == _AUTHALIC
    end
end

@testset "grids are ConstructionBase-compatible" begin
    # `setproperties` is what `treeify`'s manifold override is built on.
    cbg = ConstructionBase.setproperties(CellBasedGrid(_AUTHALIC, _SPH_PTS), (; manifold = GO.Planar()))
    @test GOCore.manifold(cbg) === GO.Planar()
    @test cbg.points === _SPH_PTS               # geometry shared, not copied

    polygons = make_polygon_matrix(_SPH_PTS)
    epg = ConstructionBase.setproperties(ExplicitPolygonGrid(_AUTHALIC, polygons), (; manifold = GO.Planar()))
    @test GOCore.manifold(epg) === GO.Planar()
    @test epg.polygons === polygons

    rg = ConstructionBase.setproperties(RegularGrid(_AUTHALIC, _SPH_LONS, _SPH_LATS), (; manifold = GO.Planar()))
    @test GOCore.manifold(rg) === GO.Planar()
    @test rg.x === _SPH_LONS && rg.y === _SPH_LATS
end

@testset "treeify overrides the grid's manifold with the one it is given" begin
    treeify = ConservativeRegridding.Trees.treeify
    getgrid = ConservativeRegridding.Trees.getgrid
    grid = CellBasedGrid(_AUTHALIC, _SPH_PTS)

    # One-argument `treeify` routes through `best_manifold`, i.e. the grid's own.
    tree = treeify(grid)
    @test tree isa TopDownQuadtreeCursor
    @test getgrid(tree) === grid
    @test GOCore.best_manifold(tree) == _AUTHALIC

    # Naming the grid's own manifold is a pass-through, not a rebuild.
    @test getgrid(treeify(_AUTHALIC, grid)) === grid

    # A different manifold overrides the grid's own, so the tree stays self-consistent
    # (this is what `cell_range_extent` dispatches on).
    for m in (GOCore.Spherical(), GOCore.Spherical(; radius = 1.0), GO.Planar())
        overridden = treeify(m, grid)
        @test overridden isa TopDownQuadtreeCursor
        @test GOCore.manifold(getgrid(overridden)) === m
        @test GOCore.best_manifold(overridden) === m
        @test getgrid(overridden).points === grid.points   # geometry shared, not copied
    end

    for g in (ExplicitPolygonGrid(_AUTHALIC, make_polygon_matrix(_SPH_PTS)),
              CellBasedGrid(_AUTHALIC, _SPH_PTS),
              RegularGrid(_AUTHALIC, _SPH_LONS, _SPH_LATS))
        @test GOCore.manifold(getgrid(treeify(GO.Planar(), g))) === GO.Planar()
    end
end

@testset "Regridder carries a declared radius end to end" begin
    src = CellBasedGrid(_AUTHALIC, _authalic_corners(9, 5))
    dst = CellBasedGrid(_AUTHALIC, _authalic_corners(7, 4))

    r = ConservativeRegridding.Regridder(dst, src; normalize = false, threaded = false)
    @test size(r) == (18, 32)

    # Both grids tile the whole sphere, so areas and intersections sum to 4πR² -
    # at the declared radius, not at `Spherical()`'s.
    sphere = 4π * _AUTHALIC.radius^2
    @test sum(r.intersections) ≈ sphere
    @test sum(r.src_areas) ≈ sphere
    @test sum(r.dst_areas) ≈ sphere
    @test !isapprox(sphere, 4π * GOCore.Spherical().radius^2; rtol = 1e-8)  # the radii really do differ
end

@testset "Regridder refuses to guess between two manifolds" begin
    src = CellBasedGrid(_AUTHALIC, _authalic_corners(9, 5))

    # Two spheres of different radii.
    dst_mean_radius = CellBasedGrid(GOCore.Spherical(), _authalic_corners(7, 4))
    @test_throws "manifolds must be the same" ConservativeRegridding.Regridder(dst_mean_radius, src)

    # Planar against spherical is no longer promoted to spherical either.
    dst_planar = CellBasedGrid(GO.Planar(), _authalic_corners(7, 4))
    @test_throws "Regridder(manifold, dst, src)" ConservativeRegridding.Regridder(dst_planar, src)

    # Naming the manifold is what gets you past it.
    r = ConservativeRegridding.Regridder(_AUTHALIC, dst_planar, src; normalize = false, threaded = false)
    @test sum(r.intersections) ≈ 4π * _AUTHALIC.radius^2
end
