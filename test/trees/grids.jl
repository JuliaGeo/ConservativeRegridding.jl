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
import LinearAlgebra

# Regional spherical grids — chosen away from poles/antipodes so the bounding cap
# is non-degenerate. (A pole-to-pole range has antipodal corners whose mean is the
# origin, giving a NaN center; that pre-existing edge case is out of scope here.)
const _SPH_LONS = collect(range(0.0, 40.0, length = 5))   # 4 cells along i
const _SPH_LATS = collect(range(0.0, 30.0, length = 4))   # 3 cells along j
const _SPH_LONLAT = [(lon, lat) for lon in _SPH_LONS, lat in _SPH_LATS]
const _SPH_PTS = GO.UnitSphereFromGeographic().(_SPH_LONLAT)
_regular_sph()   = RegularGrid(GOCore.Spherical(), _SPH_LONS, _SPH_LATS)
_cellbased_sph() = CellBasedGrid(_SPH_PTS)   # UnitSphericalPoint eltype ⇒ Spherical

@testset "cell_range_extent spherical: bounding cap (characterization)" begin
    pts = _SPH_PTS
    for (ir, jr) in ((1:4, 1:3), (1:1, 1:1), (2:3, 1:2), (1:4, 2:2), (1:2, 1:3))
        imin, imax = first(ir), last(ir) + 1
        jmin, jmax = first(jr), last(jr) + 1
        expected_center = LinearAlgebra.normalize(
            (pts[imin, jmin] + pts[imax, jmin] + pts[imax, jmax] + pts[imin, jmax]) / 4)
        for g in (_regular_sph(), _cellbased_sph())
            cap = cell_range_extent(g, ir, jr)
            # the cap must bound every vertex of the requested cell range …
            @test all(GO.spherical_distance(cap.point, pts[i, j]) <= cap.radius
                      for i in imin:imax, j in jmin:jmax)
            # … and be centered on the normalized mean of the 4 range corners
            @test isapprox(cap.point, expected_center; atol = 1e-12)
        end
    end
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
