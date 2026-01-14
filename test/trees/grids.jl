using ConservativeRegridding.Trees
using Test
import GeoInterface as GI

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
