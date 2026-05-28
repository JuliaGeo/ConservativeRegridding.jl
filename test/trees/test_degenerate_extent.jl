using Test
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeometryOps as GO
using GeometryOps.UnitSpherical: UnitSphericalPoint

@testset "CellBasedGrid SphericalCap extent — degenerate cases" begin
    manifold = GO.Spherical()

    # Regression test: a (nlon+1, 2) grid spanning 180° of longitude across
    # the equator produces corner points whose Cartesian sum is exactly
    # (0, 0, 0). Before the fix, normalize((0,0,0)) → NaN poisoned the
    # SphericalCap radius, silently breaking the dual DFS.
    @testset "half-sphere equatorial band (symmetric cancellation)" begin
        to_sphere = GO.UnitSphereFromGeographic()
        nlon = 8  # 8 cells spanning 0°–180° (half the sphere)
        dlon = 180.0 / nlon
        pts = Matrix{UnitSphericalPoint{Float64}}(undef, nlon + 1, 2)
        for k in 1:(nlon + 1)
            lon = (k - 1) * dlon
            pts[k, 1] = to_sphere((lon, -15.0))  # south face
            pts[k, 2] = to_sphere((lon, +15.0))  # north face
        end
        grid = Trees.CellBasedGrid(manifold, pts)

        # Full-range extent: all 8 cells
        cap = Trees.cell_range_extent(grid, 1:nlon, 1:1)
        @test isfinite(cap.radius)
        @test cap.radius > 0

        # The cap should cover roughly a hemisphere (180° span in lon)
        @test cap.radius > π / 4   # at least 45°
        @test cap.radius < π + 0.1 # at most slightly over π
    end

    @testset "full-sphere equatorial ring (360° longitude)" begin
        to_sphere = GO.UnitSphereFromGeographic()
        nlon = 16  # full ring
        dlon = 360.0 / nlon
        pts = Matrix{UnitSphericalPoint{Float64}}(undef, nlon + 1, 2)
        for k in 1:(nlon + 1)
            lon = (k - 1) * dlon
            pts[k, 1] = to_sphere((lon, -15.0))
            pts[k, 2] = to_sphere((lon, +15.0))
        end
        grid = Trees.CellBasedGrid(manifold, pts)

        # Top-level extent covers the full ring
        cap = Trees.cell_range_extent(grid, 1:nlon, 1:1)
        @test isfinite(cap.radius)
        @test cap.radius > π / 2  # should cover > 90° (it's the full equatorial belt)

        # Sub-range: half the ring (cells 1:8 → 0°–180°, the degenerate case)
        half_cap = Trees.cell_range_extent(grid, 1:(nlon ÷ 2), 1:1)
        @test isfinite(half_cap.radius)
        @test half_cap.radius > 0
    end

    @testset "non-degenerate mid-latitude band" begin
        # Sanity: a mid-latitude band should never trigger the fallback
        to_sphere = GO.UnitSphereFromGeographic()
        nlon = 12
        dlon = 360.0 / nlon
        pts = Matrix{UnitSphericalPoint{Float64}}(undef, nlon + 1, 2)
        for k in 1:(nlon + 1)
            lon = (k - 1) * dlon
            pts[k, 1] = to_sphere((lon, 30.0))
            pts[k, 2] = to_sphere((lon, 60.0))
        end
        grid = Trees.CellBasedGrid(manifold, pts)

        cap = Trees.cell_range_extent(grid, 1:nlon, 1:1)
        @test isfinite(cap.radius)
        # Center should be near (0, 0, sin(45°)) ≈ north of equator
        @test cap.point[3] > 0.5
    end

    @testset "regridder finds intersections for equatorial ring" begin
        # End-to-end: build a regridder where the source is an equatorial
        # ring grid (the case that was broken). Verify nonzero intersections.
        # Both src and dst use CellBasedGrid with UnitSphericalPoint vertices
        # so that the intersection algorithm gets consistent polygon types.
        to_sphere = GO.UnitSphereFromGeographic()

        # Source: equatorial ring (16 cells, lon 0°–360°, lat ±15°)
        nlon = 16
        dlon = 360.0 / nlon
        src_pts = Matrix{UnitSphericalPoint{Float64}}(undef, nlon + 1, 2)
        for k in 1:(nlon + 1)
            lon = (k - 1) * dlon
            src_pts[k, 1] = to_sphere((lon, -15.0))
            src_pts[k, 2] = to_sphere((lon, +15.0))
        end

        # Destination: small CellBasedGrid lat-lon (24×12)
        dst_pts = Matrix{UnitSphericalPoint{Float64}}(undef, 25, 13)
        lons_d = range(0.0, 360.0, length=25)
        lats_d = range(-90.0, 90.0, length=13)
        for j in 1:13, i in 1:25
            dst_pts[i, j] = to_sphere((lons_d[i], lats_d[j]))
        end

        dst_tree = Trees.KnownFullSphereExtentWrapper(
            Trees.TopDownQuadtreeCursor(Trees.CellBasedGrid(manifold, dst_pts)))
        src_tree = Trees.KnownFullSphereExtentWrapper(
            Trees.TopDownQuadtreeCursor(Trees.CellBasedGrid(manifold, src_pts)))

        r = ConservativeRegridding.Regridder(manifold, dst_tree, src_tree;
                                             normalize = false)
        @test length(r.intersections.nzval) > 0
        # All 16 source cells should have at least one intersection
        using SparseArrays
        for col in 1:nlon
            @test nnz(r.intersections[:, col]) > 0
        end
    end
end
