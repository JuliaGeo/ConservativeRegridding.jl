using ConservativeRegridding
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS
import GeometryOps: SpatialTreeInterface as STI
import GridInterface as GrI
using GridInterface: ncells, cell_area

using Oceananigans
using Oceananigans.Grids: RightCenterFolded

const OceananigansExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)

@testset "The U-pivot layout grid keeps the field's index domain" begin
    grid = TripolarGrid(size=(40, 20, 1), fold_topology=RightCenterFolded)
    g = ConservativeRegridding.asgrid(grid)
    @test g isa OceananigansExt.FoldedLayoutGrid
    @test ConservativeRegridding.asgrid(CenterField(grid)) isa OceananigansExt.FoldedLayoutGrid
    Nx, Ny = g.grid.Nx, g.grid.Ny
    @test ncells(g) == Nx * Ny
    @test GO.best_manifold(g) == GO.best_manifold(grid)

    # the fold row's duplicates are ghosts: zero area, a degenerate ring, never emitted
    members = collect(GrI.parentpositions(OceananigansExt.unique_cells(g.grid)))
    ghosts = setdiff(1:ncells(g), members)
    @test length(ghosts) == Nx ÷ 2
    @test all(k -> cell_area(g, k) == 0.0, ghosts)
    @test all(k -> GrI.cell_extent(g, k).radius == 0.0, ghosts)
    @test !any(k -> GrI.cell_contains(g, k, GrI.cell_centroid(g, 1)), ghosts)   # a ghost contains nothing
    @test all(k -> cell_area(g, k) == cell_area(g.grid, k), members)

    tree = GrI.treeify(g; leaf_capacity = GrI.ForSweeps())
    @test tree isa GrI.Structured.MaskedCursor
    @test sort(GrI.subtree_indices(tree)) == members
    emitted = Int[]
    STI.depth_first_search(_ -> true, tree) do i
        push!(emitted, i)
    end
    @test sort(emitted) == members

    # the regridder is sized on the layout, its areas are the layout's
    R = ConservativeRegridding.Regridder(grid, grid; threaded = false)
    @test size(R) == (Nx * Ny, Nx * Ny)
    @test all(k -> R.src_areas[k] == 0.0, ghosts)
    @test all(k -> R.intersections[k, k] ≈ cell_area(g, k), members)
end

@testset "An immersed-boundary grid regrids its underlying layout" begin
    bathy(λ, φ) = φ > 60 ? 10.0 : -1000.0
    grid = TripolarGrid(size=(40, 20, 2), z=(-1000, 0))
    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bathy))
    @test ConservativeRegridding.asgrid(ibg) isa OceananigansExt.FoldedLayoutGrid
    @test ncells(ConservativeRegridding.asgrid(ibg)) == 40 * 20
    ll = LatitudeLongitudeGrid(size=(20, 10, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    @test ConservativeRegridding.asgrid(ll) isa OceananigansExt.OceananigansLatLonGrid
end
