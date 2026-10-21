using ConservativeRegridding
using Test

using RingGrids
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
import GridInterface as GrI
using GridInterface: ncells, cell_geometry, cell_area

const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

# Reduced grids without a face structure are stored cells (RingGrids' E/N/W/S
# vertices) under GridInterface's generic R-tree.
const REDUCED_GRIDS = (OctahedralGaussianGrid, OctahedralClenshawGrid, OctaminimalGaussianGrid)

@testset "$G: asgrid + treeify" for G in REDUCED_GRIDS
    field = rand(G, 6)
    g = ConservativeRegridding.asgrid(field)
    @test g isa RingGridsExt.ReducedRingGrid
    @test ncells(g) == length(field)
    @test GOCore.best_manifold(g) == GO.Spherical()
    @test GOCore.best_manifold(field) == GO.Spherical()
    tree = GrI.treeify(g; leaf_capacity = GrI.ForSweeps())
    @test STI.isspatialtree(typeof(tree))
    @test sort(GrI.subtree_indices(tree)) == 1:ncells(g)

    # Cells tile the sphere and are counterclockwise (positive area).
    cell_areas = [cell_area(g, i) for i in 1:ncells(g)]
    @test all(>(0), cell_areas)
    @test sum(cell_areas) ≈ 4π * GO.Spherical().radius^2 rtol = 1e-10
end

@testset "$G → FullGaussianGrid: conservation" for G in REDUCED_GRIDS
    src = rand(G, 12)
    dst = rand(FullGaussianGrid, 9)
    R = ConservativeRegridding.Regridder(dst, src)
    @test size(R.intersections) == (length(dst), length(src))

    # A constant field stays constant in both directions.
    dst_out = zeros(length(dst))
    ConservativeRegridding.regrid!(dst_out, R, ones(length(src)))
    @test all(isapprox.(dst_out, 1.0; atol = 1e-8))
    src_out = zeros(length(src))
    ConservativeRegridding.regrid!(src_out, transpose(R), ones(length(dst)))
    @test all(isapprox.(src_out, 1.0; atol = 1e-8))

    # Area-weighted total is preserved for a random field.
    src_vals = collect(Float64, src.data)
    ConservativeRegridding.regrid!(dst_out, R, src_vals)
    @test sum(dst_out .* R.dst_areas) ≈ sum(src_vals .* R.src_areas) rtol = 1e-10
end

@testset "Reduced ↔ reduced and face grids" begin
    for (dst, src) in ((rand(OctaminimalGaussianGrid, 10), rand(OctahedralClenshawGrid, 8)),
                       (rand(OctaHEALPixGrid, 8), rand(OctahedralGaussianGrid, 10)),
                       (rand(OctahedralGaussianGrid, 8), rand(HEALPixGrid, 8)))
        R = ConservativeRegridding.Regridder(dst, src)
        out = zeros(length(dst))
        ConservativeRegridding.regrid!(out, R, ones(length(src)))
        @test all(isapprox.(out, 1.0; atol = 1e-8))
    end
end
