using ConservativeRegridding
using ConservativeRegridding.Trees
using Test

using RingGrids
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI

# Reduced grids without a native tree are treeified by materializing a vector of
# cell polygons (RingGrids' E/S/W/N vertices) and R-treeing it.
const REDUCED_GRIDS = (OctahedralGaussianGrid, OctahedralClenshawGrid, OctaminimalGaussianGrid)

@testset "$G: treeify" for G in REDUCED_GRIDS
    field = rand(G, 6)
    tree = Trees.treeify(GO.Spherical(), field)
    @test tree isa Trees.GeometryMaintainingTreeWrapper
    @test STI.isspatialtree(tree)
    @test Trees.ncells(tree) == length(field)
    @test Trees.cell_index_count(tree) == length(field)
    @test GOCore.best_manifold(field) == GO.Spherical()

    # Cells tile the sphere and are counterclockwise (positive area).
    cell_areas = [GO.area(GO.Spherical(; radius = 1.0), c) for c in Trees.getcell(tree)]
    @test all(>(0), cell_areas)
    @test sum(cell_areas) ≈ 4π rtol = 1e-10
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

@testset "Reduced ↔ reduced and native trees" begin
    for (dst, src) in ((rand(OctaminimalGaussianGrid, 10), rand(OctahedralClenshawGrid, 8)),
                       (rand(OctaHEALPixGrid, 8), rand(OctahedralGaussianGrid, 10)),
                       (rand(OctahedralGaussianGrid, 8), rand(HEALPixGrid, 8)))
        R = ConservativeRegridding.Regridder(dst, src)
        out = zeros(length(dst))
        ConservativeRegridding.regrid!(out, R, ones(length(src)))
        @test all(isapprox.(out, 1.0; atol = 1e-8))
    end
end
