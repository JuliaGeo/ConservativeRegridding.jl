using ConservativeRegridding
using Test

using RingGrids
import Healpix
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI
import GridInterface as GrI
using GridInterface: ncells, cell_geometry, cell_area
using GridInterface.Structured: nfaces

const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

# all leaf nodes beneath `node`
leaves(node) = STI.isleaf(node) ? [node] :
    reduce(vcat, (leaves(STI.getchild(node, i)) for i in 1:STI.nchild(node)))

sweep_tree(g) = GrI.treeify(g; leaf_capacity = GrI.ForSweeps())

@testset "HEALPix provider: asgrid" begin
    field = rand(HEALPixGrid, 8)               # nlat_half = 8, nside = 4
    g = ConservativeRegridding.asgrid(field)

    @test g isa RingGridsExt.HEALPixRingGrid
    @test g.nside == 4
    @test ncells(g) == 12 * 4^2                # 192 pixels
    @test nfaces(g) == 12                      # 12 base faces
    @test GOCore.best_manifold(g) == GO.Spherical()
    @test GOCore.best_manifold(field) == GO.Spherical()
    @test ConservativeRegridding.asgrid(field.grid) isa RingGridsExt.HEALPixRingGrid

    tree = sweep_tree(g)
    @test STI.isspatialtree(typeof(tree))
    @test STI.node_extent(tree) isa GO.UnitSpherical.SphericalCap
end

@testset "HEALPix provider: any even nlat_half (no power-of-two requirement)" begin
    # Range subdivision needs no nested index, so every (even) nlat_half works.
    for nlat_half in (2, 4, 6, 8, 10, 12, 16)
        g = ConservativeRegridding.asgrid(rand(HEALPixGrid, nlat_half))
        nside = nlat_half ÷ 2
        @test g isa RingGridsExt.HEALPixRingGrid
        @test g.nside == nside
        @test ncells(g) == 12 * nside^2
        @test sort(GrI.subtree_indices(sweep_tree(g))) == 1:12nside^2
    end
end

@testset "HEALPix provider: the tree partitions the pixels" begin
    g = RingGridsExt.HEALPixRingGrid(4)        # nside = 4
    tree = sweep_tree(g)
    @test STI.isleaf(tree) == false

    # Leaves partition the faces; every pixel is emitted exactly once: 12·nside² total.
    allcells = reduce(vcat, [collect(STI.child_indices_extents(l)) for l in leaves(tree)])
    @test length(allcells) == 12 * 4^2
    @test sort(first.(allcells)) == 1:12 * 4^2
end

@testset "HEALPix provider: cell corners match Healpix.boundariesRing" begin
    pdist(p, q) = maximum(abs.(Tuple(p) .- q))
    for nside in (1, 2, 4, 8)                   # Healpix.jl oracle is power-of-two only
        g = RingGridsExt.HEALPixRingGrid(nside)
        res  = Healpix.Resolution(nside)
        maxerr = 0.0
        for ipix in 1:12nside^2
            pts = collect(GI.getpoint(GI.getexterior(cell_geometry(g, ipix))))
            @test length(pts) == 5             # 4 corners + closing point
            bd = Healpix.boundariesRing(res, ipix, 1, Float64)
            for k in 1:4
                maxerr = max(maxerr, pdist(pts[k], bd[k, :]))
            end
        end
        @test maxerr < 1e-12
    end
end

@testset "HEALPix provider: node extents contain the cells beneath" begin
    incap(p, cap) = GO.spherical_distance(cap.point, p) <= cap.radius + 1e-9

    for nside in (4, 5)                         # power-of-two and non-power-of-two
        g = RingGridsExt.HEALPixRingGrid(nside)
        tree = sweep_tree(g)

        # Each child's cap must contain every corner of every cell beneath it.
        for i in 1:STI.nchild(tree)
            face = STI.getchild(tree, i)
            cap  = STI.node_extent(face)
            @test cap isa GO.UnitSpherical.SphericalCap
            for leaf in leaves(face), (ij, _) in STI.child_indices_extents(leaf)
                for p in GI.getpoint(GI.getexterior(cell_geometry(g, ij)))
                    @test incap(p, cap)
                end
            end
        end

        # Leaf cells map bijectively onto the ring-order data indices 1:npix.
        ringidxs = Int[]
        for leaf in leaves(tree), (idx, ext) in STI.child_indices_extents(leaf)
            @test ext isa GO.UnitSpherical.SphericalCap
            push!(ringidxs, idx)
        end
        @test sort(ringidxs) == collect(1:12nside^2)
    end
end

@testset "HEALPix: cells tessellate the sphere" begin
    S = GO.Spherical()
    sphere = 4π * S.radius^2
    for nside in (1, 2, 4, 3, 5)               # power-of-two and non-power-of-two
        g = RingGridsExt.HEALPixRingGrid(nside)
        total = sum(cell_area(g, i) for i in 1:ncells(g))
        @test isapprox(total, sphere; rtol = 1e-12)
    end
end

@testset "HEALPix: end-to-end regridding + conservation" begin
    # Power-of-two resolutions.
    src = rand(HEALPixGrid, 8); dst = rand(HEALPixGrid, 16)
    R = ConservativeRegridding.Regridder(dst, src)
    @test R isa ConservativeRegridding.Regridder
    @test size(R.intersections) == (length(dst), length(src))

    src_ones = ones(Float64, length(src)); dst_out = zeros(Float64, length(dst))
    ConservativeRegridding.regrid!(dst_out, R, src_ones)
    @test all(isapprox.(dst_out, 1.0; atol = 1e-9))

    src_grid = HEALPixGrid(16); dst_grid = HEALPixGrid(8)
    R2 = ConservativeRegridding.Regridder(dst_grid, src_grid)
    londs, latds = RingGrids.get_londlatds(src_grid)
    src_vals = [2.0 + 0.5 * sin(2 * deg2rad(λ)) * cos(3 * deg2rad(φ)) for (λ, φ) in zip(londs, latds)]
    dst_vals = zeros(Float64, RingGrids.get_npoints(dst_grid))
    ConservativeRegridding.regrid!(dst_vals, R2, src_vals)
    @test isapprox(sum(src_vals .* R2.src_areas), sum(dst_vals .* R2.dst_areas); rtol = 1e-9)

    # Non-power-of-two resolutions (nside 3 → 5) still partition and conserve.
    srcn = HEALPixGrid(6); dstn = HEALPixGrid(10)
    Rn = ConservativeRegridding.Regridder(dstn, srcn)
    outn = zeros(Float64, RingGrids.get_npoints(dstn))
    ConservativeRegridding.regrid!(outn, Rn, ones(Float64, RingGrids.get_npoints(srcn)))
    @test all(isapprox.(outn, 1.0; atol = 1e-9))

    # Cross-grid: HEALPix → full Clenshaw, constant field preserved.
    src3 = rand(HEALPixGrid, 8); dst3 = rand(FullClenshawGrid, 16)
    R3 = ConservativeRegridding.Regridder(dst3, src3)
    dst3_out = zeros(Float64, length(dst3))
    ConservativeRegridding.regrid!(dst3_out, R3, ones(Float64, length(src3)))
    @test all(isapprox.(dst3_out, 1.0; atol = 1e-9))

    # Field-path (primary user API) and the reverse direction via `transpose`.
    dst_field = zeros(HEALPixGrid, 16)
    ConservativeRegridding.regrid!(dst_field, R, ones(HEALPixGrid, 8))
    @test all(x -> isapprox(x, 1; atol = 1e-6), dst_field)
    back = zeros(HEALPixGrid, 8)
    ConservativeRegridding.regrid!(back, transpose(R), ones(HEALPixGrid, 16))
    @test all(x -> isapprox(x, 1; atol = 1e-6), back)
end
