using ConservativeRegridding
using ConservativeRegridding.Trees
using Test

using RingGrids
using SmallCollections: SmallVector, capacity
import Healpix
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI

const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

# all leaf nodes beneath `node`
leaves(node) = STI.isleaf(node) ? [node] :
    reduce(vcat, (leaves(STI.getchild(node, i)) for i in 1:STI.nchild(node)))

@testset "HEALPix tree: root node + treeify" begin
    field = rand(HEALPixGrid, 8)               # nlat_half = 8, nside = 4
    tree  = Trees.treeify(GO.Spherical(), field)

    @test tree isa RingGridsExt.HEALPixRootNode
    @test tree.nside == 4
    @test Trees.ncells(tree) == 12 * 4^2       # 192 pixels

    @test STI.isspatialtree(typeof(tree)) == true
    @test STI.isleaf(tree) == false
    @test STI.nchild(tree) == 12               # 12 base faces
    @test STI.node_extent(tree) isa GO.UnitSpherical.SphericalCap
    @test GOCore.best_manifold(tree) == GO.Spherical()

    @test Trees.treeify(GO.Spherical(), field.grid) isa RingGridsExt.HEALPixRootNode
end

@testset "HEALPix tree: any even nlat_half (no power-of-two requirement)" begin
    # Range subdivision needs no nested index, so every (even) nlat_half works.
    for nlat_half in (2, 4, 6, 8, 10, 12, 16)
        tree = Trees.treeify(GO.Spherical(), rand(HEALPixGrid, nlat_half))
        nside = nlat_half ÷ 2
        @test tree isa RingGridsExt.HEALPixRootNode
        @test tree.nside == nside
        @test Trees.ncells(tree) == 12 * nside^2
    end
end

@testset "HEALPix tree: parent/child traversal" begin
    tree = RingGridsExt.HEALPixRootNode(4)     # nside = 4

    @test STI.nchild(tree) == 12
    face = STI.getchild(tree, 1)
    @test face isa Trees.TopDownQuadtreeCursor
    @test face.grid isa RingGridsExt.HEALPixFaceGrid
    @test face.grid.face == 0
    @test face.leafranges == (1:4, 1:4)
    @test Trees.cell_index_count(face) == 12 * 4^2
    @test STI.getchild(tree, 12).grid.face == 11
    @test STI.isleaf(face) == false
    @test STI.nchild(face) == 4                # 4×4 block → quartered

    # A 4×4 face quarters into four 2×2 cursor leaves; each enumerates its 4 cells.
    leaf = STI.getchild(face, 1)
    @test leaf isa Trees.TopDownQuadtreeCursor && leaf.grid.face == 0
    @test length.(leaf.leafranges) == (2, 2)
    @test STI.isleaf(leaf) == true
    entries = @inferred STI.child_indices_extents(leaf)
    @test entries isa SmallVector{4}
    @test capacity(entries) == 4
    @test length(entries) == 4

    # Leaves partition each face; every pixel is emitted exactly once: 12·nside² total.
    allcells = reduce(vcat, [collect(STI.child_indices_extents(l)) for l in leaves(tree)])
    @test length(allcells) == 12 * 4^2
end

@testset "HEALPix tree: `getcell` corners match Healpix.boundariesRing" begin
    pdist(p, q) = maximum(abs.(Tuple(p) .- q))
    for nside in (1, 2, 4, 8)                   # Healpix.jl oracle is power-of-two only
        tree = RingGridsExt.HEALPixRootNode(nside)
        res  = Healpix.Resolution(nside)
        maxerr = 0.0
        for ipix in 1:12nside^2
            pts = collect(GI.getpoint(GI.getexterior(Trees.getcell(tree, ipix))))
            @test length(pts) == 5             # 4 corners + closing point
            bd = Healpix.boundariesRing(res, ipix, 1, Float64)
            for k in 1:4
                maxerr = max(maxerr, pdist(pts[k], bd[k, :]))
            end
        end
        @test maxerr < 1e-12
    end

    tree = RingGridsExt.HEALPixRootNode(2)
    @test length(collect(Trees.getcell(tree))) == Trees.ncells(tree)
end

@testset "HEALPix tree: node_extent + child_indices_extents" begin
    incap(p, cap) = GO.spherical_distance(cap.point, p) <= cap.radius + 1e-9

    for nside in (4, 5)                         # power-of-two and non-power-of-two
        tree = RingGridsExt.HEALPixRootNode(nside)

        # Each face cap must contain every corner of every cell beneath it.
        for i in 1:STI.nchild(tree)
            face = STI.getchild(tree, i)
            cap  = STI.node_extent(face)
            @test cap isa GO.UnitSpherical.SphericalCap
            for leaf in leaves(face), (ij, _) in STI.child_indices_extents(leaf)
                for p in GI.getpoint(GI.getexterior(Trees.getcell(tree, ij)))
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
        tree = RingGridsExt.HEALPixRootNode(nside)
        total = sum(GO.area(S, Trees.getcell(tree, i)) for i in 1:Trees.ncells(tree))
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
