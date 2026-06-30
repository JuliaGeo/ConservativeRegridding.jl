using ConservativeRegridding
using ConservativeRegridding.Trees
using Test

using RingGrids
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI

const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

# all leaf nodes beneath `node`
leaves(node) = STI.isleaf(node) ? [node] :
    reduce(vcat, (leaves(STI.getchild(node, i)) for i in 1:STI.nchild(node)))

@testset "OctaHEALPix tree: root node + treeify" begin
    field = rand(OctaHEALPixGrid, 4)          # nlat_half = nside = 4
    tree  = Trees.treeify(GO.Spherical(), field)

    @test tree isa RingGridsExt.OctaHEALPixRootNode
    @test Trees.ncells(tree) == 4 * 4^2        # 64 pixels

    @test STI.isspatialtree(typeof(tree)) == true
    @test STI.isleaf(tree) == false
    @test STI.nchild(tree) == 4                # 4 base faces (octahedral)
    @test STI.node_extent(tree) isa GO.UnitSpherical.SphericalCap
    @test GOCore.best_manifold(tree) == GO.Spherical()

    # treeify on the bare grid works too
    @test Trees.treeify(GO.Spherical(), field.grid) isa RingGridsExt.OctaHEALPixRootNode
end

@testset "OctaHEALPix tree: any nlat_half (no power-of-two requirement)" begin
    # The curvilinear-face design subdivides the (r,c) matrix by index range, not a
    # Morton nest, so every nlat_half works — pow2 and not (the old nested hierarchy
    # required pow2).
    for nside in (1, 2, 3, 4, 5, 7, 8, 16)
        tree = Trees.treeify(GO.Spherical(), rand(OctaHEALPixGrid, nside))
        @test tree isa RingGridsExt.OctaHEALPixRootNode
        @test tree.nside == nside
        @test Trees.ncells(tree) == 4 * nside^2
    end
end

@testset "OctaHEALPix tree: parent/child traversal" begin
    tree = RingGridsExt.OctaHEALPixRootNode(4)        # nside = 4

    @test STI.nchild(tree) == 4
    face = STI.getchild(tree, 1)
    @test face isa Trees.TopDownQuadtreeCursor
    @test face.grid isa RingGridsExt.OctaHEALPixFaceGrid
    @test face.grid.q == 1                      # quadrant 1..4
    @test face.leafranges == (1:4, 1:4)
    @test STI.getchild(tree, 4).grid.q == 4
    @test STI.isleaf(face) == false
    @test STI.nchild(face) == 4                 # 4×4 block → quartered

    # A 4×4 face quarters into four 2×2 cursor leaves; each enumerates its 4 cells.
    leaf = STI.getchild(face, 1)
    @test leaf isa Trees.TopDownQuadtreeCursor && leaf.grid.q == 1
    @test length.(leaf.leafranges) == (2, 2)
    @test STI.isleaf(leaf) == true
    @test length(collect(STI.child_indices_extents(leaf))) == 4

    # Leaves partition each face; every pixel is emitted exactly once: 4·nside² total.
    allcells = reduce(vcat, [collect(STI.child_indices_extents(l)) for l in leaves(tree)])
    @test length(allcells) == 4 * 4^2
end

@testset "OctaHEALPix tree: `getcell` corners match RingGrids `get_vertices`" begin
    usp(lon, lat) = GO.UnitSphereFromGeographic()((lon, lat))
    pdist(p, q) = maximum(abs.(Tuple(p) .- Tuple(q)))

    for nside in (4, 8, 16)
        tree = RingGridsExt.OctaHEALPixRootNode(nside)
        E, S, W, N = RingGrids.get_vertices(OctaHEALPixGrid, nside)  # reference (ring order)
        npts = 4 * nside^2

        # getcell(root, i) is ring-indexed; corners come out in (E,S,W,N) order
        maxerr = 0.0
        for ij in 1:npts
            poly = Trees.getcell(tree, ij)
            pts  = collect(GI.getpoint(GI.getexterior(poly)))
            @test length(pts) == 5            # 4 corners + closing point
            # getcell emits corners CCW: (E, N, W, S)
            ref = (usp(E[1,ij],E[2,ij]), usp(N[1,ij],N[2,ij]), usp(W[1,ij],W[2,ij]), usp(S[1,ij],S[2,ij]))
            for k in 1:4
                maxerr = max(maxerr, pdist(pts[k], ref[k]))
            end
        end
        @test maxerr < 1e-9
    end

    # full-grid iterator
    tree = RingGridsExt.OctaHEALPixRootNode(4)
    @test length(collect(Trees.getcell(tree))) == Trees.ncells(tree)
end

@testset "OctaHEALPix tree: node_extent + child_indices_extents" begin
    incap(p, cap) = GO.spherical_distance(cap.point, p) <= cap.radius + 1e-9

    for nside in (4, 5)                          # power-of-two and non-power-of-two
        tree = RingGridsExt.OctaHEALPixRootNode(nside)

        # A face cap must contain every corner of every cell beneath it (DFS pruning).
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

        # Leaf cells map bijectively onto the ring-order data indices 1:npts.
        ringidxs = Int[]
        for leaf in leaves(tree), (idx, ext) in STI.child_indices_extents(leaf)
            @test ext isa GO.UnitSpherical.SphericalCap
            push!(ringidxs, idx)
        end
        @test sort(ringidxs) == collect(1:4nside^2)
    end
end

@testset "OctaHEALPix: end-to-end regridding + conservation" begin
    # Regridder construction (nlat_half must be a power of two for the nested tree)
    src = rand(OctaHEALPixGrid, 8)
    dst = rand(OctaHEALPixGrid, 16)
    R = ConservativeRegridding.Regridder(dst, src)
    @test R isa ConservativeRegridding.Regridder
    @test size(R.intersections) == (length(dst), length(src))

    # Constant-field preservation: canary for partition gaps/overlaps and for
    # polygon orientation (wrong winding ⇒ negative areas ⇒ dropped ⇒ NaN).
    src_ones = ones(Float64, length(src))
    dst_out  = zeros(Float64, length(dst))
    ConservativeRegridding.regrid!(dst_out, R, src_ones)
    @test all(isapprox.(dst_out, 1.0; atol = 1e-10))

    # Area-weighted mean conservation for a non-vanishing analytic field.
    src_grid = OctaHEALPixGrid(16)
    dst_grid = OctaHEALPixGrid(8)
    R2 = ConservativeRegridding.Regridder(dst_grid, src_grid)
    londs, latds = RingGrids.get_londlatds(src_grid)
    src_vals = [2.0 + 0.5 * sin(2 * deg2rad(λ)) * cos(3 * deg2rad(φ)) for (λ, φ) in zip(londs, latds)]
    dst_vals = zeros(Float64, RingGrids.get_npoints(dst_grid))
    ConservativeRegridding.regrid!(dst_vals, R2, src_vals)
    @test isapprox(sum(src_vals .* R2.src_areas), sum(dst_vals .* R2.dst_areas); rtol = 1e-10)

    # Cross-grid: reduced OctaHEALPix → full Clenshaw, constant field preserved.
    src3 = rand(OctaHEALPixGrid, 8)
    dst3 = rand(FullClenshawGrid, 16)
    R3 = ConservativeRegridding.Regridder(dst3, src3)
    dst3_out = zeros(Float64, length(dst3))
    ConservativeRegridding.regrid!(dst3_out, R3, ones(Float64, length(src3)))
    @test all(isapprox.(dst3_out, 1.0; atol = 1e-10))

    # Non-power-of-two resolutions now partition and conserve (no Morton nest needed).
    srcn = OctaHEALPixGrid(6); dstn = OctaHEALPixGrid(10)
    Rn = ConservativeRegridding.Regridder(dstn, srcn)
    outn = zeros(Float64, RingGrids.get_npoints(dstn))
    ConservativeRegridding.regrid!(outn, Rn, ones(Float64, RingGrids.get_npoints(srcn)))
    @test all(isapprox.(outn, 1.0; atol = 1e-9))

    # Field-path (primary user API) and the reverse direction via `transpose`.
    dst_field = zeros(OctaHEALPixGrid, 16)
    ConservativeRegridding.regrid!(dst_field, R, ones(OctaHEALPixGrid, 8))
    @test all(x -> isapprox(x, 1; atol = 1e-6), dst_field)
    back = zeros(OctaHEALPixGrid, 8)
    ConservativeRegridding.regrid!(back, transpose(R), ones(OctaHEALPixGrid, 16))
    @test all(x -> isapprox(x, 1; atol = 1e-6), back)
end
