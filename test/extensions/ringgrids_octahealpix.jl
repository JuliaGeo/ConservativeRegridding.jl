using ConservativeRegridding
using ConservativeRegridding.Trees
using Test

using RingGrids
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI

const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

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

@testset "OctaHEALPix tree: nlat_half must be a power of two" begin
    # The nested hierarchy (child = 4·parent) only exists for power-of-two nside;
    # RingGrids itself asserts this for ring↔nested reordering. Error early and clearly.
    @test_throws "power of two" RingGridsExt.OctaHEALPixRootNode(12)
    @test_throws "power of two" RingGridsExt.OctaHEALPixRootNode(6)
    @test_throws "power of two" Trees.treeify(GO.Spherical(), rand(OctaHEALPixGrid, 6))
    # powers of two are accepted
    for n in (1, 2, 4, 8, 16, 32)
        @test_nowarn RingGridsExt.OctaHEALPixRootNode(n)
    end
end

@testset "OctaHEALPix tree: parent/child traversal" begin
    tree = RingGridsExt.OctaHEALPixRootNode(4)        # leaf level = log2(4) = 2

    # Level-0 base faces: 4 children, pixels 0..3 (0-based nested)
    @test STI.nchild(tree) == 4
    node0 = STI.getchild(tree, 1)
    @test node0 isa RingGridsExt.OctaHEALPixTreeNode
    @test node0.level == 0
    @test node0.pixel == 0
    @test node0.leaf_level == 2                 # cached, propagated from root
    @test STI.getchild(tree, 4).pixel == 3
    @test STI.isleaf(node0) == false
    @test STI.nchild(node0) == 4

    # child = 4·parent + offset (nested ordering)
    node1 = STI.getchild(node0, 1)
    @test node1.level == 1
    @test node1.pixel == 0
    @test STI.getchild(node0, 3).pixel == 2     # 4*0 + (3-1)
    node0b = STI.getchild(tree, 2)              # base face pixel 1
    @test STI.getchild(node0b, 1).pixel == 4    # 4*1 + 0

    # Level-2 node is a leaf for nside=4
    node2 = STI.getchild(node1, 1)
    @test node2.level == 2
    @test STI.isleaf(node2) == true
    @test STI.nchild(node2) == 0
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
    # all leaf nodes under a node
    leaves(node) = STI.isleaf(node) ? [node] :
        reduce(vcat, (leaves(STI.getchild(node, i)) for i in 1:STI.nchild(node)))

    incap(p, cap) = GO.spherical_distance(cap.point, p) <= cap.radius + 1e-9

    tree = RingGridsExt.OctaHEALPixRootNode(4)

    # A node's cap must contain every corner of every leaf beneath it (DFS-pruning correctness)
    for i in 1:STI.nchild(tree)
        base = STI.getchild(tree, i)
        cap  = STI.node_extent(base)
        @test cap isa GO.UnitSpherical.SphericalCap
        for leaf in leaves(base)
            ij = STI.child_indices_extents(leaf)[1][1]
            for p in GI.getpoint(GI.getexterior(Trees.getcell(tree, ij)))
                @test incap(p, cap)
            end
        end
    end

    # Leaf maps its nested pixel to the correct ring index, and the leaves
    # bijection onto the ring-order data indices 1:npts.
    grid = rand(OctaHEALPixGrid, 4).grid
    npts = 4 * 4^2
    ringidxs = Int[]
    for leaf in leaves(tree)
        idx, ext = STI.child_indices_extents(leaf)[1]
        @test ext isa GO.UnitSpherical.SphericalCap
        @test idx == RingGrids.nest2ring(leaf.pixel + 1, grid)   # 0-based pixel → 1-based nested
        push!(ringidxs, idx)
    end
    @test sort(ringidxs) == collect(1:npts)
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

    # Field-path (primary user API) and the reverse direction via `transpose`.
    dst_field = zeros(OctaHEALPixGrid, 16)
    ConservativeRegridding.regrid!(dst_field, R, ones(OctaHEALPixGrid, 8))
    @test all(x -> isapprox(x, 1; atol = 1e-6), dst_field)
    back = zeros(OctaHEALPixGrid, 8)
    ConservativeRegridding.regrid!(back, transpose(R), ones(OctaHEALPixGrid, 16))
    @test all(x -> isapprox(x, 1; atol = 1e-6), back)
end
