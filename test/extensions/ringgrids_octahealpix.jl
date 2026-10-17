using ConservativeRegridding
using Test

using RingGrids
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

@testset "OctaHEALPix provider: asgrid" begin
    field = rand(OctaHEALPixGrid, 4)          # nlat_half = nside = 4
    g = ConservativeRegridding.asgrid(field)

    @test g isa RingGridsExt.OctaHEALPixRingGrid
    @test ncells(g) == 4 * 4^2                 # 64 pixels
    @test nfaces(g) == 4                       # 4 base faces (octahedral)
    @test GOCore.best_manifold(g) == GO.Spherical()
    @test GOCore.best_manifold(field) == GO.Spherical()
    @test ConservativeRegridding.asgrid(field.grid) isa RingGridsExt.OctaHEALPixRingGrid

    tree = sweep_tree(g)
    @test STI.isspatialtree(typeof(tree))
    @test STI.node_extent(tree) isa GO.UnitSpherical.SphericalCap
end

@testset "OctaHEALPix provider: any nlat_half (no power-of-two requirement)" begin
    # The quadrant faces subdivide by index range, not a Morton nest, so every
    # nlat_half works — pow2 and not.
    for nside in (1, 2, 3, 4, 5, 7, 8, 16)
        g = ConservativeRegridding.asgrid(rand(OctaHEALPixGrid, nside))
        @test g isa RingGridsExt.OctaHEALPixRingGrid
        @test g.nside == nside
        @test ncells(g) == 4 * nside^2
        @test sort(GrI.subtree_indices(sweep_tree(g))) == 1:4nside^2
    end
end

@testset "OctaHEALPix provider: the tree partitions the pixels" begin
    g = RingGridsExt.OctaHEALPixRingGrid(4)   # nside = 4
    tree = sweep_tree(g)
    @test STI.isleaf(tree) == false

    # Leaves partition the faces; every pixel is emitted exactly once: 4·nside² total.
    allcells = reduce(vcat, [collect(STI.child_indices_extents(l)) for l in leaves(tree)])
    @test length(allcells) == 4 * 4^2
    @test sort(first.(allcells)) == 1:4 * 4^2
end

@testset "OctaHEALPix provider: cell corners match RingGrids `get_vertices`" begin
    usp(lon, lat) = GO.UnitSphereFromGeographic()((lon, lat))
    pdist(p, q) = maximum(abs.(Tuple(p) .- Tuple(q)))

    for nside in (4, 8, 16)
        g = RingGridsExt.OctaHEALPixRingGrid(nside)
        E, S, W, N = RingGrids.get_vertices(OctaHEALPixGrid, nside)  # reference (ring order)
        npts = 4 * nside^2

        # cell_geometry(g, i) is ring-indexed; corners come out CCW: (E, N, W, S)
        maxerr = 0.0
        for ij in 1:npts
            pts  = collect(GI.getpoint(GI.getexterior(cell_geometry(g, ij))))
            @test length(pts) == 5            # 4 corners + closing point
            ref = (usp(E[1,ij],E[2,ij]), usp(N[1,ij],N[2,ij]), usp(W[1,ij],W[2,ij]), usp(S[1,ij],S[2,ij]))
            for k in 1:4
                maxerr = max(maxerr, pdist(pts[k], ref[k]))
            end
        end
        @test maxerr < 1e-9
    end
end

@testset "OctaHEALPix provider: node extents contain the cells beneath" begin
    incap(p, cap) = GO.spherical_distance(cap.point, p) <= cap.radius + 1e-9

    for nside in (4, 5)                          # power-of-two and non-power-of-two
        g = RingGridsExt.OctaHEALPixRingGrid(nside)
        tree = sweep_tree(g)

        # A child's cap must contain every corner of every cell beneath it (DFS pruning).
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

        # Leaf cells map bijectively onto the ring-order data indices 1:npts.
        ringidxs = Int[]
        for leaf in leaves(tree), (idx, ext) in STI.child_indices_extents(leaf)
            @test ext isa GO.UnitSpherical.SphericalCap
            push!(ringidxs, idx)
        end
        @test sort(ringidxs) == collect(1:4nside^2)
    end
end

@testset "OctaHEALPix: cells tessellate the sphere" begin
    S = GO.Spherical()
    for nside in (2, 3, 4, 5)
        g = RingGridsExt.OctaHEALPixRingGrid(nside)
        @test isapprox(sum(cell_area(g, i) for i in 1:ncells(g)), 4π * S.radius^2; rtol = 1e-10)
    end
end

@testset "OctaHEALPix: end-to-end regridding + conservation" begin
    # Regridder construction
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
