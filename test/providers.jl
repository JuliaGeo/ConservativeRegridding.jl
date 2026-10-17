# The GridInterface providers of CR's extensions: GridInterfaceConformance on
# every native grid type through `grid_of` (what `Regridder` regrids) and the
# providers' active-cell views, the native inputs against their grids passed
# directly (and the U-pivot layout against its unique cells) on a few small
# pairs, and the issue #147 reproducer (a stretched-z U-pivot tripolar grid
# conserves a constant field).
using Test
using ConservativeRegridding
using ConservativeRegriddingTestHelpers
import GridInterface as GrI
using GridInterface: ncells, cellshape, tessellation, isglobal, parentpositions, Subset
using GridInterface.Structured
import GridInterfaceConformance as GIC
import GeometryOps as GO
import SparseArrays

using Oceananigans
using Oceananigans.Grids: RightFaceFolded, RightCenterFolded
using ClimaCore: CommonSpaces, Spaces, Meshes, Topologies, Domains, ClimaComms
import Healpix
using RingGrids

const OcExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)
const CCExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)
const HPExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)
const RGExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

# ---------------------------------------------------------------------------
# Conformance with known gaps recorded as broken (DiscreteGlobalGrids' pattern):
# a failing law whose name contains one of `known` is `@test_broken`, so a fix
# shows up as a broken test starting to pass. Every failure is printed.
# ---------------------------------------------------------------------------
_known(law, known) = any(k -> occursin(k, law), known)
function conformance_grid(grid; name, known=String[], nsamples=64, kw...)
    rep = GIC.check_grid(grid; name, nsamples, kw...)
    @testset "$name" begin
        for r in rep.results
            if r.status === :pass
                @test true
            elseif r.status === :fail
                gap = _known(r.law, known)
                println("  [", gap ? "known gap" : "FAIL", "] ", name, " / ", r.family, " / ", r.law,
                    isempty(r.detail) ? "" : ": " * r.detail)
                gap ? (@test_broken false) : (@test false)
            elseif r.status === :skip
                @test_skip "$(r.family): $(r.detail)"
            end
        end
    end
    return rep
end

function cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=false, radius=GO.Spherical().radius)
    context = ClimaComms.context()
    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(radius), h_elem)
    h_topology = use_sfc ? Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh)) :
        Topologies.Topology2D(context, h_mesh)
    return CommonSpaces.CubedSphereSpace(; radius=h_mesh.domain.radius, n_quad_points, h_elem, h_mesh, h_topology)
end

bathy(λ, φ) = (abs(φ) < 40 && 0 < mod(λ, 360) < 60) || φ > 75 ? 10.0 : -1000.0

const SUBSET_EXTENT_GAP = "GridAndTree forwards every verb in WRAPPED_VERBS"
const DIAMOND_DUAL_GAPS = ["vertex fans wind CCW about the vertex", "dual cells are valid CCW rings"]

# (name, native grid thunk, grid under test (`grid_of` the native, or a provider view of it),
#  expected type, expected faces, known gaps)
const CASES = [
    ("LatitudeLongitudeGrid 360x180", () -> LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1)),
        grid_of, OcExt.OceananigansLatLonGrid, 1, String[]),
    ("TripolarGrid F-pivot 120x60", () -> TripolarGrid(size=(120, 60, 1), fold_topology=RightFaceFolded),
        grid_of, OcExt.OceananigansCurvilinearGrid, 1, String[]),
    # the U-pivot provider's active-cell views (the layout `Regridder` regrids, `FoldedLayoutGrid`,
    # is checked in extensions/oceananigans.jl); a Subset's grid_extent is the members' union
    # while its GridAndTree answers the tree root (the parent's whole-sphere cap)
    ("TripolarGrid U-pivot 120x60 (unique cells)", () -> TripolarGrid(size=(120, 60, 1), fold_topology=RightCenterFolded),
        g -> OcExt.unique_cells(OcExt.structured_grid(g)), Subset, 1, [SUBSET_EXTENT_GAP]),
    ("TripolarGrid U-pivot, masked (wet cells)",
        () -> ImmersedBoundaryGrid(TripolarGrid(size=(40, 20, 2), z=(-1000, 0)), GridFittedBottom(bathy)),
        OcExt.wet_cells, Subset, 1, [SUBSET_EXTENT_GAP]),
    ("RotatedLatitudeLongitudeGrid 90x40", () -> RotatedLatitudeLongitudeGrid(size=(90, 40, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1), north_pole=(70, 55)),
        grid_of, OcExt.OceananigansCurvilinearGrid, 1, String[]),
    ("RectilinearGrid 100x100 (planar)", () -> RectilinearGrid(size=(100, 100), x=(0, 2), y=(0, 2), topology=(Periodic, Periodic, Flat)),
        grid_of, OcExt.OceananigansLatLonGrid, 1, String[]),
    ("ConformalCubedSphereGrid 16x16x6", () -> ConformalCubedSphereGrid(; panel_size=(16, 16, 1), z=(0, 1)),
        grid_of, OcExt.OceananigansCubedSphereGrid, 6, String[]),
    ("ClimaCore cubed sphere h16 regular", () -> cubedsphere_space(; h_elem=16), grid_of, CCExt.ClimaCoreCubedSphereGrid, 6, String[]),
    ("ClimaCore cubed sphere h16 Gilbert", () -> cubedsphere_space(; h_elem=16, use_sfc=true), grid_of, CCExt.ClimaCoreCubedSphereGrid, 6, String[]),
    ("Healpix nested 16", () -> Healpix.HealpixMap{Float64,Healpix.NestedOrder}(16), grid_of, HPExt.HealpixMapGrid{Healpix.NestedOrder}, 12, String[]),
    ("Healpix ring 16", () -> Healpix.HealpixMap{Float64,Healpix.RingOrder}(16), grid_of, HPExt.HealpixMapGrid{Healpix.RingOrder}, 12, String[]),
    ("FullClenshawGrid 16", () -> FullClenshawGrid(16), grid_of, RGExt.FullRingGrid, 1, String[]),
    ("FullGaussianGrid 12", () -> FullGaussianGrid(12), grid_of, RGExt.FullRingGrid, 1, String[]),
    ("HEALPixGrid 8", () -> HEALPixGrid(8), grid_of, RGExt.HEALPixRingGrid, 12, String[]),
    # Diamond cells: a cell's centroid sits equatorward of its E/W corners, so a
    # vertex lies outside its median-dual cell and the winding test about the
    # vertex reads the (CCW) dual ring as reversed; and the four equatorial face
    # corners of OctaHEALPix are hanging vertices of two cells (a 2-point dual ring).
    ("OctaHEALPixGrid 8", () -> OctaHEALPixGrid(8), grid_of, RGExt.OctaHEALPixRingGrid, 4, DIAMOND_DUAL_GAPS),
    ("OctahedralGaussianGrid 9", () -> OctahedralGaussianGrid(9), grid_of, RGExt.ReducedRingGrid, 0, DIAMOND_DUAL_GAPS),
]

@testset "providers: conformance" begin
    for (name, mk, view, T, nf, known) in CASES
        native = mk()
        g = view(native)
        @test g isa T
        base = g isa Subset ? parent(g) : g
        nf == 0 || @test nfaces(base) == nf
        nat = native isa Spaces.AbstractSpace ? Spaces.topology(native) :
            native isa Oceananigans.MultiRegion.MultiRegionGrid ? Oceananigans.MultiRegion.getregion(native, 1) : native
        @test GO.best_manifold(g) == GO.best_manifold(nat)
        println(name, ": ", nameof(typeof(base)), ", ", ncells(g), " cells")
        conformance_grid(g; name, known)
    end
end

@testset "providers: the U-pivot fold" begin
    g = OcExt.structured_grid(TripolarGrid(size=(40, 20, 1), fold_topology=RightCenterFolded))
    Nx, Ny = g.Nx, g.Ny
    @test tessellation(g) isa GrI.Overlapping && g.fold === :center
    # fold-row cell (i, Ny) is the quadrilateral (Nx+1-i, Ny) again
    for i in 1:Nx
        a = Set(collect(GrI.cell_corners(g, faceposition(g, 1, i, Ny)))); b = Set(collect(GrI.cell_corners(g, faceposition(g, 1, Nx + 1 - i, Ny))))
        @test all(p -> any(q -> GrI._near(p, q), b), a)
    end
    u = OcExt.unique_cells(g)
    @test ncells(u) == Nx * (Ny - 1) + Nx ÷ 2 && tessellation(u) isa GrI.Overlapping
    @test sum(i -> GrI.cell_area(u, i), 1:ncells(u)) ≈ sum(i -> GrI.cell_area(g, i), 1:Nx*(Ny-1)) +
        sum(i -> GrI.cell_area(g, i), (Ny-1)*Nx .+ (1:Nx)) / 2
    # neighbours across the fold are symmetric on the unique cells and reach the cell under the twin
    for k in 1:ncells(u)
        for q in GrI.neighbors(u, k)
            @test k in GrI.neighbors(u, q)
        end
    end
    i = 3; p = Nx + 1 - i
    nb = GrI.neighbors(g, faceposition(g, 1, i, Ny); connectivity=GrI.Edge())
    @test faceposition(g, 1, p, Ny - 1) in nb && length(nb) == 4
    # a duplicate answers as its primary
    @test collect(GrI.neighbors(g, faceposition(g, 1, p, Ny))) == collect(GrI.neighbors(g, faceposition(g, 1, i, Ny)))
    # the masked quadtree of the unique subset emits only members
    t = GrI.treeify(u)
    @test t isa Structured.MaskedCursor
    @test sort(GrI.subtree_indices(t)) == 1:ncells(u)
end

@testset "providers: the F-pivot fold" begin
    g = OcExt.structured_grid(TripolarGrid(size=(40, 20, 1), fold_topology=RightFaceFolded))
    Nx, Ny = g.Nx, g.Ny
    @test tessellation(g) isa GrI.Tessellating
    for i in 1:Nx
        x = facecross(g, 1, i, Ny, 3)
        @test x == (1, Nx + 1 - i, Ny, 3) && facecross(g, 1, x[2], x[3], x[4]) == (1, i, Ny, 3)
    end
    # the two pivots are closed two-cell fans
    @test length(facefan(g, 1, 1, Ny + 1)) == 2 && length(facefan(g, 1, Nx ÷ 2 + 1, Ny + 1)) == 2
end

@testset "providers: native inputs and their grids agree (small pairs)" begin
    R(d, s) = ConservativeRegridding.Regridder(d, s; threaded=false)
    ll = LatitudeLongitudeGrid(size=(60, 30, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    tf = TripolarGrid(size=(40, 20, 1), fold_topology=RightFaceFolded)
    tu = TripolarGrid(size=(40, 20, 1), fold_topology=RightCenterFolded)
    for (dst, src) in ((ll, tf), (tf, ll), (ll, tu), (tu, ll))
        test_weights_agree(R(dst, src), R(grid_of(dst), grid_of(src)))
    end
    # the U-pivot layout (ghosts of zero area) against the provider's unique cells
    U = OcExt.unique_cells(OcExt.structured_grid(tu))
    test_weights_agree(R(ll, tu), R(grid_of(ll), U); src=U)
    test_weights_agree(R(tu, ll), R(U, grid_of(ll)); dst=U)
    hp = HEALPixGrid(8); fc = FullClenshawGrid(16); oh = OctaHEALPixGrid(8); og = OctahedralGaussianGrid(8)
    for (dst, src) in ((fc, hp), (oh, fc), (og, hp), (fc, og))
        test_weights_agree(R(dst, src), R(grid_of(dst), grid_of(src)))
    end
    hm = Healpix.HealpixMap{Float64,Healpix.NestedOrder}(8)
    m = GO.Spherical()                                            # mixed radii: the manifold is explicit, as in CR's tests
    Rm(d, s) = ConservativeRegridding.Regridder(m, d, s; threaded=false)
    test_weights_agree(Rm(ll, hm), Rm(grid_of(ll), grid_of(hm)))
end

@testset "issue #147: stretched-z U-pivot tripolar grid conserves a constant field" begin
    src = TripolarGrid(size=(144, 72, 1); z=[0, 1])              # a vector z: not ZReg, which the old native tree required
    dst = LatitudeLongitudeGrid(size=(90, 42, 1), z=(0, 1), longitude=(0, 360), latitude=(-84, 84))
    S = grid_of(src); D = grid_of(dst)
    @test S isa OcExt.FoldedLayoutGrid                            # the layout, the fold row's duplicates as ghosts
    Rp = ConservativeRegridding.Regridder(D, S; threaded=false)
    out = zeros(ncells(D))
    ConservativeRegridding.regrid!(out, Rp, ones(ncells(S)))
    # the tripolar grid starts at -81.2 for this size: the first destination row (-84..-80) is only
    # partly covered, every other covered cell must see exactly 1
    covered = vec(sum(Rp.intersections; dims=2)) .> 0
    covered[1:90] .= false
    @test maximum(abs.(out[covered] .- 1)) < 1e-10
    @test maximum(out) < 1 + 1e-10                                 # the fold row is counted once
    real = Rp.src_areas .> 0                                     # the ghosts have no area to cover
    @test maximum(vec(sum(Rp.intersections; dims=1))[real] ./ Rp.src_areas[real]) < 1 + 1e-10   # no source cell used twice
    # the native input regrids the same layout: the fold row is counted once there too
    Rn = ConservativeRegridding.Regridder(dst, src; threaded=false)
    @test Rn.intersections == Rp.intersections
    outn = zeros(ncells(D)); ConservativeRegridding.regrid!(outn, Rn, ones(size(Rn.intersections, 2)))
    @test maximum(outn) < 1 + 1e-10
end
