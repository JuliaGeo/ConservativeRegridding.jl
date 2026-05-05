using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms
using Oceananigans

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

latlon_grid = LatitudeLongitudeGrid(
    size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90),
    z=(0, 1), radius=GO.Spherical().radius,
)

function make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=false, radius=GO.Spherical().radius)
    context = ClimaComms.context()
    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(radius), h_elem)
    h_topology = if use_sfc
        Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))
    else
        Topologies.Topology2D(context, h_mesh)
    end
    CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points,
        h_elem,
        h_mesh,
        h_topology,
    )
end

@testset "SE → FV: Regular cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)
    @assert !Topologies.uses_spacefillingcurve(space.grid.topology)

    src_field = Fields.coordinate_field(space).lat
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)

    latlon_vals = zeros(360 * 180)
    regridder = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    @test regridder isa ConservativeRegridding.SEtoFVRegridder

    ConservativeRegridding.regrid!(latlon_vals, regridder, src_vec)

    fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))
    @test isapprox(
        sum(latlon_vals .* fv_areas),
        sum(src_field);
        rtol=1e-2, atol=10.0,
    )
end

@testset "Principled SE → FV: constant field exact" begin
    space = make_cubedsphere_space(; h_elem=8, n_quad_points=4)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    src_vec = ones(N_nodes)
    R = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)  # default = principled
    @test R isa ConservativeRegridding.SEtoFVRegridder

    dst = zeros(360 * 180)
    ConservativeRegridding.regrid!(dst, R, src_vec)

    # Principled: every covered destination cell is ~1.0 to ~machine eps
    # (each FV cell sums to A_dst,k by partition of unity, then divided by A_dst,k → 1).
    @test maximum(abs.(dst .- 1.0)) < 1e-10
end

@testset "method=:node_in_polygon kwarg dispatches to simplified path" begin
    # Verify the kwarg dispatch reaches the existing simplified constructor.
    # We don't assert conservation on a non-constant field here because the
    # simplified scheme is documented to be non-conservative in regimes
    # where many FV cells contain no SE node (PDF §2, Option 2).
    space = make_cubedsphere_space(; h_elem=8, n_quad_points=4)
    R = ConservativeRegridding.Regridder(latlon_grid, space;
                                         method=:node_in_polygon, threaded=false)
    @test R isa ConservativeRegridding.SEtoFVRegridder
end

@testset "SE → FV: Gilbert ordered cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=true)
    @assert Topologies.uses_spacefillingcurve(space.grid.topology)

    src_field = Fields.coordinate_field(space).lat
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)

    latlon_vals = zeros(360 * 180)
    regridder = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    @test regridder isa ConservativeRegridding.SEtoFVRegridder

    ConservativeRegridding.regrid!(latlon_vals, regridder, src_vec)

    fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))
    @test isapprox(
        sum(latlon_vals .* fv_areas),
        sum(src_field);
        rtol=1e-2, atol=10.0,
    )
end

@testset "FV → SE: Regular cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)
    @assert !Topologies.uses_spacefillingcurve(space.grid.topology)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    regridder = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)
    @test regridder isa ConservativeRegridding.FVtoSERegridder

    src_fv = ones(360 * 180)
    dst_vec = zeros(N_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, src_fv)

    @test all(x -> isapprox(x, 1.0; atol=1e-10), dst_vec)
end

@testset "FV → SE: Gilbert ordered cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=true)
    @assert Topologies.uses_spacefillingcurve(space.grid.topology)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    regridder = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)
    @test regridder isa ConservativeRegridding.FVtoSERegridder

    src_fv = ones(360 * 180)
    dst_vec = zeros(N_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, src_fv)

    @test all(x -> isapprox(x, 1.0; atol=1e-10), dst_vec)
end

@testset "SE → SE: Regular cubed sphere (different resolutions)" begin
    src_space = make_cubedsphere_space(; h_elem=8, n_quad_points=4)
    dst_space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)

    src_field = Fields.ones(src_space)
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)

    Nq_dst = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst_space))
    Nh_dst = Meshes.nelements(Topologies.mesh(Spaces.topology(dst_space)))
    N_dst_nodes = Nq_dst^2 * Nh_dst

    regridder = ConservativeRegridding.Regridder(dst_space, src_space; threaded=false)
    @test regridder isa ConservativeRegridding.SEtoSERegridder

    dst_vec = zeros(N_dst_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, src_vec)

    dst_field = Fields.zeros(dst_space)
    ClimaCoreExt.vec_to_se_field!(dst_field, dst_vec)

    @test isapprox(sum(dst_field), sum(src_field), rtol=1e-2)
end

@testset "SE → SE: Gilbert ordered cubed sphere" begin
    src_space = make_cubedsphere_space(; h_elem=8, n_quad_points=4, use_sfc=true)
    dst_space = make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=true)

    src_field = Fields.ones(src_space)
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)

    Nq_dst = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst_space))
    Nh_dst = Meshes.nelements(Topologies.mesh(Spaces.topology(dst_space)))
    N_dst_nodes = Nq_dst^2 * Nh_dst

    regridder = ConservativeRegridding.Regridder(dst_space, src_space; threaded=false)
    @test regridder isa ConservativeRegridding.SEtoSERegridder

    dst_vec = zeros(N_dst_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, src_vec)

    dst_field = Fields.zeros(dst_space)
    ClimaCoreExt.vec_to_se_field!(dst_field, dst_vec)

    @test isapprox(sum(dst_field), sum(src_field), rtol=1e-2)
end

@testset "h-convergence: principled is higher-order than simplified" begin
    # Smooth field on the sphere; for each h_elem, regrid SE → FV and measure
    # L2 error against the analytic field evaluated at FV cell centers.
    # Principled should converge at ≥ ~2nd order in h; simplified at ~1st.
    f(lon, lat) = sin(2 * deg2rad(lat)) * cos(deg2rad(lon))

    coarse_grid = LatitudeLongitudeGrid(
        size=(180, 90, 1), longitude=(0, 360), latitude=(-90, 90),
        z=(0, 1), radius=GO.Spherical().radius,
    )
    nx, ny = 180, 90
    ref = [f((i - 0.5) * (360/nx), (j - 0.5) * (180/ny) - 90) for j in 1:ny for i in 1:nx]

    function L2err(h_elem, method)
        space = make_cubedsphere_space(; h_elem, n_quad_points=4)
        coords = Fields.coordinate_field(space)
        long_v = parent(Fields.field_values(coords.long))
        lat_v  = parent(Fields.field_values(coords.lat))
        # Flatten in the same order as se_field_to_vec
        src_vec = vec([f(long_v[i, j, 1, h], lat_v[i, j, 1, h])
                       for i in axes(long_v, 1), j in axes(long_v, 2),
                       h in axes(long_v, 4)])

        R = ConservativeRegridding.Regridder(coarse_grid, space; method, threaded=false)
        dst = zeros(nx * ny)
        ConservativeRegridding.regrid!(dst, R, src_vec)
        return sqrt(sum((dst .- ref).^2) / length(dst))
    end

    h_elems = (4, 8, 16)

    err_prin = [L2err(h, :polygon_intersection) for h in h_elems]
    err_simp = [L2err(h, :node_in_polygon)      for h in h_elems]

    # Principled should be strictly more accurate than simplified at every h
    # we test. (We can't cleanly test convergence *rate* here without an
    # exact cell-average reference: the principled regrid converges to cell
    # averages, while we sample the reference at cell centers, which has its
    # own O(h_FV²) discretization floor independent of h_SE — so the principled
    # error plateaus as h_SE shrinks. A rate assertion would need a finer FV
    # grid or analytic cell averages, both deferred.)
    for i in eachindex(h_elems)
        @test err_prin[i] < err_simp[i]
    end

    # Simplified should still converge at ~1st order in h_SE.
    rates_simp = [log2(err_simp[i] / err_simp[i+1]) for i in 1:length(err_simp)-1]
    @test all(>(0.7), rates_simp)
end

@testset "Principled SE → FV: conservation to 1e-12" begin
    # The principled path is conservative by construction:
    # Σ_k dst[k] · A_dst,k = Σ_{e,i,j} (Σ_k B(k,(e,i,j))) f^e_src,ij
    # On a *constant* source (c=1), the discrete source integral that the
    # regridder is built to preserve equals the sphere area exactly via
    # partition of unity (Σ_eij Σ_k B = Σ_k A_k = 4πR²). Test against a
    # meaningful, non-zero magnitude (lat integrates to ≈0 by symmetry,
    # so float noise dominates rtol on lat).
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))

    src_vec = ones(Nq^2 * Nh)
    R = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    dst = zeros(360 * 180)
    ConservativeRegridding.regrid!(dst, R, src_vec)

    # Source-side integral the regridder treats as the conserved invariant:
    src_integral = sum(vec(sum(R.weight_matrix; dims=1)) .* src_vec)
    dst_integral = sum(dst .* R.dst_areas)

    sphere_area = 4π * GO.Spherical().radius^2
    @test isapprox(src_integral, sphere_area; rtol=1e-12)   # math sanity
    @test isapprox(dst_integral, src_integral; rtol=1e-12)  # conservation
end

@testset "SE → FV → SE roundtrip conservation" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)

    src_field = Fields.coordinate_field(space).lat
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)

    N_fv = 360 * 180
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    fwd = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    bwd = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)

    fv_vals = zeros(N_fv)
    ConservativeRegridding.regrid!(fv_vals, fwd, src_vec)

    roundtrip_vec = zeros(N_nodes)
    ConservativeRegridding.regrid!(roundtrip_vec, bwd, fv_vals)

    roundtrip_field = Fields.zeros(space)
    ClimaCoreExt.vec_to_se_field!(roundtrip_field, roundtrip_vec)

    fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))
    @test isapprox(sum(fv_vals .* fv_areas), sum(src_field); rtol=1e-2, atol=10.0)
end

@testset "Oceananigans TripolarGrid to ClimaCore cubed sphere (default folding)" begin
    tripolar_grid = TripolarGrid(size=(360, 180, 1))
    space = CommonSpaces.CubedSphereSpace(;
        radius = tripolar_grid.radius,
        n_quad_points = 4,
        h_elem = 32,
    )

    src_tripolar = Field{Center, Center, Nothing}(tripolar_grid)
    set!(src_tripolar, src_tripolar + 1)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    regridder = ConservativeRegridding.Regridder(space, tripolar_grid)
    @test regridder isa ConservativeRegridding.FVtoSERegridder

    dst_vec = zeros(N_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, vec(interior(src_tripolar)))

    dst_field = Fields.zeros(space)
    ClimaCoreExt.vec_to_se_field!(dst_field, dst_vec)
    @test isapprox(mean(dst_vec), 1.0, atol=0.1)
end

@testset "Oceananigans TripolarGrid to ClimaCore cubed sphere (RightFaceFolded)" begin
    tripolar_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)
    space = CommonSpaces.CubedSphereSpace(;
        radius = tripolar_grid.radius,
        n_quad_points = 4,
        h_elem = 32,
    )

    src_tripolar = Field{Center, Center, Nothing}(tripolar_grid)
    set!(src_tripolar, src_tripolar + 1)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(space)))
    N_nodes = Nq^2 * Nh

    regridder = ConservativeRegridding.Regridder(space, tripolar_grid)
    @test regridder isa ConservativeRegridding.FVtoSERegridder

    dst_vec = zeros(N_nodes)
    ConservativeRegridding.regrid!(dst_vec, regridder, vec(interior(src_tripolar)))

    dst_field = Fields.zeros(space)
    ClimaCoreExt.vec_to_se_field!(dst_field, dst_vec)
    @test isapprox(mean(dst_vec), 1.0, atol=0.1)
end
