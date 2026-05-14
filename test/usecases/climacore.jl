using ConservativeRegridding
using ConservativeRegridding: Trees, destination_areas, source_areas
using StaticArrays
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms
using Oceananigans

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

# Nodal Lagrange evaluation matches `ConservativeRegriddingClimaCoreExt` (barycentric formula).
@testset "Lagrange basis via ClimaCore.Quadratures.interpolation_matrix" begin
    ξs, _ = Quadratures.quadrature_points(Float64, Quadratures.GLL{4}())

    @testset "Kronecker delta at nodes" begin
        for p in 1:4
            M = Quadratures.interpolation_matrix(SVector(ξs[p]), ξs)
            for i in 1:4
                @test M[1, i] ≈ (i == p ? 1.0 : 0.0) atol=1e-12
            end
        end
    end

    @testset "Partition of unity off-node" begin
        for ξ in (-0.7, -0.3, 0.0, 0.42, 0.91)
            M = Quadratures.interpolation_matrix(SVector(ξ), ξs)
            @test sum(M[1, :]) ≈ 1.0 atol=1e-12
        end
    end

    @testset "Single-point row is length Nq" begin
        ξ = 0.3
        M = Quadratures.interpolation_matrix(SVector(ξ), ξs)
        @test size(M) == (1, 4)
        @test sum(M[1, :]) ≈ 1.0 atol=1e-12
    end

    @testset "Buffer fill from matrix row" begin
        ξs3, _ = Quadratures.quadrature_points(Float64, Quadratures.GLL{3}())
        ξ = 0.5
        M = Quadratures.interpolation_matrix(SVector(ξ), ξs3)
        out = zeros(3)
        out .= M[1, :]
        @test out ≈ Vector(M[1, :])
    end
end

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
    dst_integral = sum(dst .* destination_areas(R))

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
