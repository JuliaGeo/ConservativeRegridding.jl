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
