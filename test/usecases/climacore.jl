using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms
using Oceananigans

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

latlon_grid = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z = (0, 1), radius = GO.Spherical().radius)

@testset "Regular cubed sphere" begin
    cubedsphere_space = CommonSpaces.CubedSphereSpace(;
        radius = latlon_grid.radius,
        n_quad_points = 2,
        h_elem = 64,
    )

    # Define a field on the first space, to use as our source field
    field = Fields.coordinate_field(cubedsphere_space).long
    ones_field = Fields.ones(cubedsphere_space)
    cubed_sphere_vals = zeros(6*cubedsphere_space.grid.topology.mesh.ne^2)
    ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)
    cubed_sphere_vals

    latlon_field = Oceananigans.CenterField(latlon_grid)
    latlon_vals = vec(interior(latlon_field))
    # heatmap(interior(latlon_field, :, :, 1))
    set!(latlon_field, (x, y, z) -> 0)

    regridder = ConservativeRegridding.Regridder(latlon_grid, cubedsphere_space; threaded = false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, cubed_sphere_vals)

    # Test that the integral is conserved
    @test isapprox(
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_space))),
        rtol = 1e-13
    )

    # Test the other way
    set!(latlon_field, (x, y, z) -> x)

    ConservativeRegridding.regrid!(cubed_sphere_vals, transpose(regridder), latlon_vals)

    @test isapprox(
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_space))),
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        rtol = 1e-13
    )
end
# With a space filling curve

@testset "Gilbert ordered cubed sphere" begin
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    h_elem = 64
    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(GO.Spherical().radius), h_elem)
    h_topology = Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))
    cubedsphere_gilbert_ordered_space = CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points = 2,
        h_elem = h_elem,
        h_mesh = h_mesh,
        h_topology = h_topology,
    )


    # Define a field on the first space, to use as our source field
    field = Fields.coordinate_field(cubedsphere_gilbert_ordered_space).long
    ones_field = Fields.ones(cubedsphere_gilbert_ordered_space)
    cubed_sphere_vals = zeros(6*cubedsphere_gilbert_ordered_space.grid.topology.mesh.ne^2)
    ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)
    cubed_sphere_vals

    latlon_field = Oceananigans.CenterField(latlon_grid)
    latlon_vals = vec(interior(latlon_field))
    # heatmap(interior(latlon_field, :, :, 1))
    set!(latlon_field, (x, y, z) -> 0)

    regridder = ConservativeRegridding.Regridder(latlon_grid, cubedsphere_gilbert_ordered_space; threaded = false)
    # regridder = ConservativeRegridding.Regridder(cubedsphere_gilbert_ordered_space, latlon_grid; threaded = false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, cubed_sphere_vals)

    # Test that the integral is conserved
    @test isapprox(
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_gilbert_ordered_space))),
        rtol = 1e-13
    )

    # Test the other way
    set!(latlon_field, (x, y, z) -> x)

    ConservativeRegridding.regrid!(cubed_sphere_vals, transpose(regridder), latlon_vals)

    @test isapprox(
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_gilbert_ordered_space))),
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        rtol = 1e-13
    )
end

# Get metrics

# ## Metrics


@testset "Test helper functions in extension" begin
    @testset "integrate_each_element" begin
        @test isapprox(
            sum(ClimaCoreExt.integrate_each_element(field)), 
            sum(field), 
            atol = 1e-11
        )
        @test sum(ClimaCoreExt.integrate_each_element(ones_field)) ≈ sum(ones_field)
    end
    @testset "Convert to one value per element" begin
        value_per_element1 = zeros(Float64, Meshes.nelements(cubedsphere_space.grid.topology.mesh))
        field1_one_value_per_element = Fields.zeros(cubedsphere_space)
        ClimaCoreExt.set_value_per_element!(field1_one_value_per_element, value_per_element1)

        # Check the error of converting to one value per element
        abs_error_one_value_per_element = abs(sum(field1_one_value_per_element) - sum(field))
        @test_broken abs_error_one_value_per_element < 2e-11
        @test isapprox(mean(field), mean(field1_one_value_per_element), atol=1e-14)
    end
end
#=
using GLMakie, GeoMakie

cubedsphere_polys = collect(Trees.getcell(Trees.treeify(cubedsphere_space))) |>  x-> GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), x) .|> GI.convert(LibGEOS) |> vec
f, a, p = poly(cubedsphere_polys; color = vec(cubed_sphere_vals), axis = (; type = GlobeAxis))
=#