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

@testset "Regular cubed sphere (without spacefillingcurve)" begin
    context = ClimaComms.context()
    h_elem = 16
    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(GO.Spherical().radius), h_elem)
    h_topology = Topologies.Topology2D(context, h_mesh)
    cubedsphere_space = CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points = 2,
        h_elem,
        h_mesh,
        h_topology,
    )

    @assert !Topologies.uses_spacefillingcurve(cubedsphere_space.grid.topology)

    # Define a field on the first space, to use as our source field
    field = Fields.coordinate_field(cubedsphere_space).long

    ones_field = Fields.ones(cubedsphere_space)
    cubed_sphere_vals = zeros(Meshes.nelements(cubedsphere_space.grid.topology.mesh))
    ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)

    latlon_field = Oceananigans.CenterField(latlon_grid)
    latlon_vals = vec(interior(latlon_field))
    set!(latlon_field, (x, y, z) -> 0)

    regridder = ConservativeRegridding.Regridder(latlon_grid, cubedsphere_space; threaded = false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, cubed_sphere_vals)

    # Test that the integral is conserved
    @test isapprox(
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_space))),
        rtol = 1e-10
    )

    # Test the other way
    ConservativeRegridding.regrid!(cubed_sphere_vals, transpose(regridder), latlon_vals)

    @test isapprox(
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_space))),
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        rtol = 1e-13
    )

    field_remapped = Fields.zeros(cubedsphere_space)
    ClimaCoreExt.set_value_per_element!(field_remapped, cubed_sphere_vals)

    # Check that the integral over the space is conserved
    # TODO broken: -2 vs -4
    @test_broken isapprox(
        sum(field),
        sum(field_remapped),
        rtol = 1e-6,
    )
end

@testset "Gilbert ordered cubed sphere (with spacefillingcurve)" begin
    context = ClimaComms.context()
    h_elem = 16
    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{Float64}(GO.Spherical().radius), h_elem)
    h_topology = Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))
    cubedsphere_gilbert_ordered_space = CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points = 2,
        h_elem = h_elem,
        h_mesh = h_mesh,
        h_topology = h_topology,
    )

    @assert Topologies.uses_spacefillingcurve(cubedsphere_gilbert_ordered_space.grid.topology)

    # Define a field on the first space, to use as our source field
    field = Fields.coordinate_field(cubedsphere_gilbert_ordered_space).long

    ones_field = Fields.ones(cubedsphere_gilbert_ordered_space)
    cubed_sphere_vals = zeros(Meshes.nelements(cubedsphere_gilbert_ordered_space.grid.topology.mesh))
    ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)

    latlon_field = Oceananigans.CenterField(latlon_grid)
    latlon_vals = vec(interior(latlon_field))
    set!(latlon_field, (x, y, z) -> 0)

    regridder = ConservativeRegridding.Regridder(latlon_grid, cubedsphere_gilbert_ordered_space; threaded = false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, cubed_sphere_vals)

    # Test that the integral is conserved
    @test isapprox(
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_gilbert_ordered_space))),
        rtol = 1e-13
    )

    # Test the other way
    ConservativeRegridding.regrid!(cubed_sphere_vals, transpose(regridder), latlon_vals)

    @test isapprox(
        sum(abs, cubed_sphere_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(cubedsphere_gilbert_ordered_space))),
        sum(abs, latlon_vals .* ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))),
        rtol = 1e-13
    )

    field_remapped = Fields.zeros(cubedsphere_gilbert_ordered_space)
    ClimaCoreExt.set_value_per_element!(field_remapped, cubed_sphere_vals)

    # Check that the integral over the space is conserved
    # TODO broken: -6 vs -3
    @test_broken isapprox(
        sum(field),
        sum(field_remapped),
        rtol = 1e-8,
    )
end

# Get metrics

# ## Metrics

#=
using GLMakie, GeoMakie

cubedsphere_polys = collect(Trees.getcell(Trees.treeify(cubedsphere_space))) |>  x-> GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), x) .|> GI.convert(LibGEOS) |> vec
f, a, p = poly(cubedsphere_polys; color = vec(cubed_sphere_vals), axis = (; type = GlobeAxis))
=#
