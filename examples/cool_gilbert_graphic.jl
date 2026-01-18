using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms
using Oceananigans

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

latlon_grid = LatitudeLongitudeGrid(size=(36, 18, 1), longitude=(0, 360), latitude=(-90, 90), z = (0, 1), radius = GO.Spherical().radius)
latlon_field = CenterField(latlon_grid)

cubedsphere_space = CommonSpaces.CubedSphereSpace(;
    radius = latlon_grid.radius,
    n_quad_points = 2,
    h_elem = 15,
)

# Define a field on the first space, to use as our source field
field = Fields.coordinate_field(cubedsphere_space).long
ones_field = Fields.ones(cubedsphere_space)
cubed_sphere_vals = zeros(6*cubedsphere_space.grid.topology.mesh.ne^2)
ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)
cubed_sphere_vals


device = ClimaComms.device()
context = ClimaComms.context(device)
h_elem = 15
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

regridder = @time ConservativeRegridding.Regridder(cubedsphere_space, latlon_grid; threaded = true)
set!(latlon_field, ConservativeRegridding.VortexField(; lat0_rad = deg2rad(80)))
set!(latlon_field, ConservativeRegridding.GulfStreamField())
set!(latlon_field, ConservativeRegridding.LongitudeField())
ConservativeRegridding.regrid!(cubed_sphere_vals, regridder, vec(interior(latlon_field)))

poly(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(cubedsphere_gilbert_ordered_space))) |> vec; color = vec(cubed_sphere_vals), axis = (; type = GlobeAxis))