########################################################
# Conservative regridding between a ClimaCore spectral-element cubed sphere and
# a regular latitude–longitude (finite-volume) grid.
#
# The principled path assembles the regridding matrix per spectral-element node:
# each entry is ∫_intersection ϕᵢ ϕⱼ dA, with source areas given by the per-node
# Jacobian weights. Fields are therefore regridded directly, with no per-element
# averaging. SE → SE is not supported — go SE → FV → SE through two regridders.
#
# Run from the test environment (it carries ClimaCore, Oceananigans and Test):
#   julia --project=test examples/lowlevel/climacore_exp.jl
########################################################

using ClimaCore: CommonSpaces, Fields
using ConservativeRegridding
using ConservativeRegridding: Trees
using Oceananigans
using Test
import GeometryOps as GO

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

radius = GO.Spherical().radius

space = CommonSpaces.CubedSphereSpace(; radius, n_quad_points = 4, h_elem = 8)

latlon_grid = LatitudeLongitudeGrid(;
    size = (360, 180, 1), longitude = (0, 360), latitude = (-90, 90),
    z = (0, 1), radius,
)
N_fv = 360 * 180
fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))

### SE → FV
src_field = Fields.coordinate_field(space).lat
fwd = ConservativeRegridding.Regridder(latlon_grid, space; threaded = false)

fv_vals = zeros(N_fv)
ConservativeRegridding.regrid!(fv_vals, fwd, src_field)

@test isapprox(sum(fv_vals .* fv_areas), sum(src_field); rtol = 1e-2, atol = 10.0)

### Constant fields are reproduced to machine precision (partition of unity)
const_fv = zeros(N_fv)
ConservativeRegridding.regrid!(const_fv, fwd, Fields.ones(space))
@test maximum(abs.(const_fv .- 1.0)) < 1e-10

### FV → SE, completing an SE → FV → SE roundtrip
bwd = ConservativeRegridding.Regridder(space, latlon_grid; threaded = false)

roundtrip_field = Fields.zeros(space)
ConservativeRegridding.regrid!(roundtrip_field, bwd, fv_vals)

@test isapprox(sum(fv_vals .* fv_areas), sum(src_field); rtol = 1e-2, atol = 10.0)

### Flat nodal access round-trips through se_field_to_vec / vec_to_se_field!
v = ClimaCoreExt.se_field_to_vec(roundtrip_field)
back = Fields.zeros(space)
ClimaCoreExt.vec_to_se_field!(back, v)
@test ClimaCoreExt.se_field_to_vec(back) == v

# # Visual comparison with ClimaCoreMakie / CairoMakie (docs environment):
# using CairoMakie, ClimaCoreMakie
# save("src.png", ClimaCoreMakie.fieldheatmap(src_field))
# save("roundtrip.png", ClimaCoreMakie.fieldheatmap(roundtrip_field))
