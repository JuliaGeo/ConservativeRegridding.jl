using ConservativeRegridding
using ConservativeRegridding: Trees
using StaticArrays
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, Meshes, Quadratures, Topologies, Domains, ClimaComms
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
    latlon_vals = zeros(360 * 180)
    regridder = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, src_field)

    fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))
    @test isapprox(
        sum(latlon_vals .* fv_areas),
        sum(src_field);
        rtol=1e-2, atol=10.0,
    )
end

@testset "Principled SE → FV: constant field exact" begin
    space = make_cubedsphere_space(; h_elem=8, n_quad_points=4)

    src_field = Fields.ones(space)
    R = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)

    dst = zeros(360 * 180)
    ConservativeRegridding.regrid!(dst, R, src_field)

    # Principled: every covered destination cell is ~1.0 to ~machine eps
    # (each FV cell sums to A_dst,k by partition of unity, then divided by A_dst,k → 1).
    @test maximum(abs.(dst .- 1.0)) < 1e-10
end

@testset "SE → FV: Gilbert ordered cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=true)
    @assert Topologies.uses_spacefillingcurve(space.grid.topology)

    src_field = Fields.coordinate_field(space).lat
    latlon_vals = zeros(360 * 180)
    regridder = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)

    ConservativeRegridding.regrid!(latlon_vals, regridder, src_field)

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

    regridder = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)

    src_fv = ones(360 * 180)
    dst_field = Fields.zeros(space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_fv)

    dst_vec = ClimaCoreExt.se_field_to_vec(dst_field)
    @test all(x -> isapprox(x, 1.0; atol=1e-10), dst_vec)
end

@testset "FV → SE: Gilbert ordered cubed sphere" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4, use_sfc=true)
    @assert Topologies.uses_spacefillingcurve(space.grid.topology)

    regridder = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)

    src_fv = ones(360 * 180)
    dst_field = Fields.zeros(space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_fv)

    dst_vec = ClimaCoreExt.se_field_to_vec(dst_field)
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

    src_field = Fields.ones(space)
    src_vec = ClimaCoreExt.se_field_to_vec(src_field)
    R = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    dst = zeros(360 * 180)
    ConservativeRegridding.regrid!(dst, R, src_field)

    # Source-side integral the regridder treats as the conserved invariant:
    src_integral = sum(vec(sum(R.intersections; dims=1)) .* src_vec)
    dst_integral = sum(dst .* R.dst_areas)

    sphere_area = 4π * GO.Spherical().radius^2
    @test isapprox(src_integral, sphere_area; rtol=1e-12)   # math sanity
    @test isapprox(dst_integral, src_integral; rtol=1e-12)  # conservation
end

@testset "SE → FV → SE roundtrip conservation" begin
    space = make_cubedsphere_space(; h_elem=16, n_quad_points=4)

    src_field = Fields.coordinate_field(space).lat
    N_fv = 360 * 180

    fwd = ConservativeRegridding.Regridder(latlon_grid, space; threaded=false)
    bwd = ConservativeRegridding.Regridder(space, latlon_grid; threaded=false)

    fv_vals = zeros(N_fv)
    ConservativeRegridding.regrid!(fv_vals, fwd, src_field)

    roundtrip_field = Fields.zeros(space)
    ConservativeRegridding.regrid!(roundtrip_field, bwd, fv_vals)

    fv_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(latlon_grid))
    @test isapprox(sum(fv_vals .* fv_areas), sum(src_field); rtol=1e-2, atol=10.0)
end

@testset "Threaded vs serial assembly is identical (SE↔FV)" begin
    # Parallel and serial assembly must produce identical matrices: there are no
    # cross-pair duplicate (row,col) entries, so chunking can't change any sum. A
    # discrepancy signals a data race or ordering bug. Tested on regular + Gilbert.
    coarse_latlon = LatitudeLongitudeGrid(
        size=(72, 36, 1), longitude=(0, 360), latitude=(-90, 90),
        z=(0, 1), radius=GO.Spherical().radius,
    )
    for use_sfc in (false, true)
        space = make_cubedsphere_space(; h_elem=8, n_quad_points=4, use_sfc=use_sfc)

        se_to_fv_threaded = ConservativeRegridding.Regridder(coarse_latlon, space; threaded=true)
        se_to_fv_serial   = ConservativeRegridding.Regridder(coarse_latlon, space; threaded=false)
        @test se_to_fv_threaded.intersections == se_to_fv_serial.intersections

        fv_to_se_threaded = ConservativeRegridding.Regridder(space, coarse_latlon; threaded=true)
        fv_to_se_serial   = ConservativeRegridding.Regridder(space, coarse_latlon; threaded=false)
        @test fv_to_se_threaded.intersections == fv_to_se_serial.intersections
    end
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

    regridder = ConservativeRegridding.Regridder(space, tripolar_grid)

    dst_field = Fields.zeros(space)
    ConservativeRegridding.regrid!(dst_field, regridder, vec(interior(src_tripolar)))

    dst_vec = ClimaCoreExt.se_field_to_vec(dst_field)
    @test isapprox(mean(dst_vec), 1.0, atol=0.1)
end

@testset "Oceananigans TripolarGrid to ClimaCore cubed sphere (FaceFolded)" begin
    tripolar_grid = TripolarGrid(size=(360, 180, 1), fold_topology = FaceFolded)
    space = CommonSpaces.CubedSphereSpace(;
        radius = tripolar_grid.radius,
        n_quad_points = 4,
        h_elem = 32,
    )

    src_tripolar = Field{Center, Center, Nothing}(tripolar_grid)
    set!(src_tripolar, src_tripolar + 1)

    regridder = ConservativeRegridding.Regridder(space, tripolar_grid)

    dst_field = Fields.zeros(space)
    ConservativeRegridding.regrid!(dst_field, regridder, vec(interior(src_tripolar)))

    dst_vec = ClimaCoreExt.se_field_to_vec(dst_field)
    @test isapprox(mean(dst_vec), 1.0, atol=0.1)
end

@testset "SE → FV: constants exact with a non-default radius" begin
    R = 6.371e6
    @assert R != GO.Spherical().radius
    manifold = GO.Spherical(; radius = R)
    space = make_cubedsphere_space(; h_elem = 8, n_quad_points = 4, radius = R)
    grid = LatitudeLongitudeGrid(
        size = (90, 45, 1), longitude = (0, 360), latitude = (-90, 90),
        z = (0, 1), radius = R,
    )
    regridder = ConservativeRegridding.Regridder(manifold, grid, space; threaded = false)
    dst = zeros(90 * 45)
    ConservativeRegridding.regrid!(dst, regridder, Fields.ones(space))
    @test maximum(abs.(dst .- 1)) < 1e-12
end

@testset "B over a whole element matches the mass-matrix row sums" begin
    # For the element polygon itself, B[i, j] = ∫_e ϕᵢ ϕⱼ dA, which is also the row sum of the
    # local mass matrix (computed with the interpolated Jacobian).
    space = make_cubedsphere_space(; h_elem = 16, n_quad_points = 4)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    ij_linear_ind = LinearIndices((Base.OneTo(Nq), Base.OneTo(Nq)))
    manifold = GO.Spherical()
    se_tree = Trees.treeify(manifold, space)
    # A cube-corner element (most distorted) and an element at the center of a face.
    for elem_idx in (1, 8 + 7 * 16)
        poly = Trees.getcell(se_tree, elem_idx)
        B = ClimaCoreExt.accumulate_principled_b(manifold, space, elem_idx, poly;
                                                 triangle_quad_degree = 8)
        row_sums = vec(sum(ClimaCoreExt.compute_local_mass_matrix(space, elem_idx); dims = 2))
        rel_err = maximum(abs(B[i, j] / row_sums[ij_linear_ind[i, j]] - 1)
                          for j in 1:Nq, i in 1:Nq)
        # What remains is the error of the interpolated Jacobian in the row sums
        @test rel_err < 1e-5
    end
end

@testset "FV → SE: partially covered elements are conservative and bounded" begin
    # A source that stops at ±60° cuts through elements. As in the FV → FV and SE → FV
    # transfers, the uncovered part counts as zero.
    space = make_cubedsphere_space(; h_elem = 16, n_quad_points = 4)
    band = LatitudeLongitudeGrid(
        size = (360, 120, 1), longitude = (0, 360), latitude = (-60, 60),
        z = (0, 1), radius = GO.Spherical().radius,
    )
    regridder = ConservativeRegridding.Regridder(space, band; threaded = false)
    band_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(band))
    lat = ClimaCoreExt.se_field_to_vec(Fields.coordinate_field(space).lat)
    lon = ClimaCoreExt.se_field_to_vec(Fields.coordinate_field(space).long)

    # Constant source
    dst = Fields.zeros(space)
    ConservativeRegridding.regrid!(dst, regridder, ones(360 * 120))
    vals = ClimaCoreExt.se_field_to_vec(dst)
    @test isapprox(sum(dst), sum(band_areas); rtol = 1e-6)
    @test all(v -> -0.25 < v < 1.25, vals)
    @test all(v -> isapprox(v, 1; atol = 1e-12), vals[abs.(lat) .< 50])
    @test all(iszero, vals[abs.(lat) .> 70])

    # Non-constant source in [0, 1]
    band_lon = repeat(range(0.5, 359.5; length = 360), 1, 120)
    src = vec(@. (1 + sind(band_lon)) / 2)
    ConservativeRegridding.regrid!(dst, regridder, src)
    vals = ClimaCoreExt.se_field_to_vec(dst)
    @test isapprox(sum(dst), sum(src .* band_areas); rtol = 1e-6)
    @test all(v -> -0.25 < v < 1.25, vals)
    @test maximum(abs.(vals[abs.(lat) .< 50] .- (1 .+ sind.(lon[abs.(lat) .< 50])) ./ 2)) < 2e-3
end
