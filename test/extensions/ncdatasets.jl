using ConservativeRegridding
using Test
using SparseArrays
using NCDatasets
import GeometryOps as GO, GeoInterface as GI

# Two tiny planar polygon grids that perfectly tile [0, 2] × [0, 2].
# Same setup as test/usecases/simple.jl so frac_a ≈ frac_b ≈ 1.0.
const DST_POLYS = begin
    pts = [(i, j) for i in 0:2, j in 0:2]
    [GI.Polygon([GI.LinearRing([pts[i, j], pts[i, j+1], pts[i+1, j+1], pts[i+1, j], pts[i, j]])])
     for i in 1:size(pts, 1)-1, j in 1:size(pts, 2)-1] |> vec
end

const SRC_POLYS = begin
    diamond = GI.Polygon([GI.LinearRing([(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)])])
    triangles = GI.Polygon.([
        [GI.LinearRing([(0, 0), (1, 0), (0, 1), (0, 0)])],
        [GI.LinearRing([(0, 1), (0, 2), (1, 2), (0, 1)])],
        [GI.LinearRing([(1, 2), (2, 1), (2, 2), (1, 2)])],
        [GI.LinearRing([(2, 1), (2, 0), (1, 0), (2, 1)])],
    ])
    [diamond, triangles...]
end

function build_regridder()
    dst = GO.SpatialTreeInterface.FlatNoTree(DST_POLYS)
    src = GO.SpatialTreeInterface.FlatNoTree(SRC_POLYS)
    return ConservativeRegridding.Regridder(GO.Planar(), dst, src; normalize=false)
end

@testset "ESMF round-trip schema" begin
    r = build_regridder()
    A = r.intersections
    path = tempname() * ".nc"
    @test ConservativeRegridding.save_esmf_weights(path, r) == path
    @test isfile(path)

    NCDataset(path, "r") do ds
        @test haskey(ds.dim, "n_s") && haskey(ds.dim, "n_a") && haskey(ds.dim, "n_b")
        @test ds.dim["n_a"] == length(r.src_areas)
        @test ds.dim["n_b"] == length(r.dst_areas)
        @test ds.dim["n_s"] == nnz(A)

        for v in ("S", "row", "col", "frac_a", "frac_b", "area_a", "area_b")
            @test haskey(ds, v)
        end

        @test size(ds["S"])      == (nnz(A),)
        @test size(ds["row"])    == (nnz(A),)
        @test size(ds["col"])    == (nnz(A),)
        @test size(ds["frac_a"]) == (length(r.src_areas),)
        @test size(ds["frac_b"]) == (length(r.dst_areas),)
        @test size(ds["area_a"]) == (length(r.src_areas),)
        @test size(ds["area_b"]) == (length(r.dst_areas),)

        @test ds.attrib["normalization"] == "destarea"
        @test ds.attrib["source_grid"]   == "source"
        @test ds.attrib["destination_grid"] == "destination"
        @test !haskey(ds.attrib, "created_at")
        @test !haskey(ds.attrib, "source_grid_shape")
        @test !haskey(ds.attrib, "destination_grid_shape")
    end
    rm(path; force=true)
end

@testset "ESMF round-trip values" begin
    r = build_regridder()
    A = r.intersections
    src_areas = Float64.(r.src_areas)
    dst_areas = Float64.(r.dst_areas)
    path = tempname() * ".nc"
    ConservativeRegridding.save_esmf_weights(path, r)

    NCDataset(path, "r") do ds
        S      = Array(ds["S"][:])
        row    = Array(ds["row"][:])
        col    = Array(ds["col"][:])
        frac_a = Array(ds["frac_a"][:])
        frac_b = Array(ds["frac_b"][:])
        area_a = Array(ds["area_a"][:])
        area_b = Array(ds["area_b"][:])

        @test area_a ≈ src_areas
        @test area_b ≈ dst_areas

        # Rebuild the intersection matrix from (S, row, col) and check it matches.
        rebuilt = sparse(row, col, S .* dst_areas[row], length(dst_areas), length(src_areas))
        @test Array(rebuilt) ≈ Array(A)

        # Marginals from the file match marginals computed fresh from A.
        @test frac_a ≈ vec(sum(A; dims=1)) ./ src_areas
        @test frac_b ≈ vec(sum(A; dims=2)) ./ dst_areas

        # Both grids tile [0,2]×[0,2] exactly → full coverage.
        @test all(isapprox.(frac_a, 1.0; atol=1e-12))
        @test all(isapprox.(frac_b, 1.0; atol=1e-12))
    end
    rm(path; force=true)
end

@testset "ESMF optional attributes" begin
    r = build_regridder()
    path = tempname() * ".nc"
    ConservativeRegridding.save_esmf_weights(
        path, r;
        src_grid_name = "era5_0.25deg",
        dst_grid_name = "c90",
        src_shape = (720, 361),
        dst_shape = (90, 90, 6),
        created_at = "2026-04-21T00:00:00",
    )

    NCDataset(path, "r") do ds
        @test ds.attrib["source_grid"]      == "era5_0.25deg"
        @test ds.attrib["destination_grid"] == "c90"
        @test ds.attrib["created_at"]       == "2026-04-21T00:00:00"
        @test collect(ds.attrib["source_grid_shape"])      == [720, 361]
        @test collect(ds.attrib["destination_grid_shape"]) == [90, 90, 6]
    end
    rm(path; force=true)
end
