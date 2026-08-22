using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS
import GeometryOps: SpatialTreeInterface as STI

using Oceananigans

const OceananigansExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)

@testset "Padded tree keeps its offset global index domain" begin
    transform = GO.UnitSphereFromGeographic()
    points = [transform((lon, lat)) for lon in (0.0, 10.0, 20.0), lat in (0.0, 10.0)]
    grid = Trees.CellBasedGrid(GO.Spherical(), points)
    real = Trees.IndexOffsetQuadtreeCursor(grid, 10)
    padding_polygon = Trees.getcell(grid, 1, 1)
    padded = OceananigansExt.PaddedTreeWrapper(real, 2, padding_polygon, 10)

    # The tree emits only the real IDs 11:12, but its addressable output domain
    # also contains padded IDs 13:14, which remain zero in sparse assembly.
    @test vec(first.(collect(STI.child_indices_extents(padded)))) == [11, 12]
    @test Trees.ncells(padded) == (4, 1)
    @test Trees.cell_index_count(padded) == 14
    @test Trees.getcell(padded, 13) === padding_polygon
    @test Trees.getcell(padded, 14) === padding_polygon
end
