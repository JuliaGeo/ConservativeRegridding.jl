using ConservativeRegridding
using ConservativeRegridding.Trees

import SphericalSpatialTrees as SST
import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI, UnitSpherical
using StaticArrays: SA

using Test

using Oceananigans # for source data

using GeoMakie, GLMakie

GOCore.best_manifold(tree::SST.ISEACircleTree) = GO.Spherical()
function Trees.treeify(manifold::GO.Spherical, tree::SST.ISEACircleTree)
    return tree
end

function Trees.ncells(tree::SST.ISEACircleTree)
    return prod(SST.gridsize(tree))
end

function Trees.getcell(tree::SST.ISEACircleTree, i::Int)
    return GI.Polygon(SA[GI.LinearRing(SST.index_to_polygon_unitsphere(i, tree))])
end

function Trees.getcell(tree::SST.ISEACircleTree)
    return (getcell(tree, i) for i in 1:ncells(tree))
end

target_grid = SST.ISEACircleTree(6 #= resolution =#)
target_data = zeros(SST.gridsize(target_grid))

source_grid = Oceananigans.LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1), radius = GO.Spherical().radius)
source_field = Oceananigans.CenterField(source_grid)
Oceananigans.set!(source_field, LongitudeField())

regridder = ConservativeRegridding.Regridder(target_grid, source_field; normalize = false)

ConservativeRegridding.regrid!(vec(target_data), regridder, vec(interior(source_field)))

# using GeoMakie, GLMakie, Geodesy
f, a, p = poly(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(target_grid))) |> vec; color = vec(target_data), axis = (; type = GlobeAxis))
lines!(a, GeoMakie.coastlines(); zlevel = 100_000, color = :orange)
f

# f, a, p = poly(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(source_grid))) |> vec; color = vec(interior(source_field)), axis = (; type = GlobeAxis))


poly(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(SST.ISEACircleTree(2))); color = :white, strokewidth = 1, axis = (; type = GlobeAxis))

computed_areas = sum(regridder.intersections, dims=2)[:, 1]
direct_areas = ConservativeRegridding.areas(GO.Spherical(), target_grid)

mismatch_idxs = findall(!, computed_areas .≈ direct_areas)

computed_areas[2774]
direct_areas[2774]

@testset "Regridder object conserved areas" begin
    @test sum(regridder.intersections, dims=2)[:, 1] ≈ ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(target_grid))
    @test sum(regridder.intersections, dims=1)[1, :] ≈ ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(source_grid))
end

@testset "Integral was conserved" begin
    target_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(target_grid))
    source_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(source_grid))
    @test sum(vec(target_data) .* target_areas) ≈ sum(vec(interior(source_field)) .* vec(source_areas))
end