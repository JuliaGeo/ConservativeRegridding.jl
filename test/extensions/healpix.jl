using ConservativeRegridding
using Test

import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI
import GridInterface as GrI
using GridInterface: ncells, cell_geometry, cell_area
using GridInterface.Structured: nfaces

import Healpix

const HealpixExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)

@testset "HealpixMapGrid provider" begin
    @testset "asgrid, sizes, manifold" begin
        map = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(4)
        g = ConservativeRegridding.asgrid(map)
        @test g isa HealpixExt.HealpixMapGrid{Healpix.NestedOrder}
        @test ncells(g) == 12 * 4^2  # 192 pixels
        @test nfaces(g) == 12
        @test GOCore.best_manifold(g) == GO.Spherical()
        @test GOCore.best_manifold(map) == GO.Spherical()

        ring = Healpix.HealpixMap{Float64, Healpix.RingOrder}(8)
        gr = ConservativeRegridding.asgrid(ring)
        @test gr isa HealpixExt.HealpixMapGrid{Healpix.RingOrder}
        @test ncells(gr) == 12 * 8^2
    end

    @testset "cell geometry: Healpix's own pixel corners" begin
        res = Healpix.Resolution(2)
        for O in (Healpix.NestedOrder, Healpix.RingOrder)
            g = HealpixExt.HealpixMapGrid{O}(res)
            maxerr = 0.0
            for k in 1:ncells(g)
                cell = cell_geometry(g, k)
                @test GI.geomtrait(cell) == GI.PolygonTrait()
                pts = collect(GI.getpoint(GI.getexterior(cell)))
                @test length(pts) == 5  # 4 corners + closing point
                ringpix = O === Healpix.NestedOrder ? Healpix.nest2ring(res, k) : k
                bd = Healpix.boundariesRing(res, ringpix, 1, Float64)
                for c in 1:4
                    maxerr = max(maxerr, minimum(maximum(abs.(Tuple(pts[c]) .- bd[b, :])) for b in 1:4))
                end
            end
            @test maxerr < 1e-12
        end
    end

    @testset "cells tessellate the sphere" begin
        S = GO.Spherical()
        for nside in (1, 2, 4)
            g = HealpixExt.HealpixMapGrid{Healpix.NestedOrder}(nside)
            total = sum(cell_area(g, k) for k in 1:ncells(g))
            @test isapprox(total, 4π * S.radius^2; rtol = 1e-12)
        end
    end

    @testset "the tree emits every pixel once" begin
        g = HealpixExt.HealpixMapGrid{Healpix.NestedOrder}(4)
        tree = GrI.treeify(g; leaf_capacity = GrI.ForSweeps())
        @test STI.isspatialtree(typeof(tree))
        @test STI.node_extent(tree) isa GO.UnitSpherical.SphericalCap
        @test sort(GrI.subtree_indices(tree)) == 1:ncells(g)
        idxs = Int[]
        STI.depth_first_search(_ -> true, tree) do i
            push!(idxs, i)
        end
        @test sort(idxs) == 1:ncells(g)
    end

    @testset "end-to-end: a constant field is preserved" begin
        src = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(2)
        dst = Healpix.HealpixMap{Float64, Healpix.RingOrder}(4)
        R = ConservativeRegridding.Regridder(dst, src; threaded = false)
        @test size(R) == (12 * 4^2, 12 * 2^2)
        out = zeros(12 * 4^2)
        ConservativeRegridding.regrid!(out, R, ones(12 * 2^2))
        @test all(isapprox.(out, 1.0; atol = 1e-9))
    end
end
