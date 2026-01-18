

using ConservativeRegridding
using ConservativeRegridding.Trees

import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI, UnitSpherical
using StaticArrays: SA
using LinearAlgebra: normalize

import Healpix

const HealpixExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)

@testset "HealpixTree basic functionality" begin
    @testset "Core types and constructors" begin
        tree = HealpixExt.HealpixTree(4)
        @test tree isa HealpixExt.HealpixRootNode{Healpix.NestedOrder}
        @test tree.nside_max == 4
        @test Trees.ncells(tree) == 12 * 4^2  # 192 pixels

        tree_ring = HealpixExt.HealpixTree(Healpix.RingOrder, 8)
        @test tree_ring isa HealpixExt.HealpixRootNode{Healpix.RingOrder}
        @test tree_ring.nside_max == 8
    end

    @testset "SpatialTreeInterface - HealpixRootNode" begin
        tree = HealpixExt.HealpixTree(2)
        @test STI.isspatialtree(typeof(tree)) == true
        @test STI.isleaf(tree) == false
        @test STI.nchild(tree) == 12

        # Check children are HealpixTreeNodes at level 0
        child = STI.getchild(tree, 1)
        @test child isa HealpixExt.HealpixTreeNode{Healpix.NestedOrder}
        @test child.level == 0
        @test child.pixel == 0

        child12 = STI.getchild(tree, 12)
        @test child12.pixel == 11

        # Check extent is full sphere
        extent = STI.node_extent(tree)
        @test extent isa GO.UnitSpherical.SphericalCap
    end

    @testset "SpatialTreeInterface - HealpixTreeNode" begin
        tree = HealpixExt.HealpixTree(4)  # leaf level is log2(4) = 2

        # Level 0 node (base face)
        node0 = STI.getchild(tree, 1)
        @test STI.isleaf(node0) == false
        @test STI.nchild(node0) == 4

        # Level 1 node
        node1 = STI.getchild(node0, 1)
        @test node1.level == 1
        @test node1.pixel == 0
        @test STI.isleaf(node1) == false

        # Level 2 node (leaf for nside=4)
        node2 = STI.getchild(node1, 1)
        @test node2.level == 2
        @test STI.isleaf(node2) == true
        @test STI.nchild(node2) == 0
    end

    @testset "getcell" begin
        tree = HealpixExt.HealpixTree(2)

        # Single cell
        cell = Trees.getcell(tree, 1)
        @test GI.geomtrait(cell) == GI.PolygonTrait()
        ring = GI.getexterior(cell)
        @test GI.npoint(ring) == 5  # 4 corners + closing point

        # Iterator
        cells = collect(Trees.getcell(tree))
        @test length(cells) == Trees.ncells(tree)
    end

    @testset "child_indices_extents" begin
        tree = HealpixExt.HealpixTree(1)  # nside=1 means leaves at level 0
        leaf = STI.getchild(tree, 1)
        @test STI.isleaf(leaf) == true

        idx_ext = STI.child_indices_extents(leaf)
        @test length(idx_ext) == 1
        idx, ext = idx_ext[1]
        @test idx == 1  # 1-based index
        @test ext isa GO.UnitSpherical.SphericalCap
    end

    @testset "Manifold" begin
        tree = HealpixExt.HealpixTree(4)
        @test GOCore.best_manifold(tree) == GO.Spherical()

        node = STI.getchild(tree, 1)
        @test GOCore.best_manifold(node) == GO.Spherical()
    end

    @testset "treeify passthrough" begin
        tree = HealpixExt.HealpixTree(4)
        @test Trees.treeify(GO.Spherical(), tree) === tree
    end
end