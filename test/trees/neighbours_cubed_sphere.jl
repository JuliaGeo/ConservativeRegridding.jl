using ConservativeRegridding
using ConservativeRegridding: Trees
using ConservativeRegridding.Trees: neighbours, has_optimized_neighbour_search
using Test
using Random
import GeometryOps as GO

using ClimaCore:
    CommonSpaces, Fields, Spaces, Meshes, Topologies, Domains, ClimaComms

# Helper: build a `Topology2D` and the corresponding `CubedSphereToplevelTree`.
function build_cubed_sphere(ne::Integer)
    context = ClimaComms.context()
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{Float64}(GO.Spherical().radius),
        ne,
    )
    h_topology = Topologies.Topology2D(context, h_mesh)
    space = CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points = 2,
        h_elem = ne,
        h_mesh,
        h_topology,
    )
    tree = Trees.treeify(GO.Spherical(), space)
    return tree, h_topology
end

# Helper: decode a global cubed-sphere linear index to (face, i, j).
function decode_cube(idx::Integer, ne::Integer)
    ne2 = ne * ne
    face = ((idx - 1) ÷ ne2) + 1
    face_local = ((idx - 1) % ne2) + 1
    i = mod1(face_local, ne)
    j = ((face_local - 1) ÷ ne) + 1
    return (face, i, j)
end

@inline _cube_lin(F, i, j, ne) = i + (j - 1) * ne + (F - 1) * ne^2

@testset "Connectivity table pinned values" begin
    tree, _ = build_cubed_sphere(2)
    table = tree.connectivity.table
    @test tree.connectivity.ne == 2
    # Edge IDs: 1=S, 2=E, 3=N, 4=W
    # Face 1 row from the design doc.
    @test table[1, 1] == (Int8(6), Int8(3), true)
    @test table[2, 1] == (Int8(2), Int8(4), true)
    @test table[3, 1] == (Int8(3), Int8(4), true)
    @test table[4, 1] == (Int8(5), Int8(3), true)
    # Face 5 row from the design doc.
    @test table[1, 5] == (Int8(4), Int8(3), true)
    @test table[2, 5] == (Int8(6), Int8(4), true)
    @test table[3, 5] == (Int8(1), Int8(4), true)
    @test table[4, 5] == (Int8(3), Int8(3), true)

    @test has_optimized_neighbour_search(tree) == true
end

@testset "Interior-of-face cells (ne=4) — 8 neighbours from arithmetic" begin
    ne = 4
    tree, _ = build_cubed_sphere(ne)
    # Pick (F=1, i=2, j=2): strict interior on face 1.
    F, i, j = 1, 2, 2
    idx = _cube_lin(F, i, j, ne)
    nbrs = neighbours(tree, idx)
    @test length(nbrs) == 8
    expected = Set{Int}([
        _cube_lin(F, i,   j-1, ne), _cube_lin(F, i+1, j-1, ne),
        _cube_lin(F, i+1, j,   ne), _cube_lin(F, i+1, j+1, ne),
        _cube_lin(F, i,   j+1, ne), _cube_lin(F, i-1, j+1, ne),
        _cube_lin(F, i-1, j,   ne), _cube_lin(F, i-1, j-1, ne),
    ])
    @test Set(nbrs) == expected
end

@testset "Edge-of-face non-corner cells — 8 nbrs, 3 cross-face" begin
    ne = 4
    tree, topology = build_cubed_sphere(ne)
    # Pick the south edge of face 1 at i=2 (non-corner): (F=1, i=2, j=1).
    F, i, j = 1, 2, 1
    idx = _cube_lin(F, i, j, ne)
    nbrs = neighbours(tree, idx)
    @test length(nbrs) == 8

    # Three of the eight neighbours decode to a face other than F.
    cross = filter(n -> decode_cube(n, ne)[1] != F, nbrs)
    @test length(cross) == 3
    # And those three should each lie on the face we expect from the table.
    other_face_expected = tree.connectivity.table[1, F][1]  # south edge → table[1, F]
    @test all(decode_cube(n, ne)[1] == other_face_expected for n in cross)
end

@testset "Corner cells return exactly 7 neighbours (all faces)" begin
    ne = 4
    tree, _ = build_cubed_sphere(ne)
    for F in 1:6
        for (i, j) in ((1, 1), (1, ne), (ne, 1), (ne, ne))
            idx = _cube_lin(F, i, j, ne)
            nbrs = neighbours(tree, idx)
            @test length(nbrs) == 7
        end
    end
end

@testset "Symmetry on a sample of cells" begin
    ne = 4
    tree, _ = build_cubed_sphere(ne)
    rng = Random.MersenneTwister(0)
    ncells = 6 * ne^2
    sample = rand(rng, 1:ncells, 200)
    for a in sample
        for b in neighbours(tree, a)
            @test a in neighbours(tree, b)
        end
    end
end

@testset "Scale test: ne=64 cubed sphere" begin
    ne = 64
    tree, _ = build_cubed_sphere(ne)
    ncells = 6 * ne^2
    rng = Random.MersenneTwister(123)
    sample = rand(rng, 1:ncells, 1000)
    for a in sample
        nbrs = neighbours(tree, a)
        @test length(nbrs) in (7, 8)
    end
    # Symmetry on the sample.
    for a in sample
        for b in neighbours(tree, a)
            @test a in neighbours(tree, b)
        end
    end
end
