using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
using Statistics

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    Nx, Ny, _ = size(field.grid)
    ℓx, ℓy    = Center(), Center()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    grid = field.grid
    arch = grid.architecture
    FT = eltype(grid)

    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, Nx+1, Ny+1)

    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, (Nx+1, Ny+1), _compute_cell_matrix!, cell_matrix, Nx, ℓx, ℓy, grid)

    return cell_matrix
end

flip(::Face) = Center()
flip(::Center) = Face()

@kernel function _compute_cell_matrix!(cell_matrix, Nx, ℓx, ℓy, grid)
    i, j = @index(Global, NTuple)

    vx = flip(ℓx)
    vy = flip(ℓy)

    xl = ξnode(i, j, 1, grid, vx, vy, nothing)
    yl = ηnode(i, j, 1, grid, vx, vy, nothing)

    @inbounds cell_matrix[i, j] = (xl, yl)
end
coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

dst = CenterField(coarse_grid)
src = CenterField(fine_grid)


dst = src

dst_cells = compute_cell_matrix(dst)
src_cells = compute_cell_matrix(src)

set!(src, (x, y, z) -> rand())

dst_qt = Trees.CellBasedQuadtree(dst_cells) |> Trees.QuadtreeCursor |> Trees.KnownFullSphereExtentWrapper
src_qt = Trees.CellBasedQuadtree(src_cells) |> Trees.QuadtreeCursor |> Trees.KnownFullSphereExtentWrapper

import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: @controlflow
 


function instrumented_dual_dfs(f::F, predicate::P, node1::T1, node2::T2, log = Any[]) where {F, P, T1, T2}
    if STI.isleaf(node1) && STI.isleaf(node2)
        # @info "Both leaf"
        push!(log, (; n1 = node1, n2 = node2, type = :leaf12, predicate_result = false))
        # both nodes are leaves, so we can just iterate over the indices and extents
        for (i1, extent1) in STI.child_indices_extents(node1)
            for (i2, extent2) in STI.child_indices_extents(node2)
                if predicate(extent1, extent2)
                    @controlflow f(i1, i2)
                end
            end
        end
    elseif STI.isleaf(node1) # node2 is not a leaf, node1 is - recurse further into node2
        for child in STI.getchild(node2)
            if predicate(STI.node_extent(node1), STI.node_extent(child))
                push!(log, (; n1 = node1, n2 = child, type = :leaf1, predicate_result = true))
                @controlflow instrumented_dual_dfs(f, predicate, node1, child, log)
            end
        end
    elseif STI.isleaf(node2) # node1 is not a leaf, node2 is - recurse further into node1
        for child in STI.getchild(node1)
            if predicate(STI.node_extent(child), STI.node_extent(node2))
                push!(log, (; n1 = child, n2 = node2, type = :leaf2, predicate_result = true))
                @controlflow instrumented_dual_dfs(f, predicate, child, node2, log)
            end
        end
    else # neither node is a leaf, recurse into both children
        @info "Both not leaf"
        for child1 in STI.getchild(node1)
            for child2 in STI.getchild(node2)
                if predicate(STI.node_extent(child1), STI.node_extent(child2))
                    push!(log, (; n1 = child1, n2 = child2, type = :inner, predicate_result = true))
                    @controlflow instrumented_dual_dfs(f, predicate, child1, child2, log)
                end
            end
        end
    end
    return log
end

idxs = Tuple{CartesianIndex{2}, CartesianIndex{2}}[]
log = instrumented_dual_dfs(GO.UnitSpherical._intersects, src_qt, dst_qt) do i1, i2
    push!(idxs, (i1, i2))
end

fidxs = filter(x -> x[1] == x[2], idxs) 

idxs1 == 1

import SphericalSpatialTrees as SST

y = range(-90.0, 90.0, length=100)
x = range(-180.0, 180.0, length=100)

tree = Trees.RegularGridQuadtree(x, y)
tree_cursor = Trees.QuadtreeCursor(tree) |> Trees.KnownFullSphereExtentWrapper
tree_cursor = Trees.TopDownQuadtreeCursor(tree) |> Trees.KnownFullSphereExtentWrapper

idxs = Tuple{NTuple{2, Int}, NTuple{2, Int}}[]
log = instrumented_dual_dfs(GO.UnitSpherical._intersects, tree_cursor, tree_cursor) do i1, i2
    push!(idxs, (i1, i2))
end


STI.getchild(tree_cursor, 1)

fidxs2 = filter(x -> x[1] == x[2], idxs) 

bools = fill(false, size(getcell(tree)))
for (i1, i2) in fidxs2
    bools[i1] = true
end

using GLMakie

heatmap(bools)

function napply(f, arg, n)
    res = arg
    for i in 1:n
        res = f(res)
    end
    return res
end

leaf1 = napply(x -> STI.getchild(x, 1), tree_cursor, 6)

c1 = STI.node_extent(leaf1)
GO.UnitSpherical._intersects(c1, c1)

chitr = STI.child_indices_extents(leaf1)

c1 = STI.getchild(leaf1, 1)
p1 = Trees.getcell(c1, 1, 1)
up1 = GO.transform(GO.UnitSphereFromGeographic(), p1)

ue1 = GI.getexterior(up1)
GO.UnitSpherical.slerp(GI.getpoint(ue1, 1), GI.getpoint(ue1, 2), 0.5)





c2 = STI.getchild(tree_cursor, 2)
STI.node_extent(c2)

GO.UnitSpherical._intersects(STI.node_extent(c2), STI.node_extent(c2))

using GeoMakie
f, a, p = meshimage(-180..180, -90..90, reshape([colorant"white"], 1, 1); axis = (; type = GlobeAxis), zlevel = -300_000)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
poly!(a, STI.node_extent(c2); color = :red, alpha = 0.5, zlevel = 100_000, transparency = true)

idxs = NTuple{2, Tuple{Int, Int}}[]
log = instrumented_dual_dfs(GO.UnitSpherical._intersects, c2, c2) do i1, i2
    push!(idxs, (i1, i2))
end
idxs

c2c = collect(STI.getchild(c2))


c2 = STI.getchild(tree_cursor, 4) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1) |> x -> STI.getchild(x, 1)
chillins = STI.getchild(c2) |> collect



f, a, p = meshimage(-180..180, -90..90, reshape([colorant"white"], 1, 1); axis = (; type = GlobeAxis), zlevel = -300_000)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
# poly!(a, STI.node_extent(c2); color = :red, alpha = 0.5, zlevel = 100_000, transparency = true)
# poly!(a, vec(collect(Trees.getcell(c2))); strokecolor = :red, strokewidth = 1, alpha = 0.5, zlevel = 100_000, transparency = true, color = :transparent)
poly!(a, STI.node_extent(chillins[1]); color = Cycled(1), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[2]); color = Cycled(2), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[3]); color = Cycled(3), alpha = 0.5, zlevel = 100_000, transparency = true)
poly!(a, STI.node_extent(chillins[4]); color = Cycled(4), alpha = 0.5, zlevel = 100_000, transparency = true)

lines!(a, vec(collect(Trees.getcell(chillins[1]))); color = Cycled(1), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[2]))); color = Cycled(2), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[3]))); color = Cycled(3), alpha = 0.5, zlevel = 100_000, transparency = true)
lines!(a, vec(collect(Trees.getcell(chillins[4]))); color = Cycled(4), alpha = 0.5, zlevel = 100_000, transparency = true)

c2 = STI.getchild(c2, 1)

