using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
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
# coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
# fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

# dst = CenterField(coarse_grid)
# src = CenterField(fine_grid)

src_grid = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightCenterFolded)

src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)

src_cells = GO.UnitSphereFromGeographic().(compute_cell_matrix(src_field))
dst_cells = GO.UnitSphereFromGeographic().(compute_cell_matrix(dst_field))

set!(src_field, (x, y, z) -> rand())

src_qt = Trees.CellBasedQuadtree(src_cells) 
dst_qt = Trees.CellBasedQuadtree(dst_cells) 

src_tree = src_qt |> Trees.TopDownQuadtreeCursor |> Trees.KnownFullSphereExtentWrapper
dst_tree = dst_qt |> Trees.TopDownQuadtreeCursor |> Trees.KnownFullSphereExtentWrapper

idxs = NTuple{2, NTuple{2, Int}}[]
@time STI.dual_depth_first_search(GO.UnitSpherical._intersects, src_tree, dst_tree) do i1, i2
    push!(idxs, (i1, i2))
end
idxs

using SparseArrays, ProgressMeter

mat = spzeros(Float64, prod(size(src_cells).-1), prod(size(dst_cells).-1))

linearizer1 = LinearIndices(size(src_cells).-1)
linearizer2 = LinearIndices(size(dst_cells).-1)

@showprogress for (i1, i2) in idxs
    p1 = Trees.getcell(src_qt, i1...)
    p2 = Trees.getcell(dst_qt, i2...)
    polygon_of_intersection = try; GO.intersection(GO.Spherical(), p1, p2; target = GO.PolygonTrait()) catch e; @show "Error during intersection" i1 i2 e; rethrow(e); end
    area_of_intersection = GO.area(GO.Spherical(), polygon_of_intersection)
    mat[linearizer1[i1...], linearizer2[i2...]] += area_of_intersection
end