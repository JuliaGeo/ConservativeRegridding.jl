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
# dst_grid = src_grid

src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)

src_longlat_cells = compute_cell_matrix(src_field)
dst_longlat_cells = compute_cell_matrix(dst_field)

src_cells = GO.UnitSphereFromGeographic().(src_longlat_cells)
dst_cells = GO.UnitSphereFromGeographic().(dst_longlat_cells)

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

unique(first.(idxs))

# idxs = [Tuple{Int, Int}[] for _ in CartesianIndices(size(src_cells).-1)]
# @time STI.dual_depth_first_search(GO.UnitSpherical._intersects, src_tree, dst_tree) do i1, i2
#     push!(idxs[i1...], (i2))
# end
# idxs

# Sort `idxs` by the first element of the tuple, i.e. the source cell index
# This allows us to then partition it by source cell index so that no computations
# "overlap"...
sorted_idxs = sort(idxs, by = first)
npartitions = Threads.nthreads()

using SparseArrays, ProgressMeter

mat = spzeros(Float64, prod(size(src_cells).-1), prod(size(dst_cells).-1))

linearizer1 = LinearIndices(size(src_cells).-1)
linearizer2 = LinearIndices(size(dst_cells).-1)

failing_pairs = NTuple{2, NTuple{2, Int}}[]

@time @showprogress for (i1, i2) in idxs
    p1 = Trees.getcell(src_qt, i1...)
    p2 = Trees.getcell(dst_qt, i2...)
    polygon_of_intersection = #=try; =#
        GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2; target = GO.PolygonTrait()) 
    # catch e; 
    #     # @show "Error during intersection" i1 i2; 
    #     push!(failing_pairs, (i1, i2))
    #     continue
    # end
    area_of_intersection = GO.area(GO.Spherical(), polygon_of_intersection)
    mat[linearizer1[i1...], linearizer2[i2...]] += area_of_intersection
end

failing_pairs

multipolys = [(; geometry = g) for g in GI.MultiPolygon.(([Trees.getcell(src_qt, i1...), Trees.getcell(dst_qt, i2...)] for (i1, i2) in failing_pairs))]
multipolys_longlat = GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), multipolys);

using GeoJSON
GeoJSON.write("bad_spherical_polys.geojson", multipolys_longlat)

# Extract idxs
i1, i2 = idxs[1]#failing_pairs[1]

# Extract cells from quadtree representations
p1 = Trees.getcell(src_qt, i1...)
p2 = Trees.getcell(dst_qt, i2...)

# Transform to geographic coordinates
p1_longlat = GO.transform(GO.GeographicFromUnitSphere(), p1)
p2_longlat = GO.transform(GO.GeographicFromUnitSphere(), p2)

# Display the points of the geographic coordinates for diagnostics
println("longlat points:")
display(GI.getexterior(p1_longlat) |> GI.getpoint)
display(GI.getexterior(p2_longlat) |> GI.getpoint)

println("spherical points:")
display(GI.getexterior(p1) |> GI.getpoint)
display(GI.getexterior(p2) |> GI.getpoint)

using GLMakie, GeoMakie
fig, ax, plt = poly(p1_longlat; transparency = true, alpha = 0.7, strokewidth = 1)
poly!(ax, p2_longlat; transparency = true, alpha = 0.7, strokewidth = 1)

GO.intersection(p1_longlat, p2_longlat; target = GO.PolygonTrait())
GO.intersection(GO.Spherical(), p1, p2; target = GO.PolygonTrait())
GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2; target = GO.PolygonTrait())
GO.intersection(GO.ConvexConvexSutherlandHodgman(), p1_longlat, p2_longlat; target = GO.PolygonTrait())
GO.intersection(GO.Spherical(), p1_longlat, p2_longlat; target = GO.PolygonTrait())


@be GO.intersection($(GO.ConvexConvexSutherlandHodgman(GO.Spherical())), $(p1), $(p2))
@be GO.intersection($(GO.ConvexConvexSutherlandHodgman(GO.Planar())), $(p1_longlat), $(p2_longlat))
@be GO.intersection($(p1_longlat), $(p2_longlat); target = $(GO.PolygonTrait()))

@be GO.intersection($(GO.Spherical()), $(p1), $(p2); target = $(GO.PolygonTrait()))

function f(p1, p2, N)
    p3 = GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2)
    for i in 1:N
        p3 = GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2)
    end
    return p3
end

f(p1, p2, 100)

@profview f(p1, p2, 100000)

fig, ax, plt = poly(p1_longlat; transparency = true, alpha = 0.7, strokewidth = 1, axis = (; type = GlobeAxis))
poly!(ax, p2_longlat; transparency = true, alpha = 0.7, strokewidth = 1)
meshimage!(ax, -180..180, -90..90, reshape([colorant"white"], 1, 1); zlevel = -300_000)

using GeoJSON
GeoJSON.write("bad_spherical_polys.geojson", [(; geometry = p1_longlat, name = "p1"), (; geometry = p2_longlat, name = "p2")])



areas = [
    begin
        poly = Trees.getcell(src_qt, i, j)
        intersection = GO.intersection(
            GO.ConvexConvexSutherlandHodgman(GO.Spherical()),
            poly,
            poly
        )
        area = GO.area(GO.Spherical(), intersection)
        area
        end
        for i in 1:ncells(src_qt, 1), j in 1:ncells(src_qt, 2)
    ]
    
