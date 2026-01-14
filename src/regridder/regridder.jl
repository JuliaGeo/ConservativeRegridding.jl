using SortTileRecursiveTree
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeoInterface as GI

const DEFAULT_NODECAPACITY = 10
const DEFAULT_MANIFOLD = GeometryOps.Planar()
const DEFAULT_FLOATTYPE = Float64
const DEFAULT_MATRIX = SparseArrays.SparseMatrixCSC{DEFAULT_FLOATTYPE}
const DEFAULT_MATRIX_CONSTRUCTOR = SparseArrays.spzeros # SparseCSC for regridder

"""
    abstract type AbstractRegridder

Defines an interface for regridding operators.

Any subtype must implement the following methods:
- ` regrid!(dst_field::AbstractVector, regridder::AbstractRegridder, src_field::AbstractVector)`

See also: [`Regridder`](@ref).
"""
abstract type AbstractRegridder end

# Primary `Regridder` struct.
struct Regridder{W, A, V} <: AbstractRegridder
    "Matrix of area intersections between cells on the source and destination grid"
    intersections :: W 
    "Vector of areas on the destination grid"
    dst_areas :: A
    "Vector of areas on the source grid"
    src_areas :: A
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    dst_temp :: V
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    src_temp :: V
end

function Base.show(io::IO, regridder::Regridder{W, A, V}) where {W, A, V}
    n2, n1 = size(regridder)
    println(io, "$n2×$n1 Regridder{$W, $A, $V}")
    Base.print_array(io, regridder.intersections)
    println(io, "\n\nSource areas: ", regridder.src_areas)
    print(io, "Dest.  areas: ", regridder.dst_areas)
end

"""$(TYPEDSIGNATURES)
Return a Regridder for the backwards regridding, i.e. from destination to source grid.
Does not copy any data, i.e. regridder for forward and backward share the same underlying arrays."""
LinearAlgebra.transpose(regridder::Regridder) =
    Regridder(transpose(regridder.intersections), regridder.src_areas, regridder.dst_areas, regridder.src_temp, regridder.dst_temp)

Base.size(regridder::Regridder, args...) = size(regridder.intersections, args...)

function LinearAlgebra.normalize!(regridder::Regridder)
    (; intersections) = regridder
    norm = maximum(intersections)   # TODO is this the best normalizer?
    intersections ./= norm

    regridder.src_areas ./= norm
    regridder.dst_areas ./= norm
    return regridder
end

# allocate the areas matrix as SparseCSC if not provided
function intersection_areas(
    dst_field, # array of polygons of the target grid
    src_field; # array of polygons of the source grid
    T::Type{<:Number} = DEFAULT_FLOATTYPE,              # float type used for the areas matrix = regridder
    matrix_constructor = DEFAULT_MATRIX_CONSTRUCTOR,    # type of the areas matrix = regridder
    manifold::GeometryOps.Manifold = DEFAULT_MANIFOLD,
    kwargs...
)
    # unless `areas::AbstractMatrix` is provided (see in-place method ! below), create a SparseCSC matrix
    areas = matrix_constructor(T, length(dst_field), length(src_field))
    return compute_intersection_areas!(areas, dst_field, src_field; kwargs...)
end

struct DefaultIntersectionFailureError{T1, T2, E} <: Base.Exception
    p1::T1
    p2::T2
    e::E
end

function Base.showerror(io::IO, e::DefaultIntersectionFailureError)
    print(io, "Intersection failed with the following error.  Capture this error object and access `err.p1` and `err.p2` to access the polygons that failed to intersect.")
    Base.showerror(io, e.e)
end

"""
    DefaultIntersectionOperator(manifold::GeometryOps.Manifold)

Default intersection operator for the given manifold.

Implemented for `Planar` and `Spherical` manifolds at the moment.
Will dispatch to the appropriate intersection operator / algorithm based on the manifold.
"""
struct DefaultIntersectionOperator{M}
    manifold::M
end

function (op::DefaultIntersectionOperator{<: GeometryOps.Planar})(p1, p2)
    intersection_polys = try
        GeometryOps.intersection(p1, p2; target = GeoInterface.PolygonTrait())
    catch
        throw(DefaultIntersectionFailureError(p1, p2, e))
    end
    return GeometryOps.area(intersection_polys)
end

function (op::DefaultIntersectionOperator{M})(p1, p2) where {M <: GeometryOps.Spherical}
    intersection_polys = try
        GeometryOps.intersection(GeometryOps.ConvexConvexSutherlandHodgman(op.manifold), p1, p2; target = GeoInterface.PolygonTrait())
    catch
        throw(DefaultIntersectionFailureError(p1, p2, e))
    end
    return GeometryOps.area(op.manifold, intersection_polys)
end

function compute_intersection_areas!(
    areas::AbstractMatrix,  # intersection areas for all combinatations of grid cells in field1, field2
    dst_grid,               # array of polygons
    src_grid;               # array of polygons
    manifold::GeometryOps.Manifold = DEFAULT_MANIFOLD,      # TODO currently not used
    dst_nodecapacity = DEFAULT_NODECAPACITY,
    src_nodecapacity = DEFAULT_NODECAPACITY,
    intersection_operator::F = DefaultIntersectionOperator(manifold)
) where {F}
    # Prepare STRtrees for the two grids, to speed up intersection queries
    # we may want to separately tune nodecapacity if one is much larger than the other.
    # specifically we may want to tune leaf node capacity via Hilbert packing while still
    # constraining inner node capacity.  But that can come later.
    tree1 = SortTileRecursiveTree.STRtree(src_grid; nodecapacity = src_nodecapacity)
    tree2 = SortTileRecursiveTree.STRtree(dst_grid; nodecapacity = dst_nodecapacity)
    # Do the dual query, which is the most efficient way to do this,
    # by iterating down both trees simultaneously, rejecting pairs of nodes that do not intersect.
    # when we find an intersection, we calculate the area of the intersection and add it to the result matrix.
    GeometryOps.SpatialTreeInterface.dual_depth_first_search(Extents.intersects, tree1, tree2) do i1, i2
        p1, p2 = src_grid[i1], dst_grid[i2]
        # may want to check if the polygons intersect first,
        # to avoid antimeridian-crossing multipolygons viewing a scanline.
        area_of_intersection = intersection_operator(p1, p2)
        if area_of_intersection > 0
            areas[i2, i1] += area_of_intersection
        end
    end

    return areas
end

get_vertices(polygons) = polygons       # normally vector of polygons, which are vectors of points
get_vertices(polygons::AbstractMatrix) = eachcol(polygons)

"""$(TYPEDSIGNATURES)
Return a Regridder that transfers data from `src_field` to `dst_field`.

Regridder stores the intersection areas between
The areas are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids. Additional `kwargs` are passed to the `intersection_areas` function.
"""
function Regridder(
    dst_vertices, # assumes something like this ::AbstractVector{<:AbstractVector{Tuple}},
    src_vertices;
    normalize::Bool = true,
    kwargs...
)
    # wrap into GeoInterface.Polygon, apply antimeridian cuttng via fix
    dst_polys = GeoInterface.Polygon.(GeoInterface.LinearRing.(get_vertices(dst_vertices))) .|> GeometryOps.fix
    src_polys = GeoInterface.Polygon.(GeoInterface.LinearRing.(get_vertices(src_vertices))) .|> GeometryOps.fix

    intersections = intersection_areas(manifold, dst_polys, src_polys; kwargs...)

    # If the two grids completely overlap, then the areas should be equivalent
    # to the sum of the intersection areas along the second and fisrt dimensions,
    # for src and dst, respectively. This is not the case if the two grids do not cover the same area.
    dst_areas = GeometryOps.area.(dst_polys)
    src_areas = GeometryOps.area.(src_polys)

    # TODO: make this GPU-compatible?
    dst_temp = zeros(length(dst_areas))
    src_temp = zeros(length(src_areas))

    regridder = Regridder(intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)
    return regridder
end

function Regridder(dst, src; kwargs...)
    dst_manifold = GOCore.best_manifold(dst)
    src_manifold = GOCore.best_manifold(src)

    manifold = if dst_manifold != src_manifold
        # Implicitly promote to spherical
        if dst_manifold == GO.Planar() && src_manifold == GO.Spherical()
            GO.Spherical()
        elseif dst_manifold == GO.Spherical() && src_manifold == GO.Planar()
            GO.Spherical()
        else
            error("Destination and source manifolds must be the same.  Got $dst_manifold and $src_manifold.")
        end
    else
        dst_manifold
    end

    return Regridder(manifold, dst, src; kwargs...)
end

function Regridder(manifold::GOCore.Manifold, dst, src; normalize = true, intersection_operator::F = DefaultIntersectionOperator(manifold), kwargs...) where {F}
    # "Normalize" the destination and source grids into trees.
    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    # Compute the intersection areas.
    intersections = intersection_areas(manifold, dst_tree, src_tree; intersection_operator, kwargs...)

    # Compute the areas of each cell
    # of the destination and source grids.
    dst_areas = areas(manifold, dst, dst_tree)
    src_areas = areas(manifold, src, src_tree)

    # TODO: make this GPU-compatible?
    # Allocate temporary arrays for the regridding operation - 
    # in case the destination and source fields are not contiguous in memory.
    dst_temp = zeros(length(dst_areas))
    src_temp = zeros(length(src_areas))

    # Construct the regridder.  Normalize if requested.
    regridder = Regridder(intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end

function areas(manifold::GOCore.Manifold, item, tree)
    areas(manifold, tree)
end

areas(manifold::GOCore.Manifold, tree::Trees.AbstractTreeWrapper) = areas(manifold, Trees.parent(tree))

function areas(manifold::GOCore.Manifold, tree::Trees.AbstractQuadtreeCursor)
    @assert Trees.istoplevel(tree) "Areas are only valid for the top level of the quadtree."
    grid = Trees.getgrid(tree)
    return vec([GO.area(manifold, cell) for cell in Trees.getcell(tree)])
end

function intersection_areas(manifold::GOCore.Manifold, dst_tree, src_tree, )
end