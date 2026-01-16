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
    AbstractRegridMethod

Abstract supertype for regridding methods.
Subtypes control how weights are computed during Regridder construction.
"""
abstract type AbstractRegridMethod end

"""
    Conservative1stOrder()

First-order conservative regridding. Weights are pure area ratios.
This is the default method and matches the original behavior.
"""
struct Conservative1stOrder <: AbstractRegridMethod end

"""
    Conservative2ndOrder()

Second-order conservative regridding. Weights incorporate gradient
information from neighboring cells for improved accuracy on smooth fields.

Note: Cannot be transposed. Build separate regridders for each direction.
"""
struct Conservative2ndOrder <: AbstractRegridMethod end

"""
    abstract type AbstractRegridder

Defines an interface for regridding operators.

Any subtype must implement the following methods:
- ` regrid!(dst_field::AbstractVector, regridder::AbstractRegridder, src_field::AbstractVector)`

See also: [`Regridder`](@ref).
"""
abstract type AbstractRegridder end

# Primary `Regridder` struct.
struct Regridder{M <: AbstractRegridMethod, W, A, V} <: AbstractRegridder
    "The regridding method used"
    method::M
    "Matrix of area intersections between cells on the source and destination grid"
    intersections::W
    "Vector of areas on the destination grid"
    dst_areas::A
    "Vector of areas on the source grid"
    src_areas::A
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    dst_temp::V
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    src_temp::V
end

function Base.show(io::IO, regridder::Regridder{M, W, A, V}) where {M, W, A, V}
    n2, n1 = size(regridder)
    println(io, "$n2×$n1 Regridder{$M, $W, $A, $V}")
    Base.print_array(io, regridder.intersections)
    println(io, "\n\nSource areas: ", regridder.src_areas)
    print(io, "Dest.  areas: ", regridder.dst_areas)
end

"""$(TYPEDSIGNATURES)
Return a Regridder for the backwards regridding, i.e. from destination to source grid.
Does not copy any data, i.e. regridder for forward and backward share the same underlying arrays.

Note: Throws an error for Conservative2ndOrder since 2nd order weights are asymmetric.
"""
function LinearAlgebra.transpose(regridder::Regridder{<:Conservative2ndOrder})
    error("Cannot transpose a 2nd order regridder. " *
          "Build a separate Regridder(src, dst; method=Conservative2ndOrder()) for reverse direction.")
end

function LinearAlgebra.transpose(regridder::Regridder{M}) where M <: AbstractRegridMethod
    Regridder(regridder.method, transpose(regridder.intersections),
              regridder.src_areas, regridder.dst_areas,
              regridder.src_temp, regridder.dst_temp)
end

Base.size(regridder::Regridder, args...) = size(regridder.intersections, args...)

function LinearAlgebra.normalize!(regridder::Regridder)
    (; intersections) = regridder
    norm = maximum(intersections)   # TODO is this the best normalizer?
    intersections ./= norm

    regridder.src_areas ./= norm
    regridder.dst_areas ./= norm
    return regridder
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
    intersection_polys = #=try; =#
        GeometryOps.intersection(GO.FosterHormannClipping(GO.Planar()), p1, p2; target = GeoInterface.PolygonTrait())
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return GeometryOps.area(GO.Planar(), intersection_polys)
end

function (op::DefaultIntersectionOperator{M})(p1, p2) where {M <: GeometryOps.Spherical}
    intersection_polys = #=try; =#
        GeometryOps.intersection(GeometryOps.ConvexConvexSutherlandHodgman(op.manifold), p1, p2; target = GeoInterface.PolygonTrait())
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return GeometryOps.area(op.manifold, intersection_polys)
end

function Regridder(dst, src; method::AbstractRegridMethod = Conservative1stOrder(), kwargs...)
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

    return Regridder(manifold, method, dst, src; kwargs...)
end

# Default threading behavior based on manifold.
# Planar manifolds don't yet have _area_criterion implemented for multithreading.
_default_threaded(::Spherical) = True()
_default_threaded(::Planar) = False()

# Convenience constructor that accepts method as keyword argument
function Regridder(
        manifold::M, dst, src;
        method::AbstractRegridMethod = Conservative1stOrder(),
        kwargs...
    ) where {M <: Manifold}
    return Regridder(manifold, method, dst, src; kwargs...)
end

function Regridder(
        manifold::M, method::Conservative1stOrder, dst, src;
        normalize = true,
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        threaded = _default_threaded(manifold),
        kwargs...
    ) where {M <: Manifold, F}
    # "Normalize" the destination and source grids into trees.
    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    _threaded = booltype(threaded)

    # Compute the intersection areas.
    intersections = intersection_areas(
        manifold,
        _threaded,
        dst_tree, src_tree;
        intersection_operator,
        kwargs...
    )

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
    regridder = Regridder(method, intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end

function areas(manifold::GOCore.Manifold, item, tree)
    areas(manifold, tree)
end

areas(manifold::GOCore.Manifold, tree::Trees.AbstractTreeWrapper) = areas(manifold, Trees.parent(tree))

function areas(manifold::GOCore.Manifold, tree::Trees.AbstractQuadtreeCursor)
    @assert Trees.istoplevel(tree) "Areas are only valid for the top level of the quadtree."
    return vec([GO.area(manifold, cell) for cell in Trees.getcell(tree)])
end

function areas(manifold::GOCore.Manifold, tree)
    return vec([GO.area(manifold, cell) for cell in Trees.getcell(tree)])
end