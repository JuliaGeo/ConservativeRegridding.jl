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
    abstract type RegriddingAlgorithm

Tag type that selects which weight-construction algorithm a [`Regridder`](@ref)
uses. The active algorithm is stored on the regridder so that capability
queries (`supports_transpose`, etc.) and dispatchers (`Base.transpose`,
`build_weights`) can specialize on it.

Concrete subtypes:
- [`FirstOrderConservative`](@ref) — area-weighted; preserves the integral.
- `SecondOrderConservative` (in `second_order.jl`) — adds a per-cell
  Taylor reconstruction with a Green's-theorem gradient stencil.
"""
abstract type RegriddingAlgorithm end

"""
    FirstOrderConservative()

The classical area-weighted conservative regridder. The intersection-area
matrix `A[dst, src]` is built from the geometric overlap between source and
destination cells, and a destination value is `(A * src) ./ dst_areas`.
Supports `Base.transpose`.
"""
struct FirstOrderConservative <: RegriddingAlgorithm end

"""
    supports_transpose(algorithm::RegriddingAlgorithm) -> Bool

Whether `Base.transpose(::Regridder)` is well-defined for a regridder built
with this algorithm. Default `false`; methods on concrete algorithm types may
override it. `Regridder`s whose algorithm reports `false` will raise a
`MethodError` on `transpose`.
"""
supports_transpose(::RegriddingAlgorithm) = false
supports_transpose(::FirstOrderConservative) = true

"""
    abstract type AbstractRegridder

Defines an interface for regridding operators.

Any subtype must implement the following methods:
- ` regrid!(dst_field::AbstractVector, regridder::AbstractRegridder, src_field::AbstractVector)`

See also: [`Regridder`](@ref).
"""
abstract type AbstractRegridder end

# Primary `Regridder` struct.
struct Regridder{T <: AbstractFloat, ALG <: RegriddingAlgorithm, W, A, V} <: AbstractRegridder
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
    "Algorithm used to construct the weight matrix"
    algorithm :: ALG
end

function Regridder(intersections::W, dst_areas::A, src_areas::A, dst_temp::V, src_temp::V, algorithm::ALG) where {ALG <: RegriddingAlgorithm, W, A, V}
    T = eltype(A)
    return Regridder{T, ALG, W, A, V}(intersections, dst_areas, src_areas, dst_temp, src_temp, algorithm)
end

function Base.show(io::IO, regridder::Regridder{T, ALG, W, A, V}) where {T, ALG, W, A, V}
    n2, n1 = size(regridder)
    println(io, "$n2×$n1 Regridder{$T, $(nameof(ALG)), …}")
    Base.print_array(io, regridder.intersections)
    println(io, "\n\nSource areas: ", regridder.src_areas)
    print(io, "Dest.  areas: ", regridder.dst_areas)
end

"""$(TYPEDSIGNATURES)
Return a Regridder for the backwards regridding, i.e. from destination to source grid.
Does not copy any data, i.e. regridder for forward and backward share the same underlying arrays.

Dispatches on the algorithm: only algorithms with `supports_transpose(alg) === true`
provide a `Base.transpose(::Regridder, alg)` method. Otherwise a `MethodError`
is raised."""
LinearAlgebra.transpose(regridder::Regridder) = LinearAlgebra.transpose(regridder, regridder.algorithm)

function LinearAlgebra.transpose(regridder::Regridder, ::FirstOrderConservative)
    return Regridder(transpose(regridder.intersections), regridder.src_areas, regridder.dst_areas,
                     regridder.src_temp, regridder.dst_temp, regridder.algorithm)
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

function Regridder(
        manifold::M, dst, src;
        normalize = true,
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        threaded = True(),
        algorithm::RegriddingAlgorithm = FirstOrderConservative(),
        T::Type{<:AbstractFloat} = DEFAULT_FLOATTYPE,
        kwargs...
    ) where {M <: Manifold, F}
    # "Normalize" the destination and source grids into trees.
    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    _threaded = booltype(threaded)

    # Compute the weight (intersection) matrix; algorithm-dispatched.
    intersections = build_weights(
        algorithm, manifold, _threaded, dst_tree, src_tree;
        intersection_operator, T, kwargs...
    )

    # Compute the areas of each cell of the destination and source grids.
    dst_areas = convert(Vector{T}, areas(manifold, dst, dst_tree))
    src_areas = convert(Vector{T}, areas(manifold, src, src_tree))

    # TODO: make this GPU-compatible?
    # Allocate temporary arrays for the regridding operation —
    # in case the destination and source fields are not contiguous in memory.
    dst_temp = zeros(T, length(dst_areas))
    src_temp = zeros(T, length(src_areas))

    # Construct the regridder.  Normalize if requested.
    regridder = Regridder(intersections, dst_areas, src_areas, dst_temp, src_temp, algorithm)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end

"""
    build_weights(algorithm, manifold, threaded, dst_tree, src_tree; kwargs...)

Construct the sparse weight (intersection) matrix used by a `Regridder`.
Multiple-dispatched on `algorithm <: RegriddingAlgorithm`. The first-order
method is the area-weighted overlap matrix; second-order lives in
`second_order.jl`.
"""
function build_weights end

function build_weights(::FirstOrderConservative, manifold, threaded, dst_tree, src_tree;
                       intersection_operator, T = DEFAULT_FLOATTYPE, kwargs...)
    return intersection_areas(manifold, threaded, dst_tree, src_tree;
                               intersection_operator, T, kwargs...)
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