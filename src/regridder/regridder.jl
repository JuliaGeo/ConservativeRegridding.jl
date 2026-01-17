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

"""
Compute the area-weighted centroid of intersection polygons.
Returns `nothing` if no valid centroid can be computed.
"""
function _intersection_centroid(manifold, intersection_polys)
    n_polys = length(intersection_polys)

    if n_polys == 0
        return nothing
    elseif n_polys == 1
        c = GO.centroid(first(intersection_polys))
        return (GI.x(c), GI.y(c))
    else
        # Area-weighted centroid for multiple polygons
        cx, cy = 0.0, 0.0
        total_area = 0.0
        for poly in intersection_polys
            a = GO.area(manifold, poly)
            c = GO.centroid(poly)
            cx += GI.x(c) * a
            cy += GI.y(c) * a
            total_area += a
        end
        if total_area > 0
            return (cx / total_area, cy / total_area)
        else
            return nothing
        end
    end
end

"""
Second-order conservative regridder constructor.

Computes weights that incorporate gradient information from neighboring cells.
"""
function Regridder(
        manifold::M, method::Conservative2ndOrder, dst, src;
        normalize = true,
        threaded = _default_threaded(manifold),
        kwargs...
    ) where {M <: Manifold}

    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    _threaded = booltype(threaded)
    grad_info = compute_gradient_coefficients(manifold, src_tree)

    # Get candidate pairs via dual DFS
    predicate_f = M <: Spherical ? GO.UnitSpherical._intersects : Extents.intersects
    candidate_idxs = get_all_candidate_pairs(_threaded, predicate_f, src_tree, dst_tree)

    # Build sparse matrix entries
    n_dst = prod(Trees.ncells(dst_tree))
    n_src = prod(Trees.ncells(src_tree))

    i_dst = Int[]
    i_src = Int[]
    weights = Float64[]

    estimated_entries = length(candidate_idxs) * 5  # ~5x for neighbor contributions
    sizehint!(i_dst, estimated_entries)
    sizehint!(i_src, estimated_entries)
    sizehint!(weights, estimated_entries)

    for (src_idx, dst_idx) in candidate_idxs
        src_poly = Trees.getcell(src_tree, src_idx)
        dst_poly = Trees.getcell(dst_tree, dst_idx)

        intersection_polys = GO.intersection(
            GO.FosterHormannClipping(manifold), src_poly, dst_poly;
            target = GI.PolygonTrait()
        )

        overlap_area = GO.area(manifold, intersection_polys)
        if overlap_area <= 0
            continue
        end

        overlap_centroid = _intersection_centroid(manifold, intersection_polys)
        if isnothing(overlap_centroid)
            continue
        end

        gi = grad_info[src_idx]
        diff_x = overlap_centroid[1] - gi.centroid[1]
        diff_y = overlap_centroid[2] - gi.centroid[2]

        if gi.valid
            # 2nd order: source weight with gradient correction
            grad_term = diff_x * gi.src_grad[1] + diff_y * gi.src_grad[2]
            push!(i_dst, dst_idx)
            push!(i_src, src_idx)
            push!(weights, overlap_area * (1 - grad_term))

            # Neighbor contributions
            for (nbr_idx, nbr_grad) in zip(gi.neighbor_indices, gi.neighbor_grads)
                nbr_weight = (diff_x * nbr_grad[1] + diff_y * nbr_grad[2]) * overlap_area
                if abs(nbr_weight) > eps(Float64) * overlap_area
                    push!(i_dst, dst_idx)
                    push!(i_src, nbr_idx)
                    push!(weights, nbr_weight)
                end
            end
        else
            # Fallback to 1st order for this cell
            push!(i_dst, dst_idx)
            push!(i_src, src_idx)
            push!(weights, overlap_area)
        end
    end

    intersections = SparseArrays.sparse(i_dst, i_src, weights, n_dst, n_src)

    dst_areas = areas(manifold, dst, dst_tree)
    src_areas = areas(manifold, src, src_tree)

    dst_temp = zeros(length(dst_areas))
    src_temp = zeros(length(src_areas))

    regridder = Regridder(method, intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end