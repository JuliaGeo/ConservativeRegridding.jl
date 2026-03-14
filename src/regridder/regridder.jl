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

function margin_relative_error(current::AbstractVector, target::AbstractVector)
    T = promote_type(eltype(current), eltype(target))
    denom = max.(abs.(target), eps(T))
    return maximum(abs.(current .- target) ./ denom)
end

function intersection_margin_error(intersections, dst_areas::AbstractVector, src_areas::AbstractVector)
    row_sums = vec(sum(intersections; dims=2))
    col_sums = vec(sum(intersections; dims=1))
    row_err = margin_relative_error(row_sums, dst_areas)
    col_err = margin_relative_error(col_sums, src_areas)
    return max(row_err, col_err)
end

function reconcile_intersection_margins(intersections,
                                        dst_areas::AbstractVector,
                                        src_areas::AbstractVector;
                                        rtol = 1e-8,
                                        maxiter = 200)
    n_dst, n_src = size(intersections)
    T = promote_type(eltype(intersections), eltype(dst_areas), eltype(src_areas))

    dst_target = max.(zero(T), T.(dst_areas))
    src_target = max.(zero(T), T.(src_areas))

    # Unreachable margins are impossible to satisfy with this sparsity pattern.
    row_support = vec(sum(intersections; dims=2))
    col_support = vec(sum(intersections; dims=1))
    @inbounds for i in 1:n_dst
        if row_support[i] <= eps(T)
            dst_target[i] = zero(T)
        end
    end
    @inbounds for j in 1:n_src
        if col_support[j] <= eps(T)
            src_target[j] = zero(T)
        end
    end

    total_dst = sum(dst_target)
    total_src = sum(src_target)
    if total_dst > zero(T) && total_src > zero(T) && !isapprox(total_dst, total_src; rtol, atol = eps(T))
        dst_target .*= total_src / total_dst
    end

    row_scale = ones(T, n_dst)
    col_scale = ones(T, n_src)

    for iter in 1:maxiter
        row_den = intersections * col_scale
        @inbounds for i in eachindex(row_scale)
            target = dst_target[i]
            den = row_den[i]
            row_scale[i] = (target <= eps(T) || den <= eps(T)) ? zero(T) : target / den
        end

        col_den = transpose(intersections) * row_scale
        @inbounds for j in eachindex(col_scale)
            target = src_target[j]
            den = col_den[j]
            col_scale[j] = (target <= eps(T) || den <= eps(T)) ? zero(T) : target / den
        end

        if iter % 10 == 0 || iter == maxiter
            row_sums = row_scale .* (intersections * col_scale)
            col_sums = col_scale .* (transpose(intersections) * row_scale)
            row_err = margin_relative_error(row_sums, dst_target)
            col_err = margin_relative_error(col_sums, src_target)
            max(row_err, col_err) <= rtol && break
        end
    end

    return LinearAlgebra.Diagonal(row_scale) * intersections * LinearAlgebra.Diagonal(col_scale)
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
        reconcile_margins = true,
        margin_rtol = 1e-8,
        margin_maxiter = 200,
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

    if reconcile_margins
        mismatch = intersection_margin_error(intersections, dst_areas, src_areas)
        if mismatch > margin_rtol
            intersections = reconcile_intersection_margins(intersections, dst_areas, src_areas;
                                                           rtol = margin_rtol,
                                                           maxiter = margin_maxiter)
            dst_areas = vec(sum(intersections; dims=2))
            src_areas = vec(sum(intersections; dims=1))
        end
    end

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
    return vec([GO.area(manifold, cell) for cell in Trees.getcell(tree)])
end

function areas(manifold::GOCore.Manifold, tree)
    return vec([GO.area(manifold, cell) for cell in Trees.getcell(tree)])
end
