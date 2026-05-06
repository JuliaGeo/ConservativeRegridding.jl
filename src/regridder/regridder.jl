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

# ────────────────────────────────────────────────────────────────────────────
# Mapping hierarchy — direction-specific data and dispatch tag for `Regridder`
# ────────────────────────────────────────────────────────────────────────────

"""
    AbstractMapping

Marker type for the kind of regridding a [`Regridder`](@ref) performs. Each
subtype encodes the direction-specific data (e.g. cell areas) and dispatches
the post-`mul!` behaviour. Concrete subtypes shipped with this package are
[`FVtoFV`](@ref), [`SEtoFV`](@ref), and [`FVtoSE`](@ref).

To add a new mapping:
- If the forward step is `dst ./= dst_areas`, subtype
  [`AbstractNormalizingMapping`](@ref) with a `dst_areas` field. `regrid!` is
  inherited.
- Otherwise, subtype `AbstractMapping` directly. The default `regrid!` does
  just `mul!`. If a different post-step is needed, define a
  `regrid!(::Regridder{<:Any, MyMapping}, …)` method.
"""
abstract type AbstractMapping end

"""
    AbstractNormalizingMapping <: AbstractMapping

Mappings whose forward `regrid!` step divides element-wise by destination
cell areas. Subtypes must carry a `dst_areas::AbstractVector` field.
"""
abstract type AbstractNormalizingMapping <: AbstractMapping end

"""
    FVtoFV{V <: AbstractVector} <: AbstractNormalizingMapping

Finite-volume → finite-volume mapping. Carries source and destination cell
areas; invertible by matrix transposition.
"""
struct FVtoFV{V <: AbstractVector} <: AbstractNormalizingMapping
    dst_areas :: V
    src_areas :: V
end

"""
    SEtoFV{V <: AbstractVector} <: AbstractNormalizingMapping

Spectral-element → finite-volume mapping. Carries destination cell areas only;
not invertible by transposition (the inverse direction is a different
algorithm — construct it explicitly with `Regridder(src, dst; …)`).
"""
struct SEtoFV{V <: AbstractVector} <: AbstractNormalizingMapping
    dst_areas :: V
end

"""
    FVtoSE <: AbstractMapping

Finite-volume → spectral-element mapping (per-element L2 projection). The
inverse mass matrix is already baked into the weight matrix, so no
post-`mul!` normalization is needed and no extra data is stored.
"""
struct FVtoSE <: AbstractMapping end

# ────────────────────────────────────────────────────────────────────────────
# Primary `Regridder` struct
# ────────────────────────────────────────────────────────────────────────────

struct Regridder{W, M <: AbstractMapping, V} <: AbstractRegridder
    "Sparse matrix of regridding weights"
    weight_matrix :: W
    "Mapping (direction-specific data + behaviour tag); see [`AbstractMapping`](@ref)"
    mapping :: M
    "Dense work array on the destination side (for non-contiguous inputs)"
    dst_temp :: V
    "Dense work array on the source side (for non-contiguous inputs)"
    src_temp :: V
end

# Convenience type aliases
const FVtoFVRegridder{W, V} = Regridder{W, <:FVtoFV, V}
const SEtoFVRegridder{W, V} = Regridder{W, <:SEtoFV, V}
const FVtoSERegridder{W, V} = Regridder{W, FVtoSE, V}

# ────────────────────────────────────────────────────────────────────────────
# Public accessors and shared interface methods
# ────────────────────────────────────────────────────────────────────────────

"""$(TYPEDSIGNATURES)
Destination cell areas. Defined for any [`AbstractNormalizingMapping`](@ref)
(currently `FVtoFV` and `SEtoFV`).
"""
destination_areas(r::Regridder{<:Any, <:AbstractNormalizingMapping}) = r.mapping.dst_areas

"""$(TYPEDSIGNATURES)
Source cell areas. Defined only for `FVtoFV` mappings.
"""
source_areas(r::Regridder{<:Any, <:FVtoFV}) = r.mapping.src_areas

Base.size(r::Regridder, args...) = size(r.weight_matrix, args...)

function Base.show(io::IO, ::MIME"text/plain", r::Regridder{W, M}) where {W, M}
    Ndst, Nsrc = size(r.weight_matrix)
    println(io, "Regridder{", nameof(M), "}: ", Nsrc, " → ", Ndst)
    println(io, "  weight_matrix :: ", typeof(r.weight_matrix))
    print(io,   "  mapping       :: ", r.mapping)
end

"""$(TYPEDSIGNATURES)
Return a Regridder for the reverse direction. Defined for `FVtoFV` mappings
(matrix transposition encodes the inverse). Other mappings raise an explicit
error pointing at the right alternative.
"""
function LinearAlgebra.transpose(r::Regridder{<:Any, <:FVtoFV})
    m = r.mapping
    return Regridder(transpose(r.weight_matrix),
                     FVtoFV(m.src_areas, m.dst_areas),
                     r.src_temp, r.dst_temp)
end

function LinearAlgebra.transpose(r::Regridder)
    error("`$(typeof(r.mapping))` is not invertible by transposition. " *
          "Construct the inverse-direction regridder with `Regridder(src, dst; …)`.")
end

"""$(TYPEDSIGNATURES)
Normalize the weight matrix and the FV-FV cell-area vectors by the matrix's
maximum entry. Defined only for `FVtoFV` mappings.
"""
function LinearAlgebra.normalize!(r::Regridder{<:Any, <:FVtoFV})
    norm = maximum(r.weight_matrix)
    r.weight_matrix ./= norm
    r.mapping.dst_areas ./= norm
    r.mapping.src_areas ./= norm
    return r
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

    # Construct the regridder. Normalize if requested.
    regridder = Regridder(intersections, FVtoFV(dst_areas, src_areas), dst_temp, src_temp)
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