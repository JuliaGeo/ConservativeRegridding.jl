import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeoInterface as GI

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

"""
    Regridder(dst, src; normalize = false, intersection_operator, threaded = true, kwargs...)
    Regridder(manifold, dst, src; kwargs...)

The conservative regridder from `src` to `dst`: the sparse matrix of the
intersection areas of every overlapping cell pair and the cell areas of both
grids. `dst` and `src` are anything [`asgrid`](@ref) accepts: GridInterface
grids, the native grids and fields of the package extensions, or bare corner
matrices, axis pairs and polygon vectors. The engine calls
`treeify(grid; leaf_capacity = ForSweeps())` on both grids, reads the cells
through `cell_geometry` (the areas are those polygons' areas), and chooses the
clipper from `cellshape` ([`DefaultIntersectionOperator`](@ref)`(manifold,
src, dst)`, unless `intersection_operator` is given).

The two-argument form computes on the grids' own manifold, which must be the
same on both sides; the three-argument form computes on `manifold` (areas
included), for grids of different radii. The remaining keywords
(`npartitions`, `progress`, `cache`) go to [`intersection_areas`](@ref).
"""
function Regridder(dst, src; kwargs...)
    dst_manifold = input_manifold(dst)
    src_manifold = input_manifold(src)

    # Manifolds are never promoted implicitly: `best_manifold` guesses for input that
    # declares none, so a mismatch is as likely a bad guess as a deliberate mix.
    dst_manifold == src_manifold || throw(ArgumentError("""
    Destination and source manifolds must be the same.  Got $dst_manifold (destination) and $src_manifold (source).

    Pass the manifold to compute on explicitly: `Regridder(manifold, dst, src)`."""))

    return Regridder(dst_manifold, dst, src; kwargs...)
end

function Regridder(
        manifold::M, dst, src;
        normalize = false,
        intersection_operator = nothing,
        threaded = True(),
        kwargs...
    ) where {M <: Manifold}
    # "Normalize" the destination and source inputs into grids, and tree them.
    dst = asgrid(manifold, dst)
    src = asgrid(manifold, src)
    dst_tree = treeify(dst; leaf_capacity = ForSweeps())
    src_tree = treeify(src; leaf_capacity = ForSweeps())

    # The clipper follows the cell shapes of both grids unless an operator is given.
    op = intersection_operator === nothing ? DefaultIntersectionOperator(manifold, src, dst) : intersection_operator

    _threaded = booltype(threaded)

    # Compute the intersection areas.
    intersections = intersection_areas(
        manifold,
        _threaded,
        dst, src;
        intersection_operator = op,
        dst_tree, src_tree,
        kwargs...
    )

    # Compute the areas of each cell
    # of the destination and source grids.
    dst_areas = areas(manifold, dst)
    src_areas = areas(manifold, src)

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

"""
    areas(manifold, grid) -> Vector{Float64}
    areas(grid)

The areas of the cells of `grid` (anything [`asgrid`](@ref) accepts), in
position order, measured on `manifold`: the `GeometryOps.area` of
`cell_geometry(grid, i)`, the polygon the engine clips, so a regridder's
weights sum to exactly these (a grid's own `cell_area` may be analytic, as
a HEALPix pixel's, and differ from its polygon's area; a cell of no geometry,
as a layout ghost, has none).
"""
function areas(manifold::GOCore.Manifold, item)
    grid = asgrid(manifold, item)
    return Float64[Float64(GO.area(manifold, cell_geometry(grid, i))) for i in 1:ncells(grid)]
end
areas(item) = areas(input_manifold(item), item)
