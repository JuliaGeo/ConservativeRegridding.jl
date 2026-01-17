import GeometryOps: SpatialTreeInterface as STI
import GeometryOps as GO
import GeoInterface as GI
import ConservativeRegridding.Trees

struct CubedSphereToplevelTree{QuadtreeCursorType <: Trees.AbstractQuadtreeCursor}
    quadtrees::Vector{QuadtreeCursorType}
end

STI.isspatialtree(::Type{<: CubedSphereToplevelTree}) = true
STI.isleaf(::CubedSphereToplevelTree) = false
STI.nchild(cs::CubedSphereToplevelTree) = length(cs.quadtrees)
STI.getchild(cs::CubedSphereToplevelTree, i::Int) = cs.quadtrees[i]
STI.node_extent(cs::CubedSphereToplevelTree) = GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint((0.,0.,1.)), Float64(pi) |> nextfloat)
GOCore.best_manifold(c::CubedSphereToplevelTree) = GOCore.best_manifold(first(c.quadtrees))

Trees.getcell(c::CubedSphereToplevelTree) = Iterators.flatten(Trees.getcell(qt) for qt in c.quadtrees)
function Trees.getcell(c::CubedSphereToplevelTree, i::Int)
    n_cells_in_face = prod(Trees.ncells(first(c.quadtrees)))
    face_idx = fld(i - 1, n_cells_in_face) + 1
    cell_in_face = i - (face_idx - 1) * n_cells_in_face
    return Trees.getcell(c.quadtrees[face_idx], cell_in_face)
end

Trees.ncells(c::CubedSphereToplevelTree) = prod(Trees.ncells(first(c.quadtrees))) * 6