import GeometryOps: SpatialTreeInterface as STI

#=
## KnownFullSphereExtentWrapper

The idea here is that when you try to determine the spherical cap for a grid,
it's very hard to determine that the grid covers the entire sphere - there's 
usually a grid-cell-sized hole from most algorithms.  And they take a while.

So instead of doing that we can just intelligently wrap in this.
=#

"""
    KnownFullSphereExtentWrapper(tree)

A wrapper around a SpatialTreeInterface-compliant spatial tree that
will always return a `node_extent` that is a spherical cap covering the entire sphere.

All other STI methods are forwarded to `tree`.

This is useful for trees that are known to cover the entire sphere, since the user
does not have to compute large extents.

"""
struct KnownFullSphereExtentWrapper{T}
    tree::T
    function KnownFullSphereExtentWrapper(tree::T) where T
        if !STI.isspatialtree(tree)
            throw(ArgumentError("Tree is not a spatial tree"))
        end
        return new{T}(tree)
    end
end

STI.isspatialtree(::Type{<: KnownFullSphereExtentWrapper}) = true
STI.nchild(w::KnownFullSphereExtentWrapper) = STI.nchild(w.tree)
STI.getchild(w::KnownFullSphereExtentWrapper, i::Int) = STI.getchild(w.tree, i)
STI.isleaf(w::KnownFullSphereExtentWrapper) = STI.isleaf(w.tree)
STI.child_indices_extents(w::KnownFullSphereExtentWrapper) = STI.child_indices_extents(w.tree)
# This is the only specialization - you want a full sphere extent
# This is actually larger than a full sphere but it's fine
STI.node_extent(w::KnownFullSphereExtentWrapper) = GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint((0.,0.,1.)), Float64(pi) |> nextfloat, 1.2)

