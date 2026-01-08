import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI

struct ImplicitQuadtree{PolyMatrixType <: AbstractMatrix}
    polygons::PolyMatrixType
end

struct ImplicitQuadtreeCursor{QuadtreeType <: ImplicitQuadtree}
    quadtree::QuadtreeType
    idx::CartesianIndex{2}
    "The level of the cursor - 1 is the base i.e. smallest polygon level, as you increase the number you increase the size of the thing."
    level::Int
end

function ImplicitQuadtreeCursor(quadtree::ImplicitQuadtree)
    max_level = ceil(Int, log2(max(size(quadtree.polygons)...)))
    return ImplicitQuadtreeCursor(quadtree, CartesianIndex(1, 1), max_level)
end

STI.isspatialtree(::Type{<: ImplicitQuadtreeCursor}) = true
function STI.nchild(q::ImplicitQuadtreeCursor)
    imax, jmax = (q.idx.I .- 1) .* (2^q.level) .+ 1
    ioff = size(q.quadtree.polygons, 1) - imax
    joff = size(q.quadtree.polygons, 2) - jmax
    nchildren = if ioff > 1
        2
    elseif ioff == 1
        1
    elseif ioff < 1
        0
    end * if joff > 1
        2
    elseif joff == 1
        1
    elseif joff < 1
        0
    end

    return nchildren
end

function STI.isleaf(q::ImplicitQuadtreeCursor)
    q.level < 1 && throw(ArgumentError("Quadtree level must be greater than 1; got $(q.level).  Something went wrong!"))
    return q.level == 1
end

function STI.getchild(q::ImplicitQuadtreeCursor, i::Int)
    i > STI.nchild(q) && throw(ArgumentError("Invalid child index; got $i, but there are only $(STI.nchild(q)) children in the node."))
    new_idx = ((q.idx.I .- 1) .* 2) .+ CartesianIndices((1:2, 1:2))[i].I
    return ImplicitQuadtreeCursor(q.quadtree, CartesianIndex(new_idx), q.level - 1)
end

function _get_corner_points(q::ImplicitQuadtreeCursor)
    if STI.isleaf(q)
        return GI.getpoint(q.quadtree.polygons[q.idx])
    else
        # Calculate the range of leaf indices covered by this node
        scale = 2^(q.level - 1)
        psize = size(q.quadtree.polygons)
        
        # Compute and clamp all indices to polygon matrix bounds
        imin = min((q.idx[1] - 1) * scale + 1, psize[1])
        imax = min(q.idx[1] * scale, psize[1])
        jmin = min((q.idx[2] - 1) * scale + 1, psize[2])
        jmax = min(q.idx[2] * scale, psize[2])
        
        # Collect points from all border polygons
        points = typeof(GI.getpoint(GI.getexterior(q.quadtree.polygons[imin, jmin]), 1))[]
        sizehint!(points, (imax - imin + 1) * (jmax - jmin + 1))
        # Top and bottom rows (all columns)
        for j in jmin:jmax
            append!(points, GI.getpoint(GI.getexterior(q.quadtree.polygons[imin, j])))
            if imax != imin
                append!(points, GI.getpoint(GI.getexterior(q.quadtree.polygons[imax, j])))
            end
        end
        
        # Left and right columns (excluding corners already added)
        for i in (imin + 1):(imax - 1)
            append!(points, GI.getpoint(GI.getexterior(q.quadtree.polygons[i, jmin])))
            if jmax != jmin
                append!(points, GI.getpoint(GI.getexterior(q.quadtree.polygons[i, jmax])))
            end
        end
        
        return points
    end
end

function STI.node_extent(q::ImplicitQuadtreeCursor)
    points = _get_corner_points(q)
    return GO.minimum_bounding_circle(GO.Welzl(GO.Spherical()), points)
end



# function circle_from_four_corners(p1, p2, p3, p4)
#     trans = GO.UnitSpherical.UnitSphereFromGeographic()
#     circle_from_four_corn
#     cx,cy = (x2 + x1) / 2, (y2 + y1) / 2
#     a, b, c, d, e, f, g, h = map(trans, ((x1, y1), (x2, y1), (x2, y2), (x1, y2),(cx,y1),(x2,cy),(cx,y2),(x1,cy)))
#     z = trans((cx,cy))
#     alld = map(p->spherical_distance(z,p), (a, b, c, d,e,f,g,h))
#     r = reduce(max, alld)
#     #The following is done to not miss intersections through numerical inaccuracies
#     res = SphericalCap(z, r*1.0001)
#     # if !all(_contains.((res,), (a,b,c,d)))
#     #     @show a,b,c,d,e,f,g,h
#     #     @show e
#     #     @show alld
#     #     error()
#     # end
#     res
# end