
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

nrows, ncols = 8, 9
polygons = [
    GI.Polygon([
        GI.LinearRing([
            (Float64(j - 1), Float64(i - 1)),  # bottom-left
            (Float64(j),     Float64(i - 1)),  # bottom-right
            (Float64(j),     Float64(i)),      # top-right
            (Float64(j - 1), Float64(i)),      # top-left
            (Float64(j - 1), Float64(i - 1))   # close the ring
        ])
    ])
    for i in 1:nrows, j in 1:ncols
]
    
# Create the quadtree
qt = ImplicitQuadtree(polygons)
qtc = ImplicitQuadtreeCursor(qt)

f, a, p = lines(vec(polygons); axis = (; type = GlobeAxis), transparency = true)
lines!(a, GeoMakie.coastlines(); zlevel = -100_000, color = :black, alpha = 0.5, transparency = true)
meshimage!(a, -180..180, -90..90, reshape([colorant"white"], 1, 1); zlevel = -300_000)

co = Observable{GO.UnitSpherical.SphericalCap{Float64}}(GO.minimum_bounding_circle(GO.Welzl(GO.Spherical()), vec(polygons)))

poly!(a, co; color = :steelblue, alpha = 0.7, transparency = true, zlevel = 0)
poly!(a, STI.node_extent(STI.getchild(qtc, 1)); color = :red, alpha = .7, transparency=true)






import GeoInterface as GI
# Test case: Create a grid of rectangle polygons
function test_quadtree_with_rectangle_grid()
    # Create an 8x4 grid of unit rectangles
    nrows, ncols = 8, 4
    polygons = [
        GI.Polygon([
            GI.LinearRing([
                (Float64(j - 1), Float64(i - 1)),  # bottom-left
                (Float64(j),     Float64(i - 1)),  # bottom-right
                (Float64(j),     Float64(i)),      # top-right
                (Float64(j - 1), Float64(i)),      # top-left
                (Float64(j - 1), Float64(i - 1))   # close the ring
            ])
        ])
        for i in 1:nrows, j in 1:ncols
    ]
    
    # Create the quadtree
    qt = ImplicitQuadtree(polygons)
    qtc = ImplicitQuadtreeCursor(qt)
    
    println("Grid size: ", size(polygons))
    println("Root level: ", qtc.level)
    println("Root nchildren: ", STI.nchild(qtc))
    println("Root isleaf: ", STI.isleaf(qtc))
    
    # Test traversing children
    println("\nTraversing first child:")
    child1 = STI.getchild(qtc, 1)
    println("  Child 1 idx: ", child1.idx)
    println("  Child 1 level: ", child1.level)
    println("  Child 1 nchildren: ", STI.nchild(child1))
    println("  Child 1 isleaf: ", STI.isleaf(child1))
    
    # Test _get_corner_points on root
    println("\nGetting corner points for root:")
    root_points = _get_corner_points(qtc)
    println("  Number of points: ", length(root_points))
    
    # Test _get_corner_points on a leaf
    println("\nTraversing to a leaf:")
    cursor = qtc
    while !STI.isleaf(cursor) && STI.nchild(cursor) > 0
        cursor = STI.getchild(cursor, 1)
        println("  Level: ", cursor.level, ", idx: ", cursor.idx, ", isleaf: ", STI.isleaf(cursor))
    end
    
    leaf_points = _get_corner_points(cursor)
    println("  Leaf points: ", leaf_points)
    
    return qt, qtc
end
