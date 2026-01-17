include("quadtree_geoms.jl")


import GeoInterface as GI, GeometryOps as GO

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
