#=
# Gradient Computation via Green's Theorem

Compute gradient coefficients for 2nd order conservative regridding.
Uses Green's theorem on the "neighbor polygon" formed by neighbor centroids.
=#

import GeometryOps as GO
import GeoInterface as GI
import GeometryOpsCore as GOCore
import LinearAlgebra

"""
    GradientInfo

Stores gradient coefficients for a single source cell.

Fields:
- `valid`: Whether gradients are valid (>=3 neighbors, centroid inside neighbor polygon)
- `centroid`: Cell centroid (x, y) or (x, y, z) for spherical
- `src_grad`: Gradient coefficient for the source cell itself
- `neighbor_indices`: Linear indices of neighbor cells
- `neighbor_grads`: Gradient coefficients for each neighbor
"""
struct GradientInfo{T}
    valid::Bool
    centroid::NTuple{2, T}
    src_grad::NTuple{2, T}
    neighbor_indices::Vector{Int}
    neighbor_grads::Vector{NTuple{2, T}}
end

# Constructor for invalid gradient info (fallback to 1st order)
function GradientInfo{T}(centroid::NTuple{2, T}, neighbor_indices::Vector{Int}) where T
    zero_grad = (zero(T), zero(T))
    GradientInfo{T}(false, centroid, zero_grad, neighbor_indices, [zero_grad for _ in neighbor_indices])
end

"""
    compute_gradient_coefficients(manifold, tree) -> Vector{GradientInfo}

Compute gradient coefficients for all cells using Green's theorem.

For each cell:
1. Get neighbors from adjacency
2. Compute neighbor centroids
3. Sort neighbors counter-clockwise around source centroid
4. Form neighbor polygon and apply Green's theorem
5. If validation fails, mark as invalid (will use 1st order fallback)
"""
function compute_gradient_coefficients(manifold::GOCore.Manifold, tree)
    adjacency = compute_adjacency(manifold, tree)
    n_cells = length(adjacency)
    T = Float64

    # Compute all centroids first
    centroids = Vector{NTuple{2, T}}(undef, n_cells)
    for i in 1:n_cells
        cell = Trees.getcell(tree, i)
        c = GO.centroid(cell)
        centroids[i] = (GI.x(c), GI.y(c))
    end

    # Compute gradient info for each cell
    grad_info = Vector{GradientInfo{T}}(undef, n_cells)

    for i in 1:n_cells
        neighbor_indices = adjacency[i]
        src_centroid = centroids[i]

        # Need at least 3 neighbors for valid gradient
        if length(neighbor_indices) < 3
            grad_info[i] = GradientInfo{T}(src_centroid, neighbor_indices)
            continue
        end

        # Get neighbor centroids
        neighbor_centroids = [centroids[j] for j in neighbor_indices]

        # Sort neighbors counter-clockwise around source centroid
        sorted_indices, sorted_centroids = _sort_neighbors_ccw(src_centroid, neighbor_indices, neighbor_centroids)

        # Compute gradient coefficients via Green's theorem
        src_grad, neighbor_grads, valid = _compute_greens_theorem_gradients(
            src_centroid, sorted_centroids
        )

        grad_info[i] = GradientInfo{T}(
            valid,
            src_centroid,
            src_grad,
            sorted_indices,
            neighbor_grads
        )
    end

    return grad_info
end

"""
Sort neighbor indices and centroids counter-clockwise around the source centroid.
"""
function _sort_neighbors_ccw(src_centroid::NTuple{2, T}, indices::Vector{Int},
                              centroids::Vector{NTuple{2, T}}) where T
    # Compute angles from source centroid to each neighbor
    angles = [atan(c[2] - src_centroid[2], c[1] - src_centroid[1]) for c in centroids]

    # Sort by angle
    perm = sortperm(angles)

    return indices[perm], centroids[perm]
end

"""
Compute gradient coefficients using Green's theorem on the neighbor polygon.

Returns (src_grad, neighbor_grads, valid).
"""
function _compute_greens_theorem_gradients(src_centroid::NTuple{2, T},
                                            neighbor_centroids::Vector{NTuple{2, T}}) where T
    n = length(neighbor_centroids)

    # Initialize gradients
    src_grad = (zero(T), zero(T))
    neighbor_grads = Vector{NTuple{2, T}}(undef, n)

    # Compute area of neighbor polygon (for validation and normalization)
    area = _polygon_area(neighbor_centroids)

    if abs(area) < eps(T) * 100
        # Degenerate polygon
        return src_grad, [src_grad for _ in 1:n], false
    end

    # Check if source centroid is inside neighbor polygon
    if !_point_in_polygon(src_centroid, neighbor_centroids)
        return src_grad, [src_grad for _ in 1:n], false
    end

    # Apply Green's theorem
    # For each edge of the neighbor polygon, compute contribution to gradient
    inv_area = one(T) / area

    for i in 1:n
        j = mod1(i + 1, n)  # next vertex (wrapping)

        p1 = neighbor_centroids[i]
        p2 = neighbor_centroids[j]

        # Edge vector
        dx = p2[1] - p1[1]
        dy = p2[2] - p1[2]

        # Outward normal (perpendicular to edge, pointing outward for CCW polygon)
        # For a CCW polygon, outward normal is (dy, -dx)
        nx = dy
        ny = -dx

        # Edge length
        edge_len = sqrt(dx^2 + dy^2)
        if edge_len < eps(T)
            continue
        end

        # Normalize and scale by edge length / (2 * area)
        # This is the Green's theorem coefficient
        scale = edge_len * inv_area / 2

        # Gradient contribution for vertex i and j
        grad_contrib = (nx * scale, ny * scale)

        # Add to neighbor gradients (each edge contributes to both endpoints)
        if i == 1
            neighbor_grads[i] = grad_contrib
        else
            neighbor_grads[i] = (neighbor_grads[i][1] + grad_contrib[1],
                                 neighbor_grads[i][2] + grad_contrib[2])
        end

        if j == 1
            neighbor_grads[j] = (neighbor_grads[j][1] + grad_contrib[1],
                                 neighbor_grads[j][2] + grad_contrib[2])
        elseif j > i  # Only initialize if not already set
            neighbor_grads[j] = grad_contrib
        else
            neighbor_grads[j] = (neighbor_grads[j][1] + grad_contrib[1],
                                 neighbor_grads[j][2] + grad_contrib[2])
        end
    end

    # Source gradient is negative sum of neighbor gradients (conservation)
    src_grad_x = zero(T)
    src_grad_y = zero(T)
    for ng in neighbor_grads
        src_grad_x -= ng[1]
        src_grad_y -= ng[2]
    end
    src_grad = (src_grad_x, src_grad_y)

    return src_grad, neighbor_grads, true
end

"""
Compute signed area of a polygon given as a list of vertices (CCW = positive).
"""
function _polygon_area(vertices::Vector{NTuple{2, T}}) where T
    n = length(vertices)
    area = zero(T)
    for i in 1:n
        j = mod1(i + 1, n)
        area += vertices[i][1] * vertices[j][2]
        area -= vertices[j][1] * vertices[i][2]
    end
    return area / 2
end

"""
Check if a point is inside a polygon (ray casting algorithm).
"""
function _point_in_polygon(point::NTuple{2, T}, vertices::Vector{NTuple{2, T}}) where T
    n = length(vertices)
    inside = false

    px, py = point
    j = n

    for i in 1:n
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end

    return inside
end
