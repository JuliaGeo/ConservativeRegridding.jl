module TriangleQuadrature

# Dunavant symmetric quadrature rules on the reference triangle with
# vertices (0,0), (1,0), (0,1) (area = 1/2). Points stored as
# barycentric coordinates (λ₁, λ₂, λ₃) with λ₁ + λ₂ + λ₃ = 1, weights
# normalized so Σ w = 1/2.
#
# Source: Dunavant (1985). Reproduced from standard FEM tables.

const _RULES = Dict{Int, Tuple{Vector{NTuple{3, Float64}}, Vector{Float64}}}()

# degree 1, 1 point
_RULES[1] = (
    [(1/3, 1/3, 1/3)],
    [0.5],
)

# degree 2, 3 points
_RULES[2] = (
    [(2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)],
    [1/6, 1/6, 1/6],
)

# degree 3, 4 points (one negative weight is acceptable for our use)
_RULES[3] = (
    [(1/3, 1/3, 1/3),
     (3/5, 1/5, 1/5), (1/5, 3/5, 1/5), (1/5, 1/5, 3/5)],
    [-27/96, 25/96, 25/96, 25/96],
)

# degree 4, 6 points
_RULES[4] = let
    a = 0.445948490915965
    b = 0.091576213509771
    wa = 0.111690794839005
    wb = 0.054975871827661
    (
        [(1 - 2a, a, a), (a, 1 - 2a, a), (a, a, 1 - 2a),
         (1 - 2b, b, b), (b, 1 - 2b, b), (b, b, 1 - 2b)],
        [wa, wa, wa, wb, wb, wb],
    )
end

# degree 5, 7 points
_RULES[5] = let
    a = (6 + sqrt(15)) / 21
    b = (6 - sqrt(15)) / 21
    wa = (155 + sqrt(15)) / 2400
    wb = (155 - sqrt(15)) / 2400
    (
        [(1/3, 1/3, 1/3),
         (1 - 2a, a, a), (a, 1 - 2a, a), (a, a, 1 - 2a),
         (1 - 2b, b, b), (b, 1 - 2b, b), (b, b, 1 - 2b)],
        [9/80, wa, wa, wa, wb, wb, wb],
    )
end

# degree 6, 12 points (Dunavant)
_RULES[6] = (
    [
        (0.873821971016996, 0.063089014491502, 0.063089014491502),
        (0.063089014491502, 0.873821971016996, 0.063089014491502),
        (0.063089014491502, 0.063089014491502, 0.873821971016996),
        (0.501426509658179, 0.249286745170910, 0.249286745170910),
        (0.249286745170910, 0.501426509658179, 0.249286745170910),
        (0.249286745170910, 0.249286745170910, 0.501426509658179),
        (0.636502499121399, 0.310352451033785, 0.053145049844816),
        (0.636502499121399, 0.053145049844816, 0.310352451033785),
        (0.310352451033785, 0.636502499121399, 0.053145049844816),
        (0.310352451033785, 0.053145049844816, 0.636502499121399),
        (0.053145049844816, 0.636502499121399, 0.310352451033785),
        (0.053145049844816, 0.310352451033785, 0.636502499121399),
    ],
    [
        0.025422453185103, 0.025422453185103, 0.025422453185103,
        0.058393137863189, 0.058393137863189, 0.058393137863189,
        0.041425537809187, 0.041425537809187, 0.041425537809187,
        0.041425537809187, 0.041425537809187, 0.041425537809187,
    ],
)

# degree 7, 13 points (Dunavant). Weights from Dunavant (1985) Table II
# scaled by 1/2 so they sum to the reference triangle area 1/2.
# Note: the centroid weight is negative (acceptable for our use).
_RULES[7] = (
    [
        (1/3, 1/3, 1/3),
        (0.479308067841920, 0.260345966079040, 0.260345966079040),
        (0.260345966079040, 0.479308067841920, 0.260345966079040),
        (0.260345966079040, 0.260345966079040, 0.479308067841920),
        (0.869739794195568, 0.065130102902216, 0.065130102902216),
        (0.065130102902216, 0.869739794195568, 0.065130102902216),
        (0.065130102902216, 0.065130102902216, 0.869739794195568),
        (0.048690315425316, 0.312865496004875, 0.638444188569809),
        (0.048690315425316, 0.638444188569809, 0.312865496004875),
        (0.312865496004875, 0.048690315425316, 0.638444188569809),
        (0.312865496004875, 0.638444188569809, 0.048690315425316),
        (0.638444188569809, 0.048690315425316, 0.312865496004875),
        (0.638444188569809, 0.312865496004875, 0.048690315425316),
    ],
    [
        -0.074785022233835,
         0.087807628716602,  0.087807628716602,  0.087807628716602,
         0.026673617804419,  0.026673617804419,  0.026673617804419,
         0.038556880445129,  0.038556880445129,  0.038556880445129,
         0.038556880445129,  0.038556880445129,  0.038556880445129,
    ],
)

# degree 8, 16 points (Dunavant). Weights from Dunavant (1985) Table II
# scaled by 1/2 so they sum to the reference triangle area 1/2.
_RULES[8] = (
    [
        (1/3, 1/3, 1/3),
        (0.081414823414554, 0.459292588292723, 0.459292588292723),
        (0.459292588292723, 0.081414823414554, 0.459292588292723),
        (0.459292588292723, 0.459292588292723, 0.081414823414554),
        (0.658861384496480, 0.170569307751760, 0.170569307751760),
        (0.170569307751760, 0.658861384496480, 0.170569307751760),
        (0.170569307751760, 0.170569307751760, 0.658861384496480),
        (0.898905543365938, 0.050547228317031, 0.050547228317031),
        (0.050547228317031, 0.898905543365938, 0.050547228317031),
        (0.050547228317031, 0.050547228317031, 0.898905543365938),
        (0.008394777409958, 0.263112829634638, 0.728492392955404),
        (0.008394777409958, 0.728492392955404, 0.263112829634638),
        (0.263112829634638, 0.008394777409958, 0.728492392955404),
        (0.263112829634638, 0.728492392955404, 0.008394777409958),
        (0.728492392955404, 0.008394777409958, 0.263112829634638),
        (0.728492392955404, 0.263112829634638, 0.008394777409958),
    ],
    [
        0.072157803838894,
        0.047545817133643, 0.047545817133643, 0.047545817133643,
        0.051608685267359, 0.051608685267359, 0.051608685267359,
        0.016229248811599, 0.016229248811599, 0.016229248811599,
        0.013615157087218, 0.013615157087218, 0.013615157087218,
        0.013615157087218, 0.013615157087218, 0.013615157087218,
    ],
)

"""
    reference_rule(degree) -> (bary_points, weights)

Return a barycentric Gauss rule on the reference triangle (vertices
(0,0), (1,0), (0,1)) that integrates polynomials of total degree
`degree` exactly. Weights sum to 1/2 (the triangle's area).

The returned `bary_points` and `weights` arrays alias the internal rule
table — callers must treat them as read-only.
"""
function reference_rule(degree::Int)
    haskey(_RULES, degree) && return _RULES[degree]
    max_degree = maximum(keys(_RULES))
    for d in (degree + 1):max_degree
        haskey(_RULES, d) && return _RULES[d]
    end
    error("Triangle quadrature degree $degree not available (max is $max_degree).")
end

"""
    fan_triangulate(verts) -> Vector{NTuple{3, T}}

Fan triangulate a polygon with vertices `verts` (an iterable of points,
not closed) from its arithmetic centroid. Each output triangle is
`(centroid, vᵢ, vᵢ₊₁)` where indices wrap.

For spherical use, the caller is responsible for projecting the centroid
onto the sphere if needed; this function uses the arithmetic mean as-is.
"""
function fan_triangulate(verts)
    n = length(verts)
    n < 3 && return Tuple{eltype(verts), eltype(verts), eltype(verts)}[]
    cx = sum(v[1] for v in verts) / n
    cy = sum(v[2] for v in verts) / n
    centroid = (cx, cy)
    return [(centroid, verts[i], verts[mod1(i + 1, n)]) for i in 1:n]
end

"""
    planar_triangle_area(t) -> Float64

Unsigned planar area of a triangle `t = (v₁, v₂, v₃)` where each `vₖ`
is a 2-tuple `(x, y)`.
"""
function planar_triangle_area(t)
    v₁, v₂, v₃ = t
    return 0.5 * abs((v₂[1] - v₁[1]) * (v₃[2] - v₁[2]) -
                     (v₃[1] - v₁[1]) * (v₂[2] - v₁[2]))
end

end # module TriangleQuadrature
