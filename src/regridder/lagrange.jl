module Lagrange

"""
    evaluate(ξs, i, ξ) -> eltype(ξs)

Evaluate the i-th Lagrange basis polynomial defined on nodes `ξs` at the point `ξ`:

    ϕᵢ(ξ) = Π_{p ≠ i} (ξ - ξ_p) / (ξ_i - ξ_p)

`ξs` is typically the GLL node vector for the SE space.
"""
@inline function evaluate(ξs, i, ξ)
    Nq = length(ξs)
    ϕ = one(eltype(ξs))
    ξᵢ = ξs[i]
    @inbounds for p in 1:Nq
        p == i && continue
        ϕ *= (ξ - ξs[p]) / (ξᵢ - ξs[p])
    end
    return ϕ
end

"""
    evaluate_all!(out, ξs, ξ) -> out

Fill the preallocated buffer `out` with `[ϕ₁(ξ), ϕ₂(ξ), …, ϕ_Nq(ξ)]`.
"""
function evaluate_all!(out, ξs, ξ)
    Nq = length(ξs)
    @inbounds for i in 1:Nq
        out[i] = evaluate(ξs, i, ξ)
    end
    return out
end

"""
    evaluate_all(ξs, ξ) -> Vector

Return `[ϕ₁(ξ), ϕ₂(ξ), …, ϕ_Nq(ξ)]`. Allocating wrapper around [`evaluate_all!`](@ref).
"""
evaluate_all(ξs, ξ) = evaluate_all!(Vector{eltype(ξs)}(undef, length(ξs)), ξs, ξ)

end # module Lagrange
