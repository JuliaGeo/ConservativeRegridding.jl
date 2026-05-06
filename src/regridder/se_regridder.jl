"""
    SEtoFVRegridder{W,A,V} <: AbstractRegridder

Regridder from a spectral element (SE) source grid to a finite volume (FV)
destination grid, built on the principled (polygon-intersection) approach
of PDF Appendix A.

The `weight_matrix` is a sparse matrix of size `(N_fv, N_se_nodes)` whose
entries are

    B(k, (e, i, j))  =  ∫_{k ∩ e}  ϕᵢ(ξ) ϕⱼ(η)  dA_phys ,

i.e. the integral of the source basis function `ϕᵢϕⱼ` over the physical
intersection of FV cell `k` and SE element `e`. (The Jacobian factor in
PDF Eq. 18 cancels via change of variables.) Regridding computes
`dst = (weight_matrix * src) ./ dst_areas`. Conservation is exact:
`Σ_k dst[k] · A_dst,k = Σ_{e,i,j} (Σ_k B(k,(e,i,j))) f_src,e,i,j`.
"""
struct SEtoFVRegridder{W, A, V} <: AbstractRegridder
    weight_matrix :: W
    dst_areas :: A
    dst_temp :: V
    src_temp :: V
end

"""
    FVtoSERegridder{W,V} <: AbstractRegridder

Regridder from a finite volume (FV) source grid to a spectral element (SE)
destination grid, built as a per-element L2 projection.

The `weight_matrix` is a sparse matrix of size `(N_se_nodes, N_fv)` whose
entries already include the per-element inverse mass matrix:

    weight_matrix[(e, i, j), k]  =  Σ_{a, b}  (M^{e})⁻¹[(i,j), (a,b)] · B(k, (e, a, b))

where `M^{e}_{(i,j),(a,b)} = ∫_{e} ϕᵢ ϕⱼ ϕₐ ϕᵦ Jᵉ dξ dη` is the local mass
matrix (rescaled per row to enforce `M^{e} · 1 = (Σ_k B)|_e` exactly via
partition of unity, eliminating quadrature mismatch between `B` and `M^{e}`
that otherwise causes ~1e-5 deviations on constants).

Regridding is just `dst = weight_matrix * src`; constants are preserved to
machine precision and the projection is mass-conservative. The result is
not automatically continuous across element boundaries — the field-level
`regrid!` in the ClimaCore extension applies weighted DSS.
"""
struct FVtoSERegridder{W, V} <: AbstractRegridder
    weight_matrix :: W
    dst_temp :: V
    src_temp :: V
end

Base.size(r::SEtoFVRegridder, args...) = size(r.weight_matrix, args...)
Base.size(r::FVtoSERegridder, args...) = size(r.weight_matrix, args...)

# ──────────────────────────────────────────────────────────
# SE → FV:  dst[k] = (Σ_{e,i,j} B(k,(e,i,j)) · f_src[e,i,j]) / A_dst,k
# ──────────────────────────────────────────────────────────

function regrid!(dst_field::DenseVector, regridder::SEtoFVRegridder, src_field::DenseVector)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    dst_field ./= regridder.dst_areas
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::SEtoFVRegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(regridder.dst_temp, regridder, regridder.src_temp)
    dst_field .= regridder.dst_temp
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::SEtoFVRegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::SEtoFVRegridder, src_field::DenseVector)
    regrid!(regridder.dst_temp, regridder, src_field)
    dst_field .= regridder.dst_temp
    return dst_field
end

# ──────────────────────────────────────────────────────────
# FV → SE:  dst[(e,i,j)] = (W * src)[(e,i,j)]   (M⁻¹ already baked in)
# ──────────────────────────────────────────────────────────

function regrid!(dst_field::DenseVector, regridder::FVtoSERegridder, src_field::DenseVector)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::FVtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(regridder.dst_temp, regridder, regridder.src_temp)
    dst_field .= regridder.dst_temp
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::FVtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::FVtoSERegridder, src_field::DenseVector)
    regrid!(regridder.dst_temp, regridder, src_field)
    dst_field .= regridder.dst_temp
    return dst_field
end
