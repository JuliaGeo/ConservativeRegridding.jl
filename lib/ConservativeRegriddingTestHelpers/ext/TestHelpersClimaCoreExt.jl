module TestHelpersClimaCoreExt

import ClimaCore
import ConservativeRegridding

import GeometryOps as GO

# Resolve ConservativeRegridding's own ClimaCore extension at *runtime*. Baking it
# into a `const` here captures `nothing`, because that sibling extension may not be
# loaded yet while this extension is precompiling; by call time it always is.
cr_climacore_ext() = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_spectral_element(field::ClimaCore.Fields.Field) = true

"""
    set_field_values!(field::ClimaCore.Fields.Field, values, fun; kwargs...)

For spectral element fields, sample `fun` at each nodal position instead of integrating over a cell polygon.
"""
function TestHelpers.set_field_values!(field::ClimaCore.Fields.Field, values, fun; kwargs...)
    positions = cr_climacore_ext().se_node_positions(getfield(field, :space))
    values .= Iterators.map(positions) do p
        fun((GO.UnitSpherical.GeographicFromUnitSphere()(p))...)
    end
end

function TestHelpers.test_integration_weights(field::ClimaCore.Fields.Field, regridder)
    cr_climacore_ext().se_node_weights(getfield(field, :space))
end

end