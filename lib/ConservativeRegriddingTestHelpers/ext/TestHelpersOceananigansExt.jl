module TestHelpersOceananigansExt

import Oceananigans

import ConservativeRegridding
import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_rotated(field::Oceananigans.Field) = field.grid isa Oceananigans.RotatedLatitudeLongitudeGrid

TestHelpers.has_tripolar(field::Oceananigans.Field) = field.grid isa Oceananigans.TripolarGrid


# looked up per call: the extensions load in no fixed order
OcExt() = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)

# the cells CR's native path regrids: wet cells of a masked grid, the
# non-duplicated cells of a U-pivot tripolar grid, every cell otherwise
function TestHelpers.grid_of(grid::Oceananigans.Grids.AbstractGrid)
    g = OcExt().structured_grid(grid)
    g isa OcExt().OceananigansCurvilinearGrid && g.fold === :center && return OcExt().unique_cells(g)
    return g
end
TestHelpers.grid_of(grid::Oceananigans.ImmersedBoundaryGrid) = OcExt().wet_cells(grid)
TestHelpers.grid_of(grid::Oceananigans.MultiRegion.MultiRegionGrid) = OcExt().structured_grid(grid)
TestHelpers.grid_of(field::Oceananigans.AbstractField) = TestHelpers.grid_of(field.grid)

end
