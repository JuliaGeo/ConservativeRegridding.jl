module TestHelpersRingGridsExt

import RingGrids

import ConservativeRegridding
import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_full_ring_grid(field::RingGrids.AbstractField) = TestHelpers.has_full_ring_grid(field.grid)
TestHelpers.has_full_ring_grid(field::RingGrids.AbstractFullGrid) = true

TestHelpers.has_cell_crossing_dateline(field::RingGrids.AbstractField) = TestHelpers.has_cell_crossing_dateline(field.grid)
TestHelpers.has_cell_crossing_dateline(field::RingGrids.AbstractFullGrid) = true
TestHelpers.has_cell_crossing_dateline(field::RingGrids.HEALPixGrid) = true 
TestHelpers.has_cell_crossing_dateline(field::RingGrids.OctaHEALPixGrid) = false # OctaHEALPix faces line up with meridians sometimes

# looked up per call: the extensions load in no fixed order
RGExt() = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

TestHelpers.grid_of(grid::RingGrids.AbstractGrid) = RGExt().ringgrid(grid)
TestHelpers.grid_of(field::RingGrids.AbstractField) = RGExt().ringgrid(field.grid)

end
