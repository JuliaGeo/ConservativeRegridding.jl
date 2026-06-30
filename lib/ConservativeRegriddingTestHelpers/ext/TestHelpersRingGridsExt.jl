module TestHelpersRingGridsExt

import RingGrids

import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_full_ring_grid(field::RingGrids.AbstractField) = field.grid isa RingGrids.AbstractFullGrid

TestHelpers.has_cell_crossing_dateline(field::RingGrids.AbstractFullGrid) = true
TestHelpers.has_cell_crossing_dateline(field::RingGrids.HEALPixGrid) = true 
TestHelpers.has_cell_crossing_dateline(field::RingGrids.OctaHEALPixGrid) = false # OctaHEALPix faces line up with meridians sometimes
end
