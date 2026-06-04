module TestHelpersRingGridsExt

import RingGrids

import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_full_ring_grid(field::RingGrids.AbstractField) = field.grid isa RingGrids.AbstractFullGrid

end
