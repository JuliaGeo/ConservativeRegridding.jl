module TestHelpersOceananigansExt

import Oceananigans

import ConservativeRegriddingTestHelpers as TestHelpers

TestHelpers.has_rotated(field::Oceananigans.Field) = field.grid isa Oceananigans.RotatedLatitudeLongitudeGrid

TestHelpers.has_tripolar(field::Oceananigans.Field) = field.grid isa Oceananigans.TripolarGrid

end