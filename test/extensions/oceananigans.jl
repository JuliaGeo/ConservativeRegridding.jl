using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using Oceananigans

const OceananigansExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)