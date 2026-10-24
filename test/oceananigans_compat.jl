# Oceananigans 0.115 changed TripolarGrid's fold_topology keyword from a full
# grid topology to a side topology. Keep fixtures usable on older releases too.
const TripolarFaceFold = isdefined(Oceananigans.Grids, :FaceFolded) ?
    Oceananigans.Grids.FaceFolded : Oceananigans.Grids.RightFaceFolded
const TripolarCenterFold = isdefined(Oceananigans.Grids, :CenterFolded) ?
    Oceananigans.Grids.CenterFolded : Oceananigans.Grids.RightCenterFolded
