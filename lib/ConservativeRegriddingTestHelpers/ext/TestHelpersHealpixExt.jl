module TestHelpersHealpixExt

import Healpix
import ConservativeRegridding
import ConservativeRegriddingTestHelpers as TestHelpers

# looked up per call: the extensions load in no fixed order
HPExt() = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)

TestHelpers.grid_of(map::Healpix.HealpixMap) = HPExt().HealpixMapGrid(map)

end
