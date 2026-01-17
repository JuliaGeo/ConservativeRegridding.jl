# WARNING This will not work 
# We need to hook up the speedy weather grids
# to some sort of tree structure
# Stay tuned!
using SpeedyWeather, GeoMakie
using ConservativeRegridding
import GeoInterface as GI, GeometryOps as GO

field1 = rand(OctaHEALPixGrid, 24)
field2 = rand(OctaminimalGaussianGrid, 24)

SpeedyWeatherGeoMakieExt = Base.get_extension(SpeedyWeather, :SpeedyWeatherGeoMakieExt)
faces1 = SpeedyWeatherGeoMakieExt.get_faces(field1)
faces2 = SpeedyWeatherGeoMakieExt.get_faces(field2)

polys1 = GI.Polygon.(GI.LinearRing.(eachcol(faces1))) .|> GO.fix
polys2 = GI.Polygon.(GI.LinearRing.(eachcol(faces2))) .|> GO.fix

R = ConservativeRegridding.Regridder(polys1, polys2)

ConservativeRegridding.regrid!(field1, R, field2)
ConservativeRegridding.regrid!(field2, transpose(R), field1)